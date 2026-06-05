"""Response context compressor for Responses API compact endpoint.

Takes the full conversation history (original request input) and uses the
upstream model to produce a concise summary, then returns a new compacted
response object.

This is the Phase B implementation.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from backend.adapter.standard_request import OPENCLAW_OPENAI_PROFILE, StandardRequest
from backend.runtime.execution import collect_completion_run
from backend.services.responses_formatters import new_response_id, _usage

log = logging.getLogger("qwen2api.response_compressor")

# ── Compression prompt ────────────────────────────────────────────────────────

_COMPRESS_SYSTEM = (
    "You are a context compression assistant. Your task is to produce a concise\n"
    "structured summary of the conversation below. The summary will replace the\n"
    "full conversation history in subsequent requests to save tokens.\n"
    "\n"
    "Rules:\n"
    "1. Preserve ALL factual details: names, file paths, code snippets, URLs,\n"
    "   error messages, command outputs, and technical decisions.\n"
    "2. Preserve tool/function call names, their arguments, and key results.\n"
    "3. Preserve the user's original goal and any explicit requirements.\n"
    "4. Remove pleasantries, filler, and redundant phrasing.\n"
    "5. Use bullet points and short paragraphs for readability.\n"
    "6. If the conversation includes code, keep the essential code intact.\n"
    "7. Output ONLY the summary text. No preamble like 'Here is the summary'."
)


def _serialize_conversation(original_request: dict[str, Any]) -> str:
    """Convert the original request's input into a readable text format."""
    raw_input = original_request.get("input", "")
    instructions = original_request.get("instructions", "")

    parts: list[str] = []
    if instructions:
        parts.append(f"[System Instructions]\n{instructions}\n")

    if isinstance(raw_input, str):
        parts.append(f"[User]\n{raw_input}")
    elif isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, str):
                parts.append(f"[User]\n{item}")
            elif isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            t = part.get("text") or part.get("input_text") or part.get("output_text") or ""
                            if t:
                                text_parts.append(t)
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "\n".join(text_parts)
                item_type = item.get("type", "")
                if item_type == "function_call":
                    name = item.get("name", "")
                    args = item.get("arguments", "{}")
                    parts.append(f"[Function Call] {name}({args})")
                elif item_type == "function_call_output":
                    output = item.get("output", "")
                    if isinstance(output, str) and len(output) > 2000:
                        output = output[:2000] + "...[truncated]"
                    parts.append(f"[Function Result]\n{output}")
                else:
                    parts.append(f"[{role.title()}]\n{content}")
    return "\n\n".join(parts)


def _build_compress_request(
    conversation_text: str,
    model: str = "qwen3.6-plus",
) -> tuple[StandardRequest, str]:
    """Build a StandardRequest + prompt for the compression call."""
    prompt = (
        f"{_COMPRESS_SYSTEM}\n\n"
        f"---\n\n"
        f"Conversation to compress:\n\n{conversation_text}\n\n"
        f"---\n\n"
        f"Produce the compressed summary now."
    )
    request = StandardRequest(
        prompt=prompt,
        response_model=model,
        resolved_model=model,
        surface="responses_compact",
        client_profile=OPENCLAW_OPENAI_PROFILE,
        stream=False,
    )
    return request, prompt


async def compress_conversation(
    *,
    client,
    original_request: dict[str, Any],
    original_response: dict[str, Any],
    account_pool=None,
    preferred_account_email: str | None = None,
) -> dict[str, Any]:
    """Run upstream compression and return a compacted response object.

    Parameters
    ----------
    client : QwenClient
        The upstream client instance.
    original_request : dict
        The body of the original POST /v1/responses request.
    original_response : dict
        The stored response payload.
    account_pool : AccountPool, optional
        Account pool for acquiring a token. Falls back to client.account_pool.
    preferred_account_email : str, optional
        Preferred account email for affinity.

    Returns
    -------
    dict
        A new response object with compressed output and summary as assistant message.
    """
    pool = account_pool or getattr(client, "account_pool", None)
    if pool is None:
        raise RuntimeError("No account pool available for compression")

    # Acquire an account
    acc = None
    if preferred_account_email:
        acc = await pool.acquire_wait_preferred(preferred_account_email, timeout=30)
    if acc is None:
        acc = await pool.acquire_wait(timeout=30)
    if acc is None:
        raise RuntimeError("No available upstream account for compression")

    model = original_response.get("model", "qwen3.6-plus")
    conversation_text = _serialize_conversation(original_request)

    if len(conversation_text.strip()) < 100:
        log.info("[Compressor] conversation too short (%d chars), skipping compression", len(conversation_text))
        # Return a minimal compacted response without calling upstream
        new_id = new_response_id()
        return _build_compacted_response(
            new_id=new_id,
            original_response=original_response,
            summary_text=conversation_text,
            input_tokens=len(conversation_text) // 4,
            output_tokens=0,
        )

    request, prompt = _build_compress_request(conversation_text, model=model)
    request.bound_account = acc
    request.bound_account_email = getattr(acc, "email", None)

    log.info(
        "[Compressor] starting compression account=%s model=%s input_len=%d",
        getattr(acc, "email", "-"), model, len(conversation_text),
    )

    from backend.runtime.execution import cleanup_runtime_resources

    try:
        execution = await collect_completion_run(client, request, prompt)
        summary_text = (execution.state.answer_text or "").strip()

        # Release upstream resources (account, chat)
        try:
            await cleanup_runtime_resources(client, execution.acc, execution.chat_id, preserve_chat=False)
        except Exception as cleanup_exc:
            log.warning("[Compressor] cleanup failed: %s", cleanup_exc)

        if not summary_text:
            log.warning("[Compressor] upstream returned empty summary, using fallback")
            summary_text = _fallback_truncate(conversation_text)

        input_tokens = max(1, len(conversation_text) // 4)
        output_tokens = max(1, len(summary_text) // 4)

        new_id = new_response_id()
        result = _build_compacted_response(
            new_id=new_id,
            original_response=original_response,
            summary_text=summary_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        log.info(
            "[Compressor] done new_id=%s summary_len=%d in_tok=%d out_tok=%d",
            new_id, len(summary_text), input_tokens, output_tokens,
        )
        return result

    except Exception as exc:
        log.exception("[Compressor] upstream compression failed, using fallback truncation")
        summary_text = _fallback_truncate(conversation_text)
        new_id = new_response_id()
        return _build_compacted_response(
            new_id=new_id,
            original_response=original_response,
            summary_text=summary_text,
            input_tokens=len(conversation_text) // 4,
            output_tokens=0,
        )


def _fallback_truncate(text: str, max_chars: int = 4000) -> str:
    """Last-resort truncation when upstream is unavailable."""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars * 2 // 3]
    tail = text[-max_chars // 3 :]
    return f"{head}\n\n[...compressed by truncation...]\n\n{tail}"


def _build_compacted_response(
    *,
    new_id: str,
    original_response: dict[str, Any],
    summary_text: str,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    """Assemble the compacted response object."""
    return {
        "id": new_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": original_response.get("instructions"),
        "model": original_response.get("model", ""),
        "output": [{
            "id": f"msg_{new_id[5:]}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": summary_text, "annotations": []}],
        }],
        "output_text": summary_text,
        "parallel_tool_calls": original_response.get("parallel_tool_calls", True),
        "previous_response_id": original_response.get("id"),
        "store": original_response.get("store", False),
        "tools": original_response.get("tools", []),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "max_output_tokens": original_response.get("max_output_tokens"),
        "truncation": original_response.get("truncation", "disabled"),
    }
