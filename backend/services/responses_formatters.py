from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _usage(prompt: str, output_text: str, reasoning_text: str = "") -> dict[str, Any]:
    input_tokens = max(1, len(prompt or "") // 4)
    output_tokens = max(1, len(output_text or "") // 4)
    reasoning_tokens = max(0, len(reasoning_text or "") // 4) if reasoning_text else 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens + reasoning_tokens,
        "total_tokens": input_tokens + output_tokens + reasoning_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


def _base_response_obj(
    *,
    response_id: str,
    created_at: int,
    model_name: str,
    status: str = "in_progress",
    output: list | None = None,
    output_text: str = "",
    usage: dict | None = None,
    request_payload: dict | None = None,
) -> dict[str, Any]:
    rp = request_payload or {}
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": rp.get("instructions"),
        "model": model_name,
        "output": output if output is not None else [],
        "output_text": output_text,
        "parallel_tool_calls": bool(rp.get("parallel_tool_calls", True)),
        "previous_response_id": rp.get("previous_response_id"),
        "store": bool(rp.get("store", False)),
        "tools": rp.get("tools", []),
        "usage": usage or _usage("", ""),
        "max_output_tokens": rp.get("max_output_tokens") or rp.get("max_completion_tokens"),
        "truncation": rp.get("truncation", "disabled"),
    }


def _tool_blocks_to_function_call_items(tool_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert parsed tool blocks to Responses API function_call output items."""
    items = []
    for block in tool_blocks:
        if block.get("type") != "tool_use":
            continue
        call_id = block.get("id") or f"call_{uuid.uuid4().hex[:24]}"
        name = block.get("name", "")
        raw_input = block.get("input", {})
        if isinstance(raw_input, str):
            arguments = raw_input
        else:
            arguments = json.dumps(raw_input, ensure_ascii=False)
        items.append({
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed",
        })
    return items



def _strip_tool_call_blocks(text: str) -> str:
    """[ADDED 2026-06-05] Remove ##TOOL_CALL##...##END_CALL## and <tool_call>
    blocks from answer text. When the response contains structured function_call
    output items, the raw text markers should not leak into output_text."""
    if not text:
        return text
    cleaned = re.sub(r'##TOOL_CALL##[\s\S]*?(?:##END_CALL##|$)', '', text)
    cleaned = re.sub(r'<tool_call>[\s\S]*?(?:</tool_call>|$)', '', cleaned)
    cleaned = re.sub(r'<tool_calls>[\s\S]*?(?:</tool_calls>|$)', '', cleaned)
    cleaned = re.sub(r'<\|QNML\|tool_calls>[\s\S]*?(?:</\|QNML\|tool_calls>|$)', '', cleaned)
    return cleaned.strip()


def build_responses_payload(
    *,
    response_id: str,
    created_at: int | None = None,
    model_name: str,
    prompt: str,
    execution,
    standard_request,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created = created_at if created_at is not None else int(time.time())
    request_payload = request_payload or {}
    from backend.runtime.execution import build_tool_directive
    directive = build_tool_directive(standard_request, execution.state)
    output_text = execution.state.answer_text or ""
    reasoning_text = getattr(execution.state, "reasoning_text", "") or ""

    output: list[dict[str, Any]] = []

    # Emit reasoning output item first if thinking content exists
    if reasoning_text.strip():
        output.append({
            "id": f"rs_{uuid.uuid4().hex[:24]}",
            "type": "reasoning",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": reasoning_text}],
        })

    if directive.stop_reason == "tool_use":
        # Convert tool blocks to function_call output items
        output.extend(_tool_blocks_to_function_call_items(directive.tool_blocks))
        # Strip raw ##TOOL_CALL## markers from output_text
        output_text = _strip_tool_call_blocks(output_text)
    else:
        output.append({
            "id": new_message_id(),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": output_text, "annotations": []}],
        })

    incomplete_reason = getattr(execution.state, "incomplete_reason", None)
    payload = _base_response_obj(
        response_id=response_id,
        created_at=created,
        model_name=model_name,
        status="incomplete" if incomplete_reason else "completed",
        output=output,
        output_text=output_text,
        usage=_usage(prompt, output_text, reasoning_text),
        request_payload=request_payload,
    )
    if incomplete_reason:
        payload["incomplete_details"] = {"reason": incomplete_reason}
    return payload


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ResponsesStreamTranslator:
    """Emits SSE events matching OpenAI Responses API spec for Codex/Continue.dev.

    Supports both text output and function_call output items.
    Event sequence:
      response.created -> response.in_progress -> response.output_item.added
      -> response.content_part.added -> (N x response.output_text.delta)
      -> response.content_part.done -> response.output_text.done
      -> response.output_item.done -> response.completed

    For function calls:
      response.output_item.added (function_call)
      -> response.function_call_arguments.delta
      -> response.function_call_arguments.done
      -> response.output_item.done
    """

    def __init__(
        self,
        *,
        response_id: str,
        created_at: int,
        model_name: str,
        request_payload: dict[str, Any] | None = None,
        prompt: str = "",
    ):
        self.response_id = response_id
        self.created_at = created_at
        self.model_name = model_name
        self.request_payload = request_payload or {}
        self.prompt = prompt
        self.pending_chunks: list[str] = []
        self.output_index = 0
        self.content_index = 0
        self.item_id = new_message_id()
        self.started = False
        self.text_started = False
        self.answer_fragments: list[str] = []
        self.tool_calls: list = []
        # Buffer text to suppress ##TOOL_CALL## leakage
        self._toolish_buffer: list[str] = []
        # Reasoning (thinking) output tracking
        self._reasoning_started: bool = False
        self._reasoning_item_id: str = f"rs_{uuid.uuid4().hex[:24]}"
        self.reasoning_fragments: list[str] = []

    def _response_obj(self, *, status: str = "in_progress") -> dict[str, Any]:
        return _base_response_obj(
            response_id=self.response_id,
            created_at=self.created_at,
            model_name=self.model_name,
            status=status,
            request_payload=self.request_payload,
        )

    def _wrap(self, event_type: str, data: dict[str, Any]) -> str:
        data["type"] = event_type
        return sse_event(event_type, data)

    def _ensure_started(self):
        if not self.started:
            self.pending_chunks.append(
                self._wrap("response.created", {"response": self._response_obj(status="in_progress")})
            )
            self.pending_chunks.append(
                self._wrap("response.in_progress", {"response": self._response_obj(status="in_progress")})
            )
            self.started = True

    def start(self):
        """Backward-compatible explicit stream start helper."""
        self._ensure_started()

    def on_delta(self, evt: dict[str, Any], text_chunk: str | None, tool_calls: list[dict[str, Any]] | None):
        """Compatibility shim used by older tests/callers."""
        phase = evt.get("phase", "")
        if phase in ("think", "thinking_summary") and text_chunk:
            self.on_reasoning_chunk(text_chunk)
            return
        if tool_calls:
            for tool_call in tool_calls:
                self.on_tool_call(tool_call)
            return
        if text_chunk:
            self.on_text_chunk(text_chunk)

    def _ensure_text_item(self):
        if not self.text_started:
            self._ensure_started()
            self.pending_chunks.append(self._wrap("response.output_item.added", {
                "output_index": self.output_index,
                "item": {
                    "id": self.item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            }))
            self.pending_chunks.append(self._wrap("response.content_part.added", {
                "item_id": self.item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }))
            self.text_started = True

    def _ensure_reasoning_item(self):
        """Emit a reasoning output item (OpenAI Responses API thinking output)."""
        if self._reasoning_started:
            return
        self._ensure_started()
        reasoning_item = {
            "id": self._reasoning_item_id,
            "type": "reasoning",
            "status": "in_progress",
            "summary": [],
        }
        self.pending_chunks.append(self._wrap("response.output_item.added", {
            "output_index": self.output_index,
            "item": reasoning_item,
        }))
        self._reasoning_started = True
        self.output_index += 1

    def on_reasoning_chunk(self, text: str):
        """Emit a reasoning summary delta for thinking content."""
        if not text:
            return
        self._ensure_reasoning_item()
        self.reasoning_fragments.append(text)
        self.pending_chunks.append(self._wrap("response.reasoning_summary_text.delta", {
            "item_id": self._reasoning_item_id,
            "output_index": self.output_index - 1,
            "summary_index": 0,
            "delta": text,
        }))

    def _finalize_reasoning(self):
        """Close the reasoning output item if it was started."""
        if not self._reasoning_started:
            return
        reasoning_text = "".join(self.reasoning_fragments)
        self.pending_chunks.append(self._wrap("response.reasoning_summary_text.done", {
            "item_id": self._reasoning_item_id,
            "output_index": self.output_index - 1,
            "summary_index": 0,
            "text": reasoning_text,
        }))
        self.pending_chunks.append(self._wrap("response.output_item.done", {
            "output_index": self.output_index - 1,
            "item": {
                "id": self._reasoning_item_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning_text}],
            },
        }))

    def _flush_text_buffer(self):
        """Flush non-tool-call text from the buffer as real-time deltas.

        Scans the buffer for tool-call markers. If none are found, emits the
        buffered text as an output_text.delta and clears the buffer.
        Returns True if text was flushed, False if still buffering (possible
        tool call in progress).
        """
        if not self._toolish_buffer:
            return True
        combined = "".join(self._toolish_buffer)
        if self._TOOL_MARKER_RE.search(combined):
            # Tool-like content detected — keep buffering
            return False
        # Clean text — emit as delta
        self._ensure_text_item()
        self.answer_fragments.append(combined)
        self.pending_chunks.append(self._wrap("response.output_text.delta", {
            "item_id": self.item_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "delta": combined,
        }))
        self._toolish_buffer.clear()
        return True

    _TOOL_MARKER_RE = re.compile(r"##TOOL_CALL##|<tool_call>|<tool_calls>|<invoke>|<parameter>|<\|QNML\|", re.IGNORECASE)

    def on_text_chunk(self, text_chunk: str):
        """Buffer text and flush in real-time when safe.

        Text is appended to a scan buffer. After each chunk we check whether
        the buffer contains tool-call markers. If not, we emit the text as
        an output_text.delta immediately for real-time streaming.
        Tool-call markers are kept buffered until finalize() decides the
        final disposition.
        """
        if not text_chunk:
            return
        self._ensure_started()
        self._toolish_buffer.append(text_chunk)
        self._flush_text_buffer()

    def on_tool_call(self, tool_call: dict):
        self.tool_calls.append(tool_call)

    def emit_function_call(self, tool_block: dict[str, Any]):
        """Emit a function_call output item for a parsed tool block."""
        self._ensure_started()
        call_id = tool_block.get("id") or f"call_{uuid.uuid4().hex[:24]}"
        name = tool_block.get("name", "")
        raw_input = tool_block.get("input", {})
        arguments = json.dumps(raw_input, ensure_ascii=False) if not isinstance(raw_input, str) else raw_input
        fc_id = f"fc_{uuid.uuid4().hex[:24]}"

        fc_item = {
            "id": fc_id,
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed",
        }

        self.pending_chunks.append(self._wrap("response.output_item.added", {
            "output_index": self.output_index,
            "item": {
                "id": fc_id,
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": "",
                "status": "in_progress",
            },
        }))

        self.pending_chunks.append(self._wrap("response.function_call_arguments.delta", {
            "item_id": fc_id,
            "output_index": self.output_index,
            "delta": arguments,
        }))

        self.pending_chunks.append(self._wrap("response.function_call_arguments.done", {
            "item_id": fc_id,
            "output_index": self.output_index,
            "arguments": arguments,
        }))

        self.pending_chunks.append(self._wrap("response.output_item.done", {
            "output_index": self.output_index,
            "item": fc_item,
        }))
        self.output_index += 1

    def drain(self) -> list[str]:
        chunks = list(self.pending_chunks)
        self.pending_chunks.clear()
        return chunks

    def fail(self, error: dict | str) -> list[str]:
        self._ensure_started()
        error_obj = error if isinstance(error, dict) else {"message": str(error)}
        resp_obj = self._response_obj(status="failed")
        resp_obj["error"] = error_obj
        resp_obj["incomplete_details"] = {"reason": "error"}
        chunks = self.drain()
        chunks.append(self._wrap("response.failed", {"response": resp_obj}))
        chunks.append(self._wrap("response.completed", {"response": resp_obj}))
        chunks.append("data: [DONE]\n\n")
        return chunks

    def finalize(self, *, payload: dict[str, Any] | None = None, tool_blocks: list[dict[str, Any]] | None = None) -> list[str]:
        self._ensure_started()
        final_payload = payload or {}
        output_text = final_payload.get("output_text") or "".join(self.answer_fragments)
        if tool_blocks is None:
            payload_tool_blocks: list[dict[str, Any]] = []
            for item in final_payload.get("output") or []:
                if not isinstance(item, dict) or item.get("type") != "function_call":
                    continue
                args = item.get("arguments", "{}")
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) else (args or {})
                except (TypeError, ValueError):
                    parsed_args = args if isinstance(args, str) else {}
                payload_tool_blocks.append({
                    "type": "tool_use",
                    "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "name": item.get("name", ""),
                    "input": parsed_args,
                })
            if payload_tool_blocks:
                tool_blocks = payload_tool_blocks

        chunks = self.drain()

        # If tool calls, emit function_call items instead of text completion
        if tool_blocks:
            # Close reasoning item first if any
            self._finalize_reasoning()
            # Discard buffered tool-like text
            self._toolish_buffer.clear()
            for block in tool_blocks:
                if block.get("type") == "tool_use":
                    self.emit_function_call(block)
            output_text = _strip_tool_call_blocks(output_text)
            chunks.extend(self.drain())
        else:
            # Close reasoning item first if any
            self._finalize_reasoning()
            # Flush any remaining buffered text
            self._flush_text_buffer()
            self._ensure_text_item()
            # If there is still leftover in buffer (tool-call content that
            # turned out to be false positive), flush it anyway.
            if self._toolish_buffer:
                leftover = "".join(self._toolish_buffer)
                self._toolish_buffer.clear()
                self.answer_fragments.append(leftover)
                self.pending_chunks.append(self._wrap("response.output_text.delta", {
                    "item_id": self.item_id, "output_index": self.output_index,
                    "content_index": self.content_index, "delta": leftover,
                }))
            output_text = "".join(self.answer_fragments)
            chunks.extend(self.drain())
            chunks.append(self._wrap("response.content_part.done", {
                "item_id": self.item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "part": {"type": "output_text", "text": output_text, "annotations": []},
            }))
            chunks.append(self._wrap("response.output_text.done", {
                "item_id": self.item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "text": output_text,
            }))
            chunks.append(self._wrap("response.output_item.done", {
                "output_index": self.output_index,
                "item": {
                    "id": self.item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": output_text, "annotations": []}],
                },
            }))

        final_payload = dict(final_payload)
        final_payload.setdefault("id", self.response_id)
        final_payload.setdefault("object", "response")
        final_payload.setdefault("created_at", self.created_at)
        final_payload.setdefault("model", self.model_name)
        final_payload.setdefault("status", "completed")
        final_payload["output_text"] = output_text

        if not final_payload.get("usage"):
            reasoning_text = "".join(self.reasoning_fragments)
            final_payload["usage"] = _usage(self.prompt, output_text, reasoning_text)

        chunks.append(self._wrap("response.completed", {"response": final_payload}))
        chunks.append("data: [DONE]\n\n")
        return chunks
