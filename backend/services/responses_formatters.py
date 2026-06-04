from __future__ import annotations

import json
import time
import uuid
from typing import Any


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _usage(prompt: str, output_text: str) -> dict[str, Any]:
    input_tokens = max(1, len(prompt or "") // 4)
    output_tokens = max(1, len(output_text or "") // 4)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
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

    if directive.stop_reason == "tool_use":
        # Convert tool blocks to function_call output items
        output = _tool_blocks_to_function_call_items(directive.tool_blocks)
    else:
        output = [{
            "id": new_message_id(),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": output_text, "annotations": []}],
        }]

    return _base_response_obj(
        response_id=response_id,
        created_at=created,
        model_name=model_name,
        status="completed",
        output=output,
        output_text=output_text,
        usage=_usage(prompt, output_text),
        request_payload=request_payload,
    )


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


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

    def on_text_chunk(self, text_chunk: str):
        if not text_chunk:
            return
        self._ensure_started()
        self._ensure_text_item()
        self.answer_fragments.append(text_chunk)
        self.pending_chunks.append(self._wrap("response.output_text.delta", {
            "item_id": self.item_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "delta": text_chunk,
        }))

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

        chunks = self.drain()

        # If tool calls, emit function_call items instead of text completion
        if tool_blocks:
            for block in tool_blocks:
                if block.get("type") == "tool_use":
                    self.emit_function_call(block)
        elif self.text_started:
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
            final_payload["usage"] = _usage(self.prompt, output_text)

        chunks.append(self._wrap("response.completed", {"response": final_payload}))
        chunks.append("data: [DONE]\n\n")
        return chunks
