"""Responses API → Chat Completions payload adapter.

Converts OpenAI Responses API requests (input, tools, instructions) into
the standard Chat Completions format consumed by the prompt/pipeline layer.
"""

from __future__ import annotations

import json
import re
from typing import Any


_HOSTED_TOOL_DESCRIPTION = (
    "OpenAI Responses hosted tool requested by the client. This gateway does not "
    "execute hosted tools locally; emit a function_call with the requested "
    "arguments when this capability is needed so the client can handle it."
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"text", "input_text", "output_text", "refusal"}:
                text = part.get("text") or part.get("refusal") or ""
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return _as_text(content)


def _normalize_message_content(content: Any, role: str | None = None) -> Any:
    """Translate Responses content parts into the chat-style parts."""
    del role
    if isinstance(content, str) or content is None:
        return content or ""
    if not isinstance(content, list):
        return _as_text(content)

    out: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            out.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"text", "input_text", "output_text", "refusal"}:
            text = part.get("text") or part.get("refusal") or ""
            if text:
                out.append({"type": "text", "text": str(text)})
            continue
        if part_type == "input_image":
            image_url = part.get("image_url") or part.get("url")
            if isinstance(image_url, dict):
                out.append({"type": "image_url", "image_url": image_url})
            elif isinstance(image_url, str):
                out.append({"type": "image_url", "image_url": {"url": image_url}})
            else:
                out.append(dict(part))
            continue
        if part_type in {"input_file", "file"}:
            out.append(dict(part))
            continue
        out.append({"type": "text", "text": _as_text(part)})
    return out


def _message_item_to_chat_message(item: dict[str, Any]) -> dict[str, Any] | None:
    role = item.get("role") or "user"
    if role == "developer":
        role = "system"
    if role not in {"system", "user", "assistant", "tool"}:
        role = "user"

    content = _normalize_message_content(item.get("content", ""), role)
    msg: dict[str, Any] = {"role": role, "content": content}
    if role == "tool":
        call_id = item.get("tool_call_id") or item.get("call_id") or item.get("id")
        if call_id:
            msg["tool_call_id"] = str(call_id)
    if role == "assistant" and isinstance(item.get("tool_calls"), list):
        msg["tool_calls"] = item["tool_calls"]
    return msg


def _function_call_item_to_chat_message(item: dict[str, Any]) -> dict[str, Any]:
    call_id = str(item.get("call_id") or item.get("id") or "call_unknown")
    name = str(item.get("name") or "")
    arguments = item.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _function_call_output_item_to_chat_message(item: dict[str, Any]) -> dict[str, Any]:
    call_id = str(item.get("call_id") or item.get("id") or "call_unknown")
    output = item.get("output", item.get("content", ""))
    if isinstance(output, list):
        output = _extract_content_text(output)
    elif not isinstance(output, str):
        output = _as_text(output)
    return {"role": "tool", "tool_call_id": call_id, "content": output}


def responses_input_to_messages(input_value: Any, *, instructions: str = "") -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages

    if not isinstance(input_value, list):
        messages.append({"role": "user", "content": _as_text(input_value)})
        return messages

    for item in input_value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": _as_text(item)})
            continue

        item_type = item.get("type")
        if item_type == "message" or "role" in item:
            msg = _message_item_to_chat_message(item)
            if msg:
                messages.append(msg)
            continue
        if item_type == "function_call":
            messages.append(_function_call_item_to_chat_message(item))
            continue
        if item_type == "function_call_output":
            messages.append(_function_call_output_item_to_chat_message(item))
            continue

        messages.append({"role": "user", "content": _normalize_message_content([item])})

    return messages


def _safe_tool_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())[:64].strip("_")
    if not cleaned:
        cleaned = "hosted_tool"
    if not re.match(r"^[A-Za-z]", cleaned):
        cleaned = f"tool_{cleaned}"
    return cleaned


def responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "function":
            if isinstance(tool.get("function"), dict):
                fn_name = tool["function"].get("name")
                if isinstance(fn_name, str) and fn_name:
                    used.add(fn_name)
                out.append(tool)
                continue
            fn = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters") or {},
            }
            if isinstance(fn.get("name"), str) and fn["name"]:
                used.add(fn["name"])
            if "strict" in tool:
                fn["strict"] = tool.get("strict")
            out.append({"type": "function", "function": fn})
            continue

        original_type = str(tool_type or tool.get("name") or "hosted_tool")
        base_name = _safe_tool_name(str(tool.get("name") or original_type))
        name = base_name
        index = 2
        while name in used:
            suffix = f"_{index}"
            name = f"{base_name[:64-len(suffix)]}{suffix}"
            index += 1
        used.add(name)
        description = str(tool.get("description") or _HOSTED_TOOL_DESCRIPTION)
        description = f"{description} Original hosted tool type: {original_type}."
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
                },
                "x_responses_hosted_tool": tool,
            }
        )
    return out


def _combined_instructions(req_data: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("instructions", "developer", "system"):
        value = req_data.get(key)
        if not value:
            continue
        text = _extract_content_text(value)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def adapt_responses_request_to_chat(req_data: dict[str, Any]) -> dict[str, Any]:
    """Best-effort Responses -> existing chat-completions payload adapter."""
    instructions = _combined_instructions(req_data)
    messages = responses_input_to_messages(req_data.get("input", ""), instructions=instructions)
    payload: dict[str, Any] = {
        "model": req_data.get("model", "gpt-3.5-turbo"),
        "messages": messages,
        "stream": bool(req_data.get("stream", False)),
        "tools": responses_tools_to_chat_tools(req_data.get("tools", [])),
    }

    for key in ("metadata", "conversation_id", "session_key", "upstream_files"):
        if key in req_data:
            payload[key] = req_data[key]
    if "previous_response_id" in req_data and "conversation_id" not in payload and "session_key" not in payload:
        payload["conversation_id"] = req_data["previous_response_id"]

    for key in (
        "temperature", "top_p", "max_output_tokens", "max_completion_tokens",
        "tool_choice", "parallel_tool_calls", "truncation", "store",
        "include", "reasoning", "text",
    ):
        if key in req_data:
            payload[key] = req_data[key]

    return payload
