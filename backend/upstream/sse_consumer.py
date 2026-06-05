import json
import logging

log = logging.getLogger("qwen2api.sse")


def _first_text(*values) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _extract_reasoning(delta: dict) -> str:
    extra = delta.get("extra") if isinstance(delta.get("extra"), dict) else {}
    return _first_text(
        delta.get("reasoning_content"),
        delta.get("reasoning"),
        delta.get("reasoning_text"),
        delta.get("thinking"),
        delta.get("thoughts"),
        extra.get("reasoning_content"),
        extra.get("reasoning"),
        extra.get("reasoning_text"),
        extra.get("thinking"),
        extra.get("thoughts"),
    )


def parse_sse_chunk(chunk: str) -> list[dict]:
    events = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
            events.append(obj)
        except Exception:
            continue

    parsed = []
    for evt in events:
        if evt.get("choices"):
            delta = evt["choices"][0].get("delta", {})
            phase = delta.get("phase", "answer")
            content = delta.get("content", "")




            reasoning = _extract_reasoning(delta)
            if reasoning:
                content = reasoning
                phase = "thinking_summary" if phase == "answer" else phase

            # Log if content contains "Tool" and "does not exist"
            if content and "Tool" in content and "does not exist" in content:
                log.warning(f"[SSE] Detected tool interception: content={content!r} phase={delta.get('phase')} status={delta.get('status')} extra={delta.get('extra')}")

            parsed.append(
                {
                    "type": "delta",
                    "phase": phase,
                    "content": content,
                    "status": delta.get("status", ""),
                    "extra": delta.get("extra", {}),
                }
            )

            # Capture upstream finish_reason signal (e.g. "length" when output token limit hit)
            upstream_finish = evt["choices"][0].get("finish_reason")
            if upstream_finish:
                parsed.append({
                    "type": "upstream_finish",
                    "finish_reason": upstream_finish,
                })
                if upstream_finish == "length":
                    log.warning("[SSE] Upstream signaled finish_reason=length (output token limit hit)")


    return parsed
