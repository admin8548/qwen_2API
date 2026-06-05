import time
import uuid


CUSTOM_TOOL_COMPAT_FEATURE_CONFIG = {
    "thinking_enabled": True,
    "output_schema": "phase",
    "research_mode": "normal",
    "auto_thinking": True,
    "thinking_mode": "Auto",
    "thinking_format": "summary",
    "auto_search": True,
    "code_interpreter": False,
    "plugins_enabled": False,
}

CUSTOM_TOOL_LOW_LATENCY_OVERRIDES = {
    "thinking_enabled": False,
    "auto_thinking": False,
    "thinking_mode": "Disabled",
}

IMAGE_CHAT_TYPES = {"image_gen", "t2i"}
VIDEO_CHAT_TYPES = {"t2v"}
UPSTREAM_IMAGE_CHAT_TYPE = "t2i"
UPSTREAM_VIDEO_CHAT_TYPE = "t2v"


def _apply_thinking_config(feature_config: dict, enabled: bool, reasoning_effort: str | None = None) -> None:
    """Apply thinking configuration to feature_config.

    Qwen upstream supports three thinking modes:
      - "Auto"      → model decides whether to think (auto_thinking=True)
      - "Thinking"  → always think (auto_thinking=False)
      - "Disabled"  → never think (auto_thinking=False)

    reasoning_effort maps:
      - "high"   → "Thinking" (always think)
      - "medium" → "Auto" (model decides)
      - "low"    → "Disabled" (no thinking)
      - None     → use enabled parameter: enabled→"Auto", !enabled→"Disabled"
    """
    if reasoning_effort == "high":
        mode = "Thinking"
        auto = False
        enabled = True
    elif reasoning_effort == "medium":
        mode = "Auto"
        auto = True
        enabled = True
    elif reasoning_effort == "low":
        mode = "Disabled"
        auto = False
        enabled = False
    elif enabled:
        mode = "Auto"
        auto = True
    else:
        mode = "Disabled"
        auto = False
        enabled = False
    feature_config.update(
        {
            "thinking_enabled": enabled,
            "auto_thinking": auto,
            "thinking_mode": mode,
        }
    )
    # Qwen web UI sets auto_search=true when thinking is enabled
    if enabled and mode in ("Thinking", "Auto"):
        feature_config["auto_search"] = True


def normalize_upstream_chat_type(chat_type: str) -> str:
    return UPSTREAM_IMAGE_CHAT_TYPE if chat_type in IMAGE_CHAT_TYPES else chat_type


def _image_ratio(image_options: dict) -> str:
    ratio = image_options.get("ratio") or image_options.get("aspect_ratio") or image_options.get("aspectRatio")
    return str(ratio or "1:1")


def _build_image_feature_config(image_options: dict) -> dict:
    ratio = _image_ratio(image_options)
    return {
        "thinking_enabled": False,
        "output_schema": "phase",
        "auto_thinking": False,
        "thinking_mode": "off",
        "auto_search": True,
        "code_interpreter": False,
        "function_calling": False,
        "plugins_enabled": True,
        "image_generation": True,
        "default_aspect_ratio": ratio,
    }


def _build_video_feature_config(video_options: dict) -> dict:
    ratio = _image_ratio(video_options)
    return {
        "thinking_enabled": False,
        "output_schema": "phase",
        "auto_thinking": False,
        "thinking_mode": "off",
        "auto_search": True,
        "code_interpreter": False,
        "function_calling": False,
        "plugins_enabled": True,
        "video_generation": True,
        "default_aspect_ratio": ratio,
    }


def build_chat_payload(
    chat_id: str,
    model: str,
    content: str,
    has_custom_tools: bool = False,
    files: list[dict] | None = None,
    chat_type: str = "t2t",
    image_options: dict | None = None,
    thinking_enabled: bool | None = None,
    enable_search: bool = False,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
) -> dict:
    ts = int(time.time())
    is_image_gen = chat_type in IMAGE_CHAT_TYPES
    is_video_gen = chat_type in VIDEO_CHAT_TYPES
    image_options = image_options or {}
    if is_image_gen:
        feature_config = _build_image_feature_config(image_options)
        message_chat_type = "t2t"
        sub_chat_type = UPSTREAM_IMAGE_CHAT_TYPE
        message_extra_meta = {
            "subChatType": UPSTREAM_IMAGE_CHAT_TYPE,
            "mode": "image_generation",
            "aspectRatio": _image_ratio(image_options),
            "size": _image_ratio(image_options),
        }
    elif is_video_gen:
        feature_config = _build_video_feature_config(image_options)
        message_chat_type = UPSTREAM_VIDEO_CHAT_TYPE
        sub_chat_type = UPSTREAM_VIDEO_CHAT_TYPE
        message_extra_meta = {
            "subChatType": UPSTREAM_VIDEO_CHAT_TYPE,
            "mode": "video_generation",
            "aspectRatio": _image_ratio(image_options),
            "size": _image_ratio(image_options),
        }
    else:
        feature_config = {
            **CUSTOM_TOOL_COMPAT_FEATURE_CONFIG,
            **(CUSTOM_TOOL_LOW_LATENCY_OVERRIDES if has_custom_tools else {}),
            "function_calling": False,
            "enable_tools": False,
            "enable_function_call": False,
            "tool_choice": "none",
            "auto_search": True,
            "plugins_enabled": False,
        }
        if thinking_enabled is not None:
            _apply_thinking_config(feature_config, bool(thinking_enabled), reasoning_effort=reasoning_effort)
        if max_output_tokens is not None and max_output_tokens > 0:
            # Qwen web payloads are not fully documented; include the limit in
            # both feature_config and top-level compatibility fields below so
            # upstream variants that honor any common spelling can enforce it.
            feature_config["max_tokens"] = int(max_output_tokens)
            feature_config["max_output_tokens"] = int(max_output_tokens)
        message_chat_type = chat_type
        sub_chat_type = chat_type
        message_extra_meta = {"subChatType": chat_type}

    payload = {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": model,
        "parent_id": None,
        "messages": [
            {
                "fid": str(uuid.uuid4()),
                "parentId": None,
                "childrenIds": [str(uuid.uuid4())],
                "role": "user",
                "content": content,
                "user_action": "chat",
                "files": files or [],
                "timestamp": ts,
                "models": [model],
                "chat_type": message_chat_type,
                "feature_config": feature_config,
                "extra": {"meta": message_extra_meta},
                "sub_chat_type": sub_chat_type,
                "parent_id": None,
            }
        ],
        "timestamp": ts,
    }
    if max_output_tokens is not None and max_output_tokens > 0:
        limit = int(max_output_tokens)
        payload["max_tokens"] = limit
        payload["max_output_tokens"] = limit
        payload["max_new_tokens"] = limit
    if is_image_gen or is_video_gen:
        payload["size"] = _image_ratio(image_options)
    return payload
