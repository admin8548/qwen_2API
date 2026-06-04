"""智能引号修复 + Edit/StrReplace 模糊匹配 + 通用 schema-based 参数名修复。

[ADDED 2026-06-05] 新增 fix_arguments_by_schema(): 基于工具 JSON Schema
做参数名模糊匹配，修复 Qwen 模型猜错参数名的问题（如 command→cmd、
path→file_path 等）。

动机：
    - Qwen 没有原生 function calling，靠 prompt 文本推测参数名，经常猜错。
    - AI 模型经常把普通 ASCII 引号写成中文/智能引号。
    - old_string 中有微小格式差异也会 exact fail。

策略：
    - fix_arguments_by_schema: schema-based 模糊匹配 + 别名字典
    - replace_smart_quotes：把所有智能引号变成 ASCII 引号
    - repair_exact_match：old_string 不 exact match 时，构造 fuzzy 正则
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

log = logging.getLogger("qwen2api.tool_arg_fixer")

# ── [ADDED] Alias dictionary for common Qwen mis-guesses ──────────────────
_PARAM_ALIASES: dict[str, str] = {
    "command":      "cmd",
    "cmd_line":     "cmd",
    "shell_command":"cmd",
    "exec":         "cmd",
    "execute":      "cmd",
    "line":         "cmd",
    "path":         "file_path",
    "filePath":     "file_path",
    "filename":     "file_path",
    "fname":        "file_path",
    "file":         "file_path",
    "target_file":  "file_path",
    "target":       "file_path",
    "filepath":     "file_path",
    "dir":          "path",
    "directory":    "path",
    "folder":       "path",
    "query":        "pattern",
    "search":       "pattern",
    "regex":        "pattern",
    "regexp":       "pattern",
    "re":           "pattern",
    "text":         "content",
    "data":         "content",
    "body":         "content",
    "new_str":      "new_string",
    "old_str":      "old_string",
    "replacement":  "new_string",
    "replace_with": "new_string",
    "search_for":   "old_string",
    "find":         "old_string",
    "message":      "content",
    "msg":          "content",
    "instruction":  "input",
    "instructions": "input",
    "args":         "input",
    "argument":     "input",
    "arguments":    "input",
    "params":       "input",
    "parameters":   "input",
}


def _alias_key(name: str) -> str:
    """Normalize a parameter name for fuzzy matching."""
    return re.sub(r'[^a-z0-9]', '', name.lower())


def _fuzzy_match_param(wrong_name: str, schema_params: dict[str, Any]) -> str | None:
    """Try to find the correct parameter name from the tool schema."""
    if not wrong_name or not schema_params:
        return None

    wrong_key = _alias_key(wrong_name)

    # Exact match (case-insensitive)
    for param_name in schema_params:
        if param_name.lower() == wrong_name.lower():
            return param_name

    # Alias dictionary hit
    canonical = _PARAM_ALIASES.get(wrong_name)
    if canonical and canonical in schema_params:
        return canonical

    # Fuzzy: strip non-alphanumeric and compare
    for param_name in schema_params:
        if _alias_key(param_name) == wrong_key:
            return param_name

    # Substring containment
    for param_name in schema_params:
        pk = _alias_key(param_name)
        if len(pk) >= 3 and (pk in wrong_key or wrong_key in pk):
            return param_name

    return None


def fix_arguments_by_schema(
    tool_name: str,
    args: dict[str, Any],
    tool_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """[ADDED 2026-06-05] Repair parameter names using the tool's JSON schema.

    If a key in `args` doesn't match any schema property, try alias/fuzzy
    matching. If a unique match is found, rename the key.
    """
    if not isinstance(args, dict) or not tool_schema:
        return args

    # Get the schema properties (handle nested function.tools schema)
    properties = tool_schema.get("parameters", {}).get("properties") if isinstance(tool_schema.get("parameters"), dict) else None
    if not properties:
        properties = tool_schema.get("properties")
    if not properties or not isinstance(properties, dict):
        return args

    valid_names = set(properties.keys())
    if not valid_names:
        return args

    out = dict(args)
    renamed = []
    for key in list(out.keys()):
        if key in valid_names:
            continue
        match = _fuzzy_match_param(key, properties)
        if match and match not in out:
            out[match] = out.pop(key)
            renamed.append(f"{key}->{match}")

    if renamed:
        log.info("[ArgFix] tool=%s renamed=%s", tool_name, ",".join(renamed))

    return out


# ── Smart quotes + Edit/StrReplace specific logic (original) ──────────────

_SMART_DOUBLE_QUOTES = {"\u00ab", "\u201c", "\u201d", "\u275e", "\u201f", "\u201e", "\u275d", "\u00bb"}
_SMART_SINGLE_QUOTES = {"\u2018", "\u2019", "\u201a", "\u201b"}

_DOUBLE_QUOTE_CLASS = '["\u00ab\u201c\u201d\u275e\u201f\u201e\u275d\u00bb]'
_SINGLE_QUOTE_CLASS = "['\u2018\u2019\u201a\u201b]"


def replace_smart_quotes(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = []
    for ch in text:
        if ch in _SMART_DOUBLE_QUOTES:
            out.append('"')
        elif ch in _SMART_SINGLE_QUOTES:
            out.append("'")
        else:
            out.append(ch)
    return "".join(out)


def _build_fuzzy_pattern(text: str) -> str:
    parts = []
    for ch in text:
        if ch in _SMART_DOUBLE_QUOTES or ch == '"':
            parts.append(_DOUBLE_QUOTE_CLASS)
        elif ch in _SMART_SINGLE_QUOTES or ch == "'":
            parts.append(_SINGLE_QUOTE_CLASS)
        elif ch in (" ", "\t"):
            parts.append(r"\s+")
        elif ch == "\\":
            parts.append(r"\\{1,2}")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def repair_exact_match(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """对 Edit / StrReplace / search_replace 类工具：若 old_string 在文件里 exact
    不中，用 fuzzy pattern 搜；唯一命中则替换 args 里的 old_string 为精确匹配文本。"""
    if not isinstance(args, dict):
        return args
    lower = (tool_name or "").lower()
    if not any(key in lower for key in ("edit", "str_replace", "strreplace", "search_replace")):
        return args

    old_string = args.get("old_string") or args.get("old_str")
    if not isinstance(old_string, str) or not old_string:
        return args

    file_path = args.get("file_path") or args.get("path")
    if not isinstance(file_path, str) or not file_path:
        return args

    try:
        if not os.path.exists(file_path):
            return args
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return args

    if old_string in content:
        _normalize_new_string(args)
        return args

    try:
        pattern = _build_fuzzy_pattern(old_string)
        matches = list(re.finditer(pattern, content))
    except re.error:
        return args

    if len(matches) != 1:
        return args

    matched_text = matches[0].group(0)
    if "old_string" in args:
        args["old_string"] = matched_text
    elif "old_str" in args:
        args["old_str"] = matched_text
    _normalize_new_string(args)
    return args


def _normalize_new_string(args: dict[str, Any]) -> None:
    for key in ("new_string", "new_str"):
        if key in args and isinstance(args[key], str):
            args[key] = replace_smart_quotes(args[key])


def fix_tool_call_arguments(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """对所有工具调用应用全部修复。幂等。"""
    if not isinstance(args, dict):
        return args
    return repair_exact_match(tool_name, args)
