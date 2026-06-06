"""Tool-call truncation detection and continuation helpers.

The primary prompt-visible protocol is QNML:

<|QNML|tool_calls>
  <|QNML|invoke name="TOOL_NAME">
    <|QNML|parameter name="ARG"><![CDATA[value]]></|QNML|parameter>
  </|QNML|invoke>
</|QNML|tool_calls>

Compatibility parsing still recognizes legacy XML ``<tool_calls>`` / ``<tool_call>``
and old marker blocks, but those are not the main prompt protocol.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("qwen2api.truncation_recovery")


_QNML_TOOL_CALLS_OPEN_RE = re.compile(r"<\s*\|\s*QNML\s*\|\s*tool_calls\b[^>]*>", re.IGNORECASE)
_QNML_TOOL_CALLS_CLOSE_RE = re.compile(r"<\s*/\s*\|\s*QNML\s*\|\s*tool_calls\s*>", re.IGNORECASE)
_QNML_INVOKE_OPEN_RE = re.compile(r"<\s*\|\s*QNML\s*\|\s*invoke\b[^>]*>", re.IGNORECASE)
_QNML_INVOKE_CLOSE_RE = re.compile(r"<\s*/\s*\|\s*QNML\s*\|\s*invoke\s*>", re.IGNORECASE)
_QNML_PARAMETER_OPEN_RE = re.compile(r"<\s*\|\s*QNML\s*\|\s*parameter\b[^>]*>", re.IGNORECASE)
_QNML_PARAMETER_CLOSE_RE = re.compile(r"<\s*/\s*\|\s*QNML\s*\|\s*parameter\s*>", re.IGNORECASE)

_LEGACY_TOOL_CALLS_OPEN_RE = re.compile(r"<\s*tool_calls\b[^>]*>", re.IGNORECASE)
_LEGACY_TOOL_CALLS_CLOSE_RE = re.compile(r"<\s*/\s*tool_calls\s*>", re.IGNORECASE)
_LEGACY_INVOKE_OPEN_RE = re.compile(r"<\s*invoke\b[^>]*>", re.IGNORECASE)
_LEGACY_INVOKE_CLOSE_RE = re.compile(r"<\s*/\s*invoke\s*>", re.IGNORECASE)
_LEGACY_PARAMETER_OPEN_RE = re.compile(r"<\s*parameter\b[^>]*>", re.IGNORECASE)
_LEGACY_PARAMETER_CLOSE_RE = re.compile(r"<\s*/\s*parameter\s*>", re.IGNORECASE)
_LEGACY_TOOL_CALL_OPEN_RE = re.compile(r"<\s*tool_call\b[^>]*>", re.IGNORECASE)
_LEGACY_TOOL_CALL_CLOSE_RE = re.compile(r"<\s*/\s*tool_call\s*>", re.IGNORECASE)

_TOOL_CALL_OPEN_RE = re.compile(r"##TOOL_CALL##", re.IGNORECASE)
_TOOL_CALL_CLOSE_RE = re.compile(r"##END_CALL##", re.IGNORECASE)
_CDATA_OPEN_RE = re.compile(r"<!\[CDATA\[", re.IGNORECASE)
_CDATA_CLOSE_RE = re.compile(r"\]\]>")
_PARTIAL_TOOL_MARKER_RE = re.compile(
    r"(?:<\s*/?\s*(?:\|\s*QNML(?:\s*\|\s*(?:tool_calls|invoke|parameter)?)?|tool_calls?|invoke|parameter)"
    r"|##\s*(?:TOOL_CALL|END_CALL)?)\s*$",
    re.IGNORECASE,
)


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


def _has_unclosed(open_re: re.Pattern[str], close_re: re.Pattern[str], text: str) -> bool:
    return _count(open_re, text) > _count(close_re, text)


def _contains_tool_marker(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "<|qnml|tool_calls",
            "</|qnml|tool_calls",
            "<|qnml|invoke",
            "</|qnml|invoke",
            "<|qnml|parameter",
            "</|qnml|parameter",
            "<tool_calls",
            "</tool_calls",
            "<invoke",
            "</invoke",
            "<parameter",
            "</parameter",
            "<tool_call",
            "</tool_call",
            "##tool_call##",
            "##end_call##",
        )
    )


def is_truncated(text: str) -> bool:
    """Return True when output appears cut off inside a tool-call block."""
    if not text or not text.strip():
        return False

    trimmed = text.rstrip()
    if _PARTIAL_TOOL_MARKER_RE.search(trimmed):
        return True

    if not _contains_tool_marker(trimmed):
        return False

    # Primary protocol: QNML.
    if _has_unclosed(_QNML_TOOL_CALLS_OPEN_RE, _QNML_TOOL_CALLS_CLOSE_RE, trimmed):
        return True
    if _has_unclosed(_QNML_INVOKE_OPEN_RE, _QNML_INVOKE_CLOSE_RE, trimmed):
        return True
    if _has_unclosed(_QNML_PARAMETER_OPEN_RE, _QNML_PARAMETER_CLOSE_RE, trimmed):
        return True

    # Compatibility: legacy XML / canonical XML.
    if _has_unclosed(_LEGACY_TOOL_CALLS_OPEN_RE, _LEGACY_TOOL_CALLS_CLOSE_RE, trimmed):
        return True
    if _has_unclosed(_LEGACY_INVOKE_OPEN_RE, _LEGACY_INVOKE_CLOSE_RE, trimmed):
        return True
    if _has_unclosed(_LEGACY_PARAMETER_OPEN_RE, _LEGACY_PARAMETER_CLOSE_RE, trimmed):
        return True
    if _has_unclosed(_LEGACY_TOOL_CALL_OPEN_RE, _LEGACY_TOOL_CALL_CLOSE_RE, trimmed):
        return True

    # Compatibility: old marker JSON block.
    if _has_unclosed(_TOOL_CALL_OPEN_RE, _TOOL_CALL_CLOSE_RE, trimmed):
        return True

    # Unclosed CDATA usually means a QNML/legacy parameter is still incomplete.
    if _count(_CDATA_OPEN_RE, trimmed) > _count(_CDATA_CLOSE_RE, trimmed):
        return True

    return False


def deduplicate_continuation(existing: str, continuation: str) -> str:
    """Remove the longest duplicate overlap between existing tail and continuation head."""
    if not existing or not continuation:
        return continuation
    continuation = _strip_continuation_preamble(continuation)
    if not continuation:
        return ""
    max_overlap = min(500, len(existing), len(continuation))
    if max_overlap < 10:
        return continuation

    best_overlap = 0
    for length in range(max_overlap, 9, -1):
        prefix = continuation[:length]
        if existing.endswith(prefix):
            best_overlap = length
            break

    if best_overlap >= 10:
        return continuation[best_overlap:]

    # If the continuation restarted from a sizeable earlier slice of the
    # existing tail (common when the model ignores "do not repeat"), trim that
    # repeated head even when it is not an exact suffix/prefix overlap.
    tail_window = existing[-3000:]
    head_window = continuation[:1200]
    for length in range(min(len(head_window), 800), 79, -1):
        snippet = continuation[:length]
        if snippet and tail_window.rfind(snippet) >= 0:
            return continuation[length:]

    tail_lines = existing.splitlines()[-20:]
    cont_lines = continuation.splitlines()
    if tail_lines and cont_lines:
        first_cont = cont_lines[0].strip()
        if first_cont:
            for i in range(len(tail_lines)):
                if tail_lines[i].strip() != first_cont:
                    continue
                matched = 1
                for k in range(1, len(cont_lines)):
                    if i + k >= len(tail_lines):
                        break
                    if cont_lines[k].strip() == tail_lines[i + k].strip():
                        matched += 1
                    else:
                        break
                if matched >= 2:
                    return "\n".join(cont_lines[matched:])
                if matched == 1 and len(first_cont) >= 40:
                    return "\n".join(cont_lines[1:])

    return continuation


_CONTINUATION_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"(?:继续(?:如下|：|:)?\s*)|"
    r"(?:以下是(?:继续|剩余)(?:内容|部分)?(?:：|:)?\s*)|"
    r"(?:接着(?:上文|继续)?(?:：|:)?\s*)|"
    r"(?:从(?:上次|刚才|截断处).*?(?:继续|开始)(?:：|:)?\s*)|"
    r"(?:Sure[,，]?\s*)|"
    r"(?:Continu(?:e|ing)(?: from where (?:I|we) stopped)?(?:\:)?\s*)|"
    r"(?:Here is the continuation(?:\:)?\s*)"
    r")",
    re.IGNORECASE,
)


def _strip_continuation_preamble(text: str) -> str:
    """Remove meta prefaces that are not part of the user's answer."""
    if not text:
        return text
    cleaned = text
    for _ in range(3):
        nxt = _CONTINUATION_PREAMBLE_RE.sub("", cleaned, count=1)
        if nxt == cleaned:
            break
        cleaned = nxt
    return cleaned


def build_continuation_prompt(partial_response: str, anchor_chars: int = 2000) -> tuple[str, str]:
    """Build the ``(assistant_context, user_followup)`` continuation prompt."""
    anchor = partial_response[-anchor_chars:] if len(partial_response) > anchor_chars else partial_response
    assistant_ctx = ("...\n" + anchor) if len(partial_response) > anchor_chars else anchor
    followup = (
        "Your previous response was cut off in the middle of a QNML tool-call block. "
        "The last part was:\n\n"
        "```\n"
        f"...{anchor[-300:] if len(anchor) > 300 else anchor}\n"
        "```\n\n"
        "Continue EXACTLY from where you stopped. DO NOT repeat any content already generated. "
        "DO NOT restart the response. Output ONLY the remaining QNML/tool-call text, "
        "starting immediately from the cut-off point."
    )
    return assistant_ctx, followup


# ── Plain-text truncation detection (P2-6b) ──────────────────────────────────
# Qwen upstream may hit an internal output token limit and return
# finish_reason=stop mid-sentence.  Unlike tool-call truncation (which has
# unclosed XML tags as a reliable signal), plain-text truncation requires
# heuristic detection based on trailing punctuation and structural cues.

# Characters that signal a complete sentence / segment in Chinese & English.
_SENTENCE_ENDERS = frozenset(
    "。！？…）》】"              # CJK (no colon/semicolon/comma — they imply continuation)
    ".!?)"                       # Latin
    "'"                         # quotes
    "\n"                        # newline
)

# Code-block closers / structural closers that indicate a "complete" ending.
_STRUCTURAL_ENDINGS = (
    "```",      # fenced code block close
    "</answer>",
    "</result>",
    "</output>",
    "</response>",
)

_INCOMPLETE_MARKDOWN_TAIL_RE = re.compile(
    r"(?m)(?:^|\n)\s*(?:"
    r"(?:[-*+]\s*)|"                 # bare bullet
    r"(?:\d+[.)]\s*)|"               # bare ordered-list marker: "3."
    r"(?:#{1,6}\s*)|"                # bare heading marker
    r"(?:\*\*[^*\n]*)|"              # unclosed bold marker
    r"(?:`{1,3}[^`\n]*)"             # unclosed inline/fenced code marker
    r")$"
)


def is_plain_text_truncated(text: str, *, min_len: int = 120) -> bool:
    """Heuristic: does *text* look like it was cut off mid-sentence?

    Returns True when the text is long enough to be a real response AND
    does not end with any sentence-closing punctuation or structural marker.

    Conservative by design – only triggers when there's a strong signal
    that the model was still generating when the stream ended.
    """
    if not text or len(text) < min_len:
        return False

    trimmed = text.rstrip()
    if not trimmed:
        return False

    # Already has tool-call markers?  Let is_truncated() handle that.
    if _contains_tool_marker(trimmed):
        return False

    # Markdown/list fragments are common when the upstream cuts output right at
    # a numbered next-step item, e.g. "3." or "3. **config".
    last_line = trimmed.rsplit("\n", 1)[-1].strip()
    if _INCOMPLETE_MARKDOWN_TAIL_RE.search(trimmed):
        return True
    if re.match(r"^\d+[.)]\s*$", last_line):
        return True
    if last_line.count("**") % 2 == 1:
        return True
    if last_line.count("`") % 2 == 1:
        return True
    if last_line.startswith(("**", "`")) and last_line.count(last_line[:2] if last_line.startswith("**") else "`") % 2:
        return True

    # Unclosed fenced code block is almost always a truncated answer.
    if trimmed.count("```") % 2 == 1:
        return True

    # Ends with a structural closer?  Probably complete.
    lower_tail = trimmed[-20:].lower()
    for ending in _STRUCTURAL_ENDINGS:
        if lower_tail.endswith(ending):
            return False

    # Ends with sentence-ending punctuation?  Probably complete.
    if trimmed[-1] in _SENTENCE_ENDERS:
        return False

    # Ends with a standalone closing bracket/brace that suggests the model
    # finished a structured block (JSON, list, etc.).
    if trimmed[-1] in "}]>":
        return False

    # A trailing colon/comma almost certainly means "more to come".
    if last_line and last_line[-1] in ":,，：、":
        return True

    # A final dangling opening delimiter is a high-confidence incomplete tail.
    if trimmed[-1] in "([{（【《“‘\"":
        return True

    return False



def is_explicit_max_output_truncated(text: str, max_output_tokens: int | None) -> bool:
    """Detect client-requested output cap truncation without auto-continuing.

    If the caller explicitly asks for a small max_output_tokens, the upstream may
    stop with finish_reason=stop and no explicit length signal.  In Responses API
    semantics this should be reported as incomplete due to max_output_tokens, not
    silently continued beyond the caller's limit.
    """
    if not max_output_tokens or max_output_tokens <= 0 or not text:
        return False
    # Character/token ratios vary by language; use a conservative upper bound to
    # decide whether the visible answer is plausibly capped by the requested
    # output budget.
    if len(text.rstrip()) > int(max_output_tokens) * 4 + 32:
        return False
    return is_plain_text_truncated(text, min_len=8)

def build_text_continuation_prompt(
    partial_response: str,
    anchor_chars: int = 800,
) -> tuple[str, str]:
    """Build a continuation prompt for plain-text truncation.

    Unlike tool-call continuation, this asks the model to pick up from
    the last complete thought and continue naturally.
    """
    anchor = partial_response[-anchor_chars:] if len(partial_response) > anchor_chars else partial_response
    assistant_ctx = ("...\n" + anchor) if len(partial_response) > anchor_chars else anchor

    # Find the last "safe" sentence boundary in the anchor to avoid
    # repeating a half-sentence.
    safe_cutoff = _find_safe_resume_point(anchor)

    followup = (
        "Your previous response was cut off mid-sentence due to an output length limit. "
        "The last part of your response was:\n\n"
        "```\n"
        f"{anchor[-300:]}\n"
        "```\n\n"
        "Continue from where you stopped. DO NOT repeat any content already generated. "
        "Complete the sentence/thought that was interrupted, then continue the response naturally. "
        "Output ONLY the continuation text, starting immediately from the cut-off point."
    )
    return assistant_ctx, followup


def _find_safe_resume_point(anchor: str) -> int:
    """Find the position of the last sentence-ending punctuation in anchor.

    Used to trim the continuation prompt so we don't pass half-sentences
    as context.  Returns the position just after the last sentence ender,
    or 0 if none found.
    """
    best = 0
    for i, ch in enumerate(anchor):
        if ch in _SENTENCE_ENDERS:
            best = i + 1
    return best


# ── Prompt leakage detection and cleanup ──────────────────────────────────────
# When Qwen hits its output token limit, the model may start echoing prompt
# content (Human: / Assistant: / <system> markers) into the response.
# This happens because the model's next-token prediction is still "thinking"
# about the prompt structure even after the output budget is exhausted.

import re as _re

# Patterns that indicate prompt structure has leaked into the output
_PROMPT_LEAKAGE_PATTERNS = [
    # Human/Assistant turn markers
    _re.compile(r'(?:^|\n)\s*Human\s*[:(]', _re.IGNORECASE),
    _re.compile(r'(?:^|\n)\s*Human\s*\(', _re.IGNORECASE),
    _re.compile(r'(?:^|\n)\s*Assistant\s*:', _re.IGNORECASE),
    # Task-specific markers from prompt builder
    _re.compile(r'(?:^|\n)\s*Human\s+\(CURRENT TASK', _re.IGNORECASE),
    _re.compile(r'(?:^|\n)\s*Human\s+\(ORIGINAL TASK', _re.IGNORECASE),
    # System prompt markers
    _re.compile(r'(?:^|\n)\s*<system>', _re.IGNORECASE),
    _re.compile(r'(?:^|\n)\s*System\s*:', _re.IGNORECASE),
    _re.compile(r'(?:^|\n)\s*<environment_context>', _re.IGNORECASE),
    _re.compile(r'(?:^|\n)\s*</INSTRUCTIONS>', _re.IGNORECASE),
    _re.compile(r'(?:^|\n)\s*My request for Codex:', _re.IGNORECASE),
    # Tool result markers
    _re.compile(r'(?:^|\n)\s*\[Tool Result\]', _re.IGNORECASE),
    _re.compile(r'(?:^|\n)\s*\[\/Tool Result\]', _re.IGNORECASE),
    # State notice markers
    _re.compile(r'(?:^|\n)\s*<STATE NOTICE>', _re.IGNORECASE),
    _re.compile(r'(?:^|\n)\s*<TASK MEMORY>', _re.IGNORECASE),
    # QNML tool markers that shouldn't appear in plain text output
    _re.compile(r'(?:^|\n)\s*<\|QNML\|', _re.IGNORECASE),
]

_PROMPT_LEAKAGE_HIGH_CONFIDENCE = (
    "CURRENT TASK",
    "ORIGINAL TASK",
    "<environment_context>",
    "</INSTRUCTIONS>",
    "My request for Codex:",
    "[Tool Result]",
    "[/Tool Result]",
    "<STATE NOTICE>",
    "<TASK MEMORY>",
    "<|QNML|",
)


def strip_prompt_leakage(text: str) -> tuple[str, bool]:
    """Detect and remove prompt leakage from the end of a response.

    Returns (cleaned_text, was_leaking).
    """
    if not text or len(text) < 50:
        return text, False

    # Find the earliest leakage marker
    earliest_pos = len(text)
    matched_pattern = None

    for pattern in _PROMPT_LEAKAGE_PATTERNS:
        match = pattern.search(text)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
            matched_pattern = pattern.pattern

    if earliest_pos < len(text):
        cleaned = text[:earliest_pos].rstrip()
        tail = text[earliest_pos:earliest_pos + 240]
        high_confidence = any(marker.lower() in tail.lower() for marker in _PROMPT_LEAKAGE_HIGH_CONFIDENCE)
        # Don't trim if we'd lose more than 60% of the response, unless the
        # marker is a high-confidence prompt/control structure. Qwen often
        # leaks exactly these blocks after output-budget exhaustion.
        if not high_confidence and len(cleaned) < len(text) * 0.4:
            return text, False
        log.warning(
            "[PromptLeak] detected prompt leakage at pos=%d pattern=%s trimmed=%d chars",
            earliest_pos, matched_pattern[:50], len(text) - len(cleaned),
        )
        return cleaned, True

    return text, False
