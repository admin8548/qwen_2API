"""In-memory response store for Responses API.

Stores completed response objects so that sub-endpoints (compact, get,
input_items) can retrieve them by response_id.

This is the Phase A stub: pure in-memory dict with TTL eviction.
Phase B will swap in a real context-compression backend.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("qwen2api.response_store")


@dataclass(slots=True)
class StoredResponse:
    """A single stored Responses API result."""

    response_id: str
    payload: dict[str, Any]  # the full response JSON object
    original_request: dict[str, Any]  # the original POST /v1/responses request body
    created_at: float = field(default_factory=time.time)
    compacted: bool = False  # set to True when Phase B compression is applied


class ResponseStore:
    """Simple in-memory dict with TTL-based eviction.

    Usage:
        store = ResponseStore(max_size=2000, ttl_seconds=3600)
        store.put(response_id, payload, original_request)
        entry = store.get(response_id)
    """

    def __init__(self, *, max_size: int = 2000, ttl_seconds: int = 3600) -> None:
        self._data: dict[str, StoredResponse] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    # ── write ────────────────────────────────────────────────

    def put(
        self,
        response_id: str,
        payload: dict[str, Any],
        original_request: dict[str, Any] | None = None,
    ) -> StoredResponse:
        """Store a completed response.  Evicts expired / over-capacity entries."""
        self._evict_expired()
        if len(self._data) >= self.max_size:
            self._evict_oldest(1)

        entry = StoredResponse(
            response_id=response_id,
            payload=dict(payload),
            original_request=dict(original_request) if original_request else {},
        )
        self._data[response_id] = entry
        log.info("[ResponseStore] put response_id=%s total=%d", response_id, len(self._data))
        return entry

    # ── read ─────────────────────────────────────────────────

    def get(self, response_id: str) -> StoredResponse | None:
        entry = self._data.get(response_id)
        if entry is None:
            return None
        if self._is_expired(entry):
            del self._data[response_id]
            return None
        return entry

    def mark_compacted(self, response_id: str, compacted_payload: dict[str, Any]) -> bool:
        """Replace the stored payload with a compacted version (Phase B hook)."""
        entry = self._data.get(response_id)
        if entry is None:
            return False
        entry.payload = dict(compacted_payload)
        entry.compacted = True
        log.info("[ResponseStore] mark_compacted response_id=%s", response_id)
        return True

    # ── evict helpers ────────────────────────────────────────

    def _is_expired(self, entry: StoredResponse) -> bool:
        return (time.time() - entry.created_at) > self.ttl_seconds

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            rid for rid, entry in self._data.items()
            if (now - entry.created_at) > self.ttl_seconds
        ]
        for rid in expired:
            del self._data[rid]

    def _evict_oldest(self, count: int) -> None:
        if len(self._data) < count:
            return
        by_age = sorted(self._data.items(), key=lambda kv: kv[1].created_at)
        for rid, _ in by_age[:count]:
            del self._data[rid]

    # ── stats ────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._data)
