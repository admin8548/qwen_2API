"""Model-aware Chat ID prewarm pool.

The pool pre-creates upstream ``chat_id`` values per account *and model* so a
request can skip the expensive /chats/new handshake.  Empty upstream responses
are treated as a strong signal that a prewarmed batch may be bad: the affected
(email, model) bucket is flushed and temporarily cooled down.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Optional

log = logging.getLogger("qwen2api.chat_pool")
PoolKey = tuple[str, str]


class _Entry:
    __slots__ = ("chat_id", "email", "model", "created_at")

    def __init__(self, chat_id: str, *, email: str = "", model: str = ""):
        self.chat_id = chat_id
        self.email = email
        self.model = model
        self.created_at = time.time()

    @classmethod
    def create(cls, *, chat_id: str, email: str, model: str) -> "_Entry":
        return cls(chat_id, email=email, model=model)


class ChatIdPool:
    """Coroutine-safe queues keyed by ``(email, model)``."""

    def __init__(
        self,
        client,
        *,
        target_per_account: int = 5,
        ttl_seconds: float = 10 * 60,
        max_concurrency: int = 16,
        default_model: str = "qwen3.6-plus",
        models: str | list[str] | tuple[str, ...] | set[str] | None = None,
        failure_cooldown_seconds: float = 60,
        refill_interval_seconds: float = 30,
    ):
        self._client = client
        self._target = max(0, int(target_per_account))
        self._ttl = max(1.0, float(ttl_seconds))
        self._max_concurrency = max(1, int(max_concurrency))
        self._prewarm_semaphore = asyncio.Semaphore(self._max_concurrency)
        self._default_model = default_model or "qwen3.6-plus"
        self._models = self._normalize_models(models)
        self._failure_cooldown_seconds = max(0.0, float(failure_cooldown_seconds))
        self._refill_interval_seconds = max(1.0, float(refill_interval_seconds))
        self._queues: dict[PoolKey, deque[_Entry]] = {}
        self._cooldowns: dict[PoolKey, float] = {}
        self._lock = asyncio.Lock()
        self._refill_task: Optional[asyncio.Task] = None
        self._refilling_keys: set[PoolKey] = set()
        self._shutdown = False
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stale_drops": 0,
            "failure_flushes": 0,
            "overfill_drops": 0,
        }

    def _normalize_models(self, models: str | list[str] | tuple[str, ...] | set[str] | None) -> tuple[str, ...]:
        raw: list[str]
        if models is None:
            raw = [self._default_model]
        elif isinstance(models, str):
            raw = [m.strip() for m in models.split(",")]
        else:
            raw = [str(m).strip() for m in models]
        result: list[str] = []
        for model in [*raw, self._default_model]:
            if model and model not in result:
                result.append(model)
        return tuple(result)

    def _key(self, email: str, model: str | None = None) -> PoolKey:
        return (email, model or self._default_model)

    def _key_label(self, key: PoolKey) -> str:
        return f"{key[0]}/{key[1]}"

    def _is_on_cooldown_locked(self, key: PoolKey, *, now: float | None = None) -> bool:
        deadline = self._cooldowns.get(key)
        if deadline is None:
            return False
        now = time.time() if now is None else now
        if deadline <= now:
            self._cooldowns.pop(key, None)
            return False
        return True

    def _account_by_email(self, email: str):
        pool = getattr(self._client, "account_pool", None)
        if pool is None:
            return None
        if hasattr(pool, "get_by_email"):
            return pool.get_by_email(email)
        return next((a for a in getattr(pool, "accounts", []) if getattr(a, "email", None) == email), None)

    async def _delete_entry(self, account_or_email, chat_id: str, *, source: str) -> None:
        if not chat_id:
            return
        account = self._account_by_email(account_or_email) if isinstance(account_or_email, str) else account_or_email
        token = getattr(account, "token", None)
        if not token:
            log.debug("[ChatIdPool] skip delete chat_id=%s source=%s: missing token", chat_id, source)
            return
        delete_reliable = getattr(self._client, "delete_chat_reliable", None)
        if delete_reliable is not None:
            await delete_reliable(token, chat_id, source=source)
            return
        delete_raw = getattr(self._client, "delete_chat", None)
        if delete_raw is not None:
            await delete_raw(token, chat_id)

    def _delete_entry_background(self, account_or_email, chat_id: str, *, source: str) -> None:
        if not chat_id:
            return
        account = self._account_by_email(account_or_email) if isinstance(account_or_email, str) else account_or_email
        token = getattr(account, "token", None)
        background_delete = getattr(self._client, "delete_chat_background", None)
        if background_delete is not None and token:
            background_delete(token, chat_id, source=source)
            return
        try:
            asyncio.get_running_loop().create_task(self._delete_entry(account_or_email, chat_id, source=source))
        except RuntimeError:
            log.debug("[ChatIdPool] skip background delete chat_id=%s source=%s: no running loop", chat_id, source)

    @property
    def target(self) -> int:
        return self._target

    @property
    def ttl(self) -> float:
        return self._ttl

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def models(self) -> tuple[str, ...]:
        return self._models

    def update_config(
        self,
        *,
        target: int | None = None,
        ttl_seconds: float | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        if target is not None:
            self._target = max(0, int(target))
            if self._target == 0:
                # Disable immediately and avoid handing out stale ids.  Deletions are best-effort/background.
                items = [(key[0], entry.chat_id) for key, q in self._queues.items() for entry in q]
                self._queues = {}
                self._cooldowns = {}
                for email, chat_id in items:
                    self._delete_entry_background(email, chat_id, source="chat_pool_disable")
        if ttl_seconds is not None:
            self._ttl = max(1.0, float(ttl_seconds))
        if max_concurrency is not None:
            next_value = max(1, int(max_concurrency))
            if next_value != self._max_concurrency:
                self._max_concurrency = next_value
                self._prewarm_semaphore = asyncio.Semaphore(self._max_concurrency)
        log.info(
            "[ChatIdPool] config updated target=%s ttl=%ss max_concurrency=%s models=%s",
            self._target,
            self._ttl,
            self._max_concurrency,
            ",".join(self._models),
        )

    async def apply_config(
        self,
        *,
        target: int | None = None,
        ttl_seconds: float | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        previous_target = self._target
        self.update_config(target=target, ttl_seconds=ttl_seconds, max_concurrency=max_concurrency)
        if target is not None and self._target < previous_target and self._target > 0:
            await self.prune_to_target()

    async def start(self) -> None:
        if self._target <= 0:
            log.info("[ChatIdPool] disabled by target=0")
            return
        self._refill_task = asyncio.create_task(self._refill_loop())
        log.info("[ChatIdPool] started target=%s ttl=%ss models=%s", self._target, self._ttl, ",".join(self._models))

    async def stop(self) -> None:
        self._shutdown = True
        if self._refill_task:
            self._refill_task.cancel()
            try:
                await self._refill_task
            except (asyncio.CancelledError, Exception):
                pass
        await self.flush_all(source="chat_pool_stop")

    async def acquire(self, email: str, model: str | None = None) -> Optional[str]:
        if not email or self._target <= 0:
            return None
        key = self._key(email, model)
        expired: list[str] = []
        selected: str | None = None
        async with self._lock:
            now = time.time()
            if self._is_on_cooldown_locked(key, now=now):
                self._stats["misses"] += 1
                return None
            q = self._queues.get(key)
            if not q:
                self._stats["misses"] += 1
                return None
            while q:
                entry = q.popleft()
                if now - entry.created_at < self._ttl:
                    self._stats["hits"] += 1
                    selected = entry.chat_id
                    log.debug("[ChatIdPool] HIT key=%s chat_id=%s", self._key_label(key), selected)
                    break
                self._stats["stale_drops"] += 1
                expired.append(entry.chat_id)
                log.debug("[ChatIdPool] expired key=%s chat_id=%s", self._key_label(key), entry.chat_id)
            if selected is None:
                self._stats["misses"] += 1
        for chat_id in expired:
            self._delete_entry_background(email, chat_id, source="chat_pool_expired")
        if selected:
            await self._schedule_refill(email, key[1], reason="consume")
        return selected

    async def _create_chat_direct(self, token: str, model: str) -> str:
        executor = getattr(self._client, "executor", None)
        if executor is None:
            raise RuntimeError("client executor unavailable")
        direct = getattr(executor, "create_chat_direct", None)
        if direct is not None:
            return await direct(token, model)
        return await executor.create_chat(token, model, use_prewarmed=False)

    async def _prewarm_one(self, account, model: str) -> None:
        try:
            token = getattr(account, "token", "")
            email = getattr(account, "email", "")
            if not token or not email:
                log.warning("[ChatIdPool] prewarm skipped email=%s: missing token/email", email or "-")
                return
            key = self._key(email, model)
            async with self._lock:
                if self._target <= 0 or self._is_on_cooldown_locked(key):
                    return
                if len(self._queues.get(key, [])) >= self._target:
                    return
            async with self._prewarm_semaphore:
                chat_id = await self._create_chat_direct(token, key[1])
            should_delete = False
            async with self._lock:
                if self._target <= 0 or self._is_on_cooldown_locked(key):
                    should_delete = True
                else:
                    q = self._queues.setdefault(key, deque())
                    if len(q) >= self._target:
                        should_delete = True
                    else:
                        q.append(_Entry.create(chat_id=chat_id, email=email, model=key[1]))
                        log.info("[ChatIdPool] prewarmed key=%s chat_id=%s pool_size=%s", self._key_label(key), chat_id, len(q))
            if should_delete:
                self._stats["overfill_drops"] += 1
                self._delete_entry_background(account, chat_id, source="chat_pool_overfill")
        except Exception as e:
            err = str(e) or type(e).__name__
            log.warning("[ChatIdPool] prewarm failed email=%s model=%s: %s", getattr(account, "email", "?"), model, err)

    async def _schedule_refill(self, email: str, model: str, *, reason: str) -> None:
        if self._shutdown or self._target <= 0 or not email:
            return
        key = self._key(email, model)
        async with self._lock:
            if self._is_on_cooldown_locked(key):
                return
            if key in self._refilling_keys:
                return
            if len(self._queues.get(key, [])) >= self._target:
                return
            self._refilling_keys.add(key)

        async def runner() -> None:
            try:
                await self._refill_account_once(email, model, reason=reason)
            finally:
                async with self._lock:
                    self._refilling_keys.discard(key)

        try:
            task = asyncio.create_task(runner())
            task.set_name(f"chat-id-refill-{email}-{model}")
        except RuntimeError:
            async with self._lock:
                self._refilling_keys.discard(key)

    async def _refill_account_once(self, email: str, model: str, *, reason: str) -> None:
        account = self._account_by_email(email)
        if account is None:
            return
        if not getattr(account, "token", "") or getattr(account, "status_code", "valid") != "valid":
            return
        key = self._key(email, model)
        async with self._lock:
            if self._is_on_cooldown_locked(key):
                return
            q_size = len(self._queues.get(key, []))
            if q_size >= self._target:
                return
        log.debug("[ChatIdPool] schedule refill key=%s reason=%s pool_size=%s target=%s", self._key_label(key), reason, q_size, self._target)
        await self._prewarm_one(account, key[1])

    async def _refill_loop(self) -> None:
        await asyncio.sleep(1.0)
        while not self._shutdown:
            try:
                await self._refill_once()
            except Exception as e:
                log.warning("[ChatIdPool] refill error: %s", e)
            await asyncio.sleep(self._refill_interval_seconds)

    async def _refill_once(self) -> None:
        if self._target <= 0:
            return
        pool = getattr(self._client, "account_pool", None)
        if pool is None:
            return
        await self.prune_expired()
        valid = [
            a for a in (getattr(pool, "accounts", []) or [])
            if getattr(a, "token", "") and getattr(a, "status_code", "valid") == "valid"
        ]
        refill_tasks: list[asyncio.Task] = []
        batch_size = max(1, self._max_concurrency * 4)

        async def drain_batch() -> None:
            if not refill_tasks:
                return
            results = await asyncio.gather(*refill_tasks, return_exceptions=True)
            refill_tasks.clear()
            for result in results:
                if isinstance(result, Exception):
                    log.warning("[ChatIdPool] refill task failed: %s", result)

        for acc in valid:
            for model in self._models:
                key = self._key(acc.email, model)
                async with self._lock:
                    if self._is_on_cooldown_locked(key):
                        continue
                    q_size = len(self._queues.get(key, []))
                deficit = self._target - q_size
                if deficit > 0:
                    refill_tasks.append(asyncio.create_task(self._refill_account_once(acc.email, model, reason="periodic")))
                    if len(refill_tasks) >= batch_size:
                        await drain_batch()
                elif deficit < 0:
                    await self.prune_account_to_target(acc.email, model)
        await drain_batch()

    async def invalidate(self, email: str, chat_id: str) -> None:
        if not email or not chat_id:
            return
        removed: list[tuple[str, str]] = []
        async with self._lock:
            for key, q in list(self._queues.items()):
                if key[0] != email:
                    continue
                remaining = deque(e for e in q if e.chat_id != chat_id)
                if len(remaining) != len(q):
                    removed.append((email, chat_id))
                    self._queues[key] = remaining
                    log.info("[ChatIdPool] invalidated key=%s chat_id=%s", self._key_label(key), chat_id)
        for account_email, removed_chat_id in removed:
            await self._delete_entry(account_email, removed_chat_id, source="chat_pool_invalidate")

    async def contains(self, email: str, chat_id: str) -> bool:
        if not email or not chat_id:
            return False
        async with self._lock:
            return any(key[0] == email and any(e.chat_id == chat_id for e in q) for key, q in self._queues.items())

    async def chat_ids(self, email: str | None = None) -> set[str]:
        async with self._lock:
            ids: set[str] = set()
            for key, q in self._queues.items():
                if email and key[0] != email:
                    continue
                ids.update(e.chat_id for e in q)
            return ids

    async def flush_account(self, email: str, model: str | None = None) -> int:
        if not email:
            return 0
        entries: list[_Entry] = []
        keys: list[PoolKey] = []
        async with self._lock:
            for key in list(self._queues.keys()):
                if key[0] == email and (model is None or key[1] == model):
                    keys.append(key)
            for key in keys:
                q = self._queues.get(key, deque())
                entries.extend(q)
                self._queues[key] = deque()
            if entries:
                log.info("[ChatIdPool] flushed %s entries for email=%s model=%s", len(entries), email, model or "*")
        for entry in entries:
            await self._delete_entry(email, entry.chat_id, source="chat_pool_flush")
        return len(entries)

    async def record_failure(self, email: str, model: str | None = None, *, reason: str = "failure") -> int:
        if not email:
            return 0
        key = self._key(email, model)
        flushed = await self.flush_account(email, key[1])
        async with self._lock:
            self._stats["failure_flushes"] += flushed
            if self._failure_cooldown_seconds > 0:
                self._cooldowns[key] = time.time() + self._failure_cooldown_seconds
        log.warning("[ChatIdPool] failure recorded key=%s reason=%s flushed=%s cooldown=%ss", self._key_label(key), reason, flushed, self._failure_cooldown_seconds)
        return flushed

    async def flush_all(self, *, source: str = "chat_pool_flush_all") -> int:
        async with self._lock:
            items = [(key[0], entry.chat_id) for key, q in self._queues.items() for entry in q]
            self._queues = {}
        for email, chat_id in items:
            await self._delete_entry(email, chat_id, source=source)
        if items:
            log.info("[ChatIdPool] flushed all entries count=%s source=%s", len(items), source)
        return len(items)

    async def prune_expired(self) -> int:
        now = time.time()
        expired: list[tuple[str, str]] = []
        async with self._lock:
            for key, q in list(self._queues.items()):
                kept = deque()
                for entry in q:
                    if now - entry.created_at >= self._ttl:
                        expired.append((key[0], entry.chat_id))
                    else:
                        kept.append(entry)
                self._queues[key] = kept
            self._stats["stale_drops"] += len(expired)
            for key in list(self._cooldowns.keys()):
                self._is_on_cooldown_locked(key, now=now)
        for email, chat_id in expired:
            await self._delete_entry(email, chat_id, source="chat_pool_expired")
        if expired:
            log.info("[ChatIdPool] pruned expired entries count=%s ttl=%ss", len(expired), self._ttl)
        return len(expired)

    async def prune_account_to_target(self, email: str, model: str | None = None) -> int:
        if not email:
            return 0
        removed: list[tuple[str, str]] = []
        async with self._lock:
            for key, q in list(self._queues.items()):
                if key[0] != email or (model is not None and key[1] != model):
                    continue
                while len(q) > self._target:
                    removed.append((email, q.pop().chat_id))
        for account_email, chat_id in removed:
            await self._delete_entry(account_email, chat_id, source="chat_pool_prune")
        return len(removed)

    async def prune_to_target(self) -> int:
        keys = list(self._queues.keys())
        total = 0
        for email, model in keys:
            total += await self.prune_account_to_target(email, model)
        return total

    async def size(self, email: str, model: str | None = None) -> int:
        async with self._lock:
            if model is not None:
                return len(self._queues.get(self._key(email, model), []))
            return sum(len(q) for key, q in self._queues.items() if key[0] == email)

    async def total_size(self) -> int:
        async with self._lock:
            return sum(len(q) for q in self._queues.values())

    async def snapshot(self) -> dict:
        now = time.time()
        async with self._lock:
            for key in list(self._cooldowns.keys()):
                self._is_on_cooldown_locked(key, now=now)
            per_key = {self._key_label(key): len(q) for key, q in self._queues.items()}
            per_account: dict[str, int] = {}
            for key, q in self._queues.items():
                per_account[key[0]] = per_account.get(key[0], 0) + len(q)
            cooldowns = {
                self._key_label(key): max(0.0, deadline - now)
                for key, deadline in self._cooldowns.items()
                if deadline > now
            }
            return {
                "stats": {
                    **self._stats,
                    "total_cached": sum(per_key.values()),
                    "target_per_account": self._target,
                    "ttl_seconds": self._ttl,
                    "max_concurrency": self._max_concurrency,
                },
                "models": list(self._models),
                "per_key": per_key,
                "per_account": per_account,
                "cooldowns": cooldowns,
            }
