"""Small, thread-safe TTL caches used by the online RAG application.

The cache is intentionally process-local. It survives repeated requests handled by the
same Model Serving worker, but it does not survive scale-to-zero, deployment updates,
or requests routed to another worker. Persistent conversation memory is handled by
Unity Catalog memory, not by this cache.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class _CacheItem(Generic[V]):
    value: V
    expires_at: float


class TTLCache(Generic[K, V]):
    """Minimal TTL cache with bounded size and deterministic eviction."""

    def __init__(self, ttl_seconds: float, max_entries: int = 256):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._items: dict[K, _CacheItem[V]] = {}
        self._lock = threading.RLock()

    def get(self, key: K) -> V | None:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if item.expires_at <= now:
                self._items.pop(key, None)
                return None
            return item.value

    def set(self, key: K, value: V) -> None:
        now = time.monotonic()
        with self._lock:
            self._remove_expired(now)
            if len(self._items) >= self.max_entries and key not in self._items:
                # Evict the entry that expires first. This is deterministic and is
                # sufficient for a small serving-side cache.
                oldest_key = min(
                    self._items,
                    key=lambda existing_key: self._items[existing_key].expires_at,
                )
                self._items.pop(oldest_key, None)
            self._items[key] = _CacheItem(
                value=value,
                expires_at=now + self.ttl_seconds,
            )

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def _remove_expired(self, now: float) -> None:
        expired = [
            key for key, item in self._items.items() if item.expires_at <= now
        ]
        for key in expired:
            self._items.pop(key, None)
