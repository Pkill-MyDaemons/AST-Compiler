"""Simple key-value cache with TTL."""
from typing import Dict, List, Optional


class CacheEntry:
    """A single cache entry."""

    key: str
    value: str
    ttl: int
    hits: int

    def __init__(self, key: str, value: str, ttl: int) -> None:
        self.key = key
        self.value = value
        self.ttl = ttl
        self.hits = 0

    def is_expired(self) -> bool:
        return self.ttl <= 0

    def tick(self) -> None:
        if self.ttl > 0:
            self.ttl = self.ttl - 1

    def access(self) -> str:
        self.hits = self.hits + 1
        return self.value


class Cache:
    """In-memory key-value cache."""

    entries: Dict[str, CacheEntry]
    max_size: int
    default_ttl: int

    def __init__(self, max_size: int, default_ttl: int) -> None:
        self.entries = {}
        self.max_size = max_size
        self.default_ttl = default_ttl

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        if len(self.entries) >= self.max_size and key not in self.entries:
            return False
        actual_ttl: int = ttl if ttl is not None else self.default_ttl
        self.entries[key] = CacheEntry(key, value, actual_ttl)
        return True

    def get(self, key: str) -> Optional[str]:
        entry = self.entries.get(key)
        if entry is None or entry.is_expired():
            return None
        return entry.access()

    def delete(self, key: str) -> bool:
        if key in self.entries:
            del self.entries[key]
            return True
        return False

    def evict_expired(self) -> int:
        expired: List[str] = []
        for key, entry in self.entries.items():
            if entry.is_expired():
                expired.append(key)
        for key in expired:
            del self.entries[key]
        return len(expired)

    def tick(self) -> None:
        for entry in self.entries.values():
            entry.tick()

    def size(self) -> int:
        return len(self.entries)

    def most_accessed(self) -> Optional[CacheEntry]:
        if not self.entries:
            return None
        best = None
        best_hits: int = -1
        for entry in self.entries.values():
            if entry.hits > best_hits:
                best_hits = entry.hits
                best = entry
        return best


def warm_cache(cache: Cache, data: Dict[str, str]) -> int:
    loaded: int = 0
    for key, value in data.items():
        if cache.set(key, value):
            loaded = loaded + 1
    return loaded
