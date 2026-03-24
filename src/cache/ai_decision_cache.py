"""
AI Decision Cache

Caches AI analysis results per market with a configurable TTL (default 30 minutes).
Prevents redundant API calls when the same market is analyzed multiple times within
a single trading session, which is the primary cause of rate-limit exhaustion.

Usage::

    cache = AIDecisionCache(ttl_minutes=30)

    # Check before calling AI
    cached = cache.get(market_id)
    if cached:
        probability, confidence = cached["probability"], cached["confidence"]
    else:
        probability, confidence = await _get_fast_ai_prediction(...)
        cache.set(market_id, probability, confidence)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class CachedDecision:
    """A single cached AI analysis result."""
    market_id: str
    probability: float
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.time() - self.timestamp) > ttl_seconds


class AIDecisionCache:
    """
    In-memory cache for AI market analysis results.

    Keyed by market_id.  Entries expire after ``ttl_minutes`` minutes so that
    stale predictions are never reused across trading sessions.

    Thread-safety: this cache is designed for use inside a single asyncio event
    loop and does NOT use locks.  It is safe for concurrent coroutines because
    Python's GIL protects dict operations.
    """

    def __init__(self, ttl_minutes: float = 30.0):
        self._ttl_seconds: float = ttl_minutes * 60.0
        self._store: Dict[str, CachedDecision] = {}
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, market_id: str) -> Optional[CachedDecision]:
        """
        Return a cached decision if it exists and has not expired.

        Returns ``None`` on a miss or if the entry is stale (and purges it).
        """
        entry = self._store.get(market_id)
        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired(self._ttl_seconds):
            del self._store[market_id]
            self._misses += 1
            return None

        self._hits += 1
        return entry

    def set(
        self,
        market_id: str,
        probability: float,
        confidence: float,
    ) -> None:
        """Store an AI analysis result for ``market_id``."""
        self._store[market_id] = CachedDecision(
            market_id=market_id,
            probability=probability,
            confidence=confidence,
        )

    def invalidate(self, market_id: str) -> None:
        """Remove a specific entry from the cache."""
        self._store.pop(market_id, None)

    def clear(self) -> None:
        """Flush the entire cache."""
        self._store.clear()

    def purge_expired(self) -> int:
        """Remove all expired entries.  Returns the number of entries removed."""
        expired = [
            k for k, v in self._store.items()
            if v.is_expired(self._ttl_seconds)
        ]
        for k in expired:
            del self._store[k]
        return len(expired)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of entries currently in the cache (including stale ones)."""
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate since instantiation (0.0–1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "ttl_minutes": self._ttl_seconds / 60.0,
        }


# ---------------------------------------------------------------------------
# Module-level singleton — shared across the whole process
# ---------------------------------------------------------------------------

_default_cache: Optional[AIDecisionCache] = None


def get_default_cache(ttl_minutes: float = 30.0) -> AIDecisionCache:
    """
    Return the process-wide singleton cache, creating it on first call.

    All callers that use this function share the same cache instance, which
    means a prediction fetched by the portfolio optimizer is immediately
    available to the immediate-trading path without a second API call.
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = AIDecisionCache(ttl_minutes=ttl_minutes)
    return _default_cache
