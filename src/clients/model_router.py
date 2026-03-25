"""
Unified model routing layer for the Kalshi AI Trading Bot.
Primary provider: Gemini (free tier, 1M tokens/day).
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.clients.xai_client import TradingDecision, XAIClient
from src.clients.openrouter_client import OpenRouterClient, MODEL_PRICING
from src.clients.gemini_client import GeminiClient
from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


CAPABILITY_MAP: Dict[str, List[Tuple[str, str]]] = {
    "fast": [("gemini-1.5-flash", "gemini")],
    "cheap": [("gemini-1.5-flash", "gemini")],
    "reasoning": [("gemini-1.5-flash", "gemini")],
    "balanced": [("gemini-1.5-flash", "gemini")],
}

FULL_FLEET: List[Tuple[str, str]] = [
    ("gemini-1.5-flash", "gemini"),
    ("gemini-1.5-flash-8b", "gemini"),
]


@dataclass
class ModelHealth:
    model: str
    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_latency: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests

    @property
    def is_healthy(self) -> bool:
        if self.consecutive_failures < 5:
            return True
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > timedelta(minutes=5)

    def record_success(self, latency: float) -> None:
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
        self.total_latency += latency

    def record_failure(self) -> None:
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now()


class ModelRouter(TradingLoggerMixin):

    _THROTTLE_DELAY_SECONDS: float = 0.5

    def __init__(
        self,
        xai_client: Optional[XAIClient] = None,
        openrouter_client: Optional[OpenRouterClient] = None,
        db_manager: Any = None,
    ):
        self.db_manager = db_manager
        self.xai_client: Optional[XAIClient] = xai_client
        self.openrouter_client: Optional[OpenRouterClient] = openrouter_client
        self.gemini_client: Optional[GeminiClient] = None

        self.model_health: Dict[str, ModelHealth] = {}
        for model_name, provider in FULL_FLEET:
            key = self._model_key(model_name, provider)
            self.model_health[key] = ModelHealth(model=model_name, provider=provider)

        self._last_call_time: float = 0.0

        self.logger.info(
            "ModelRouter initialized — primary provider: Gemini (free)",
            fleet_size=len(FULL_FLEET),
        )

    @staticmethod
    def _model_key(model: str, provider: str) -> str:
        return f"{provider}::{model}"

    async def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._THROTTLE_DELAY_SECONDS - (now - self._last_call_time)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call_time = time.monotonic()

    def _ensure_gemini(self) -> GeminiClient:
        if self.gemini_client is None:
            self.gemini_client = GeminiClient(db_manager=self.db_manager)
            self.logger.info("Lazily initialized GeminiClient")
        return self.gemini_client

    def _ensure_xai(self) -> XAIClient:
        if self.xai_client is None:
            self.xai_client = XAIClient(db_manager=self.db_manager)
        return self.xai_client

    def _ensure_openrouter(self) -> OpenRouterClient:
        if self.openrouter_client is None:
            self.openrouter_client = OpenRouterClient(db_manager=self.db_manager)
        return self.openrouter_client

    def _infer_provider(self, model: str) -> str:
        if model.startswith("gemini"):
            return "gemini"
        if "/" in model:
            return "openrouter"
        if model.startswith("grok"):
            return "xai"
        return "gemini"

    def _resolve_targets(
        self,
        model: Optional[str] = None,
        capability: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        targets: List[Tuple[str, str]] = []

        if model is not None:
            provider = self._infer_provider(model)
            targets.append((model, provider))
        elif capability is not None:
            targets.extend(CAPABILITY_MAP.get(capability, []))
        else:
            targets = list(FULL_FLEET)

        seen = set(targets)
        for entry in FULL_FLEET:
            if entry not in seen:
                targets.append(entry)
                seen.add(entry)

        healthy = [t for t in targets if self._is_model_healthy(t[0], t[1])]
        return healthy if len(healthy) >= 1 else targets

    def _is_model_healthy(self, model: str, provider: str) -> bool:
        health = self.model_health.get(self._model_key(model, provider))
        return health.is_healthy if health else True

    def _record_success(self, model: str, provider: str, latency: float) -> None:
        health = self.model_health.get(self._model_key(model, provider))
        if health:
            health.record_success(latency)

    def _record_failure(self, model: str, provider: str) -> None:
        health = self.model_health.get(self._model_key(model, provider))
        if health:
            health.record_failure()

    async def _dispatch_completion(
        self,
        prompt: str,
        model: str,
        provider: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        strategy: str = "unknown",
        query_type: str = "completion",
        market_id: Optional[str] = None,
    ) -> Optional[str]:
        if provider == "gemini":
            client = self._ensure_gemini()
            return await client.get_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                strategy=strategy,
                query_type=query_type,
                market_id=market_id,
            )
        elif provider == "xai":
            client = self._ensure_xai()
            return await client.get_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                strategy=strategy,
                query_type=query_type,
                market_id=market_id,
            )
        else:
            client = self._ensure_openrouter()
            return await client.get_completion(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                strategy=strategy,
                query_type=query_type,
                market_id=market_id,
            )

    async def _dispatch_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str,
        model: str,
        provider: str,
    ) -> Optional[TradingDecision]:
        if provider == "gemini":
            client = self._ensure_gemini()
            return await client.get_trading_decision(
                market_data=market_data,
                portfolio_data=portfolio_data,
                news_summary=news_summary,
            )
        elif provider == "xai":
            client = self._ensure_xai()
            return await client.get_trading_decision(
                market_data=market_data,
                portfolio_data=portfolio_data,
                news_summary=news_summary,
            )
        else:
            client = self._ensure_openrouter()
            return await client.get_trading_decision(
                market_data=market_data,
                portfolio_data=portfolio_data,
                news_summary=news_summary,
                model=model,
            )

    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        capability: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        strategy: str = "unknown",
        query_type: str = "completion",
        market_id: Optional[str] = None,
    ) -> Optional[str]:
        targets = self._resolve_targets(model=model, capability=capability)

        for target_model, provider in targets:
            await self._throttle()
            start = time.time()
            try:
                result = await self._dispatch_completion(
                    prompt=prompt,
                    model=target_model,
                    provider=provider,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    strategy=strategy,
                    query_type=query_type,
                    market_id=market_id,
                )
                if result is not None:
                    self._record_success(target_model, provider, time.time() - start)
                    return result
                self._record_failure(target_model, provider)
            except Exception as exc:
                self._record_failure(target_model, provider)
                self.logger.warning(f"Model {target_model} failed: {exc}")
                continue

        self.logger.error("All models exhausted for get_completion")
        return None

    async def get_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str = "",
        model: Optional[str] = None,
        capability: Optional[str] = None,
    ) -> Optional[TradingDecision]:
        targets = self._resolve_targets(model=model, capability=capability)

        for target_model, provider in targets:
            await self._throttle()
            start = time.time()
            try:
                decision = await self._dispatch_trading_decision(
                    market_data=market_data,
                    portfolio_data=portfolio_data,
                    news_summary=news_summary,
                    model=target_model,
                    provider=provider,
                )
                if decision is not None:
                    self._record_success(target_model, provider, time.time() - start)
                    self.logger.info(
                        "Trading decision routed successfully",
                        model=target_model,
                        provider=provider,
                        action=decision.action,
                        confidence=decision.confidence,
                    )
                    return decision
                self._record_failure(target_model, provider)
            except Exception as exc:
                self._record_failure(target_model, provider)
                self.logger.warning(f"Model {target_model} failed: {exc}")
                continue

        self.logger.error("All models exhausted for get_trading_decision")
        return None

    def get_total_cost(self) -> float:
        total = 0.0
        if self.xai_client:
            total += self.xai_client.total_cost
        if self.openrouter_client:
            total += self.openrouter_client.total_cost
        if self.gemini_client:
            total += self.gemini_client.total_cost
        return total

    def get_total_requests(self) -> int:
        total = 0
        if self.xai_client:
            total += self.xai_client.request_count
        if self.openrouter_client:
            total += self.openrouter_client.request_count
        if self.gemini_client:
            total += self.gemini_client.request_count
        return total

    def get_cost_summary(self) -> Dict[str, Any]:
        return {
            "total_cost": round(self.get_total_cost(), 6),
            "total_requests": self.get_total_requests(),
            "primary_provider": "gemini (free)",
        }

    async def close(self) -> None:
        tasks = []
        if self.gemini_client:
            tasks.append(self.gemini_client.close())
        if self.xai_client:
            tasks.append(self.xai_client.close())
        if self.openrouter_client:
            tasks.append(self.openrouter_client.close())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.logger.info("ModelRouter closed", total_requests=self.get_total_requests())
