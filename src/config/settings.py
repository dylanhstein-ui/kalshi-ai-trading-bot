"""
Configuration settings for the Kalshi trading system.
Manages trading parameters, API configurations, and risk management settings.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class APIConfig:
    """API configuration settings."""
    kalshi_api_key: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    kalshi_base_url: str = "https://api.elections.kalshi.com"
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    xai_api_key: str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openai_base_url: str = "https://api.openai.com/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"


@dataclass
class EnsembleConfig:
    """Multi-model ensemble configuration."""
    enabled: bool = True
    models: Dict[str, Dict] = field(default_factory=lambda: {
        "grok-3": {"provider": "xai", "role": "forecaster", "weight": 0.30},
        "anthropic/claude-3.5-sonnet": {"provider": "openrouter", "role": "news_analyst", "weight": 0.20},
        "openai/gpt-4o": {"provider": "openrouter", "role": "bull_researcher", "weight": 0.20},
        "google/gemini-flash-1.5": {"provider": "openrouter", "role": "bear_researcher", "weight": 0.15},
        "deepseek/deepseek-r1": {"provider": "openrouter", "role": "risk_manager", "weight": 0.15},
    })
    min_models_for_consensus: int = 3
    disagreement_threshold: float = 0.25
    parallel_requests: bool = True
    debate_enabled: bool = True
    calibration_tracking: bool = True
    max_ensemble_cost: float = 0.50


@dataclass
class SentimentConfig:
    """News and sentiment analysis configuration."""
    enabled: bool = True
    rss_feeds: List[str] = field(default_factory=lambda: [
        "https://feeds.reuters.com/reuters/topNews",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
    ])
    sentiment_model: str = "google/gemini-3.1-flash-lite-preview"
    cache_ttl_minutes: int = 30
    max_articles_per_source: int = 10
    relevance_threshold: float = 0.3


# Trading strategy configuration
# NCAAB NO-side: 74% WR, +10% ROI — ONLY profitable category.
# Economic trades: -70% ROI, 78% of all losses.
# PRICE IMPACT NOTE: Buying a position typically causes the price to drop immediately
# after purchase (e.g. buy at 13.5 cents, drops to 12.75 after fill).
# This means quick flips are usually unprofitable due to market impact.
# Strategy: hold positions longer to let the market recover and move in our favor.
@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    max_position_size_pct: float = 30.0
    max_daily_loss_pct: float = 30.0
    max_positions: int = 10
    min_balance: float = 0.0
    min_volume: float = 100.0
    max_time_to_expiry_days: int = 5

    # AI decision making
    min_confidence_to_trade: float = 0.60

    # Category-specific confidence adjustments
    category_confidence_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "sports": 0.90,
        "economics": 1.15,
        "politics": 1.05,
        "default": 1.0
    })

    scan_interval_seconds: int = 60

    # AI model configuration
    primary_model: str = "grok-3"
    fallback_model: str = "grok-4-1-fast-non-reasoning"
    ai_temperature: float = 0
    ai_max_tokens: int = 8000

    # Position sizing (LEGACY)
    default_position_size: float = 3.0
    position_size_multiplier: float = 1.0

    # Kelly Criterion settings (PRIMARY position sizing method)
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.25
    max_single_position: float = 0.03

    # Live trading mode control
    live_trading_enabled: bool = field(default_factory=lambda: os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true")
    paper_trading_mode: bool = field(default_factory=lambda: os.getenv("LIVE_TRADING_ENABLED", "false").lower() != "true")

    # Trading frequency
    market_scan_interval: int = 30
    position_check_interval: int = 15
    max_trades_per_hour: int = 20
    run_interval_minutes: int = 10
    num_processor_workers: int = 5

    # Market selection preferences
    preferred_categories: List[str] = field(default_factory=lambda: ["crypto", "bitcoin", "ethereum"])
    # Crypto market focus - capitalize on AI advantage in volatile markets
    crypto_focus_enabled: bool = True
    crypto_position_size_multiplier: float = 1.5  # 50% larger positions in crypto to exploit AI edge
    excluded_categories: List[str] = field(default_factory=lambda: [])

    # High-confidence, near-expiry strategy
    enable_high_confidence_strategy: bool = True
    high_confidence_threshold: float = 0.95
    high_confidence_market_odds: float = 0.90
    high_confidence_expiry_hours: int = 24

    # AI trading criteria
    max_analysis_cost_per_decision: float = 0.15
    min_confidence_threshold: float = 0.45

    # Cost control
    daily_ai_budget: float = 10.0
    max_ai_cost_per_decision: float = 0.08
    analysis_cooldown_hours: int = 3
    max_analyses_per_market_per_day: int = 4

    # Daily AI spending limits
    daily_ai_cost_limit: float = field(default_factory=lambda: float(os.getenv("DAILY_AI_COST_LIMIT", "10.0")))
    enable_daily_cost_limiting: bool = True
    sleep_when_limit_reached: bool = True

    # Market filtering
    min_volume_for_ai_analysis: float = 200.0
    exclude_low_liquidity_categories: List[str] = field(default_factory=lambda: [])

    # Price impact awareness
    # Buying moves the price against us immediately. Do not exit quickly.
    # Wait for market to recover before considering a close.
    min_hold_time_minutes: int = 30        # Never exit within 30 mins of entry
    price_impact_buffer: float = 0.01      # Expect ~1 cent adverse move on entry, factor into exit targets


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "DEBUG"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/trading_system.log"
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    max_log_file_size: int = 10 * 1024 * 1024
    backup_count: int = 5


# === CAPITAL ALLOCATION ACROSS STRATEGIES ===
market_making_allocation: float = 0.40
directional_allocation: float = 0.50
arbitrage_allocation: float = 0.10

# === PORTFOLIO OPTIMIZATION SETTINGS ===
use_risk_parity: bool = True
rebalance_hours: int = 6
min_position_size: float = 5.0
max_opportunities_per_batch: int = 50

# === RISK MANAGEMENT LIMITS ===
max_volatility: float = 0.40
max_correlation: float = 0.70
max_drawdown: float = 0.15
max_sector_exposure: float = 0.30

# === PERFORMANCE TARGETS ===
target_sharpe: float = 0.3
target_return: float = 0.15
min_trade_edge: float = 0.08
min_confidence_for_large_size: float = 0.50

# === DYNAMIC EXIT STRATEGIES ===
# Price impact note: after buying, price typically drops ~0.75 cents immediately.
# Quick exits lock in losses. Hold longer to let price recover.
use_dynamic_exits: bool = True
profit_threshold: float = 0.20
loss_threshold: float = 0.15
confidence_decay_threshold: float = 0.25
max_hold_time_hours: int = 240
volatility_adjustment: bool = True

# === MARKET MAKING STRATEGY ===
enable_market_making: bool = True
min_spread_for_making: float = 0.01
max_inventory_risk: float = 0.15
order_refresh_minutes: int = 15
max_orders_per_market: int = 4

# === MARKET SELECTION ===
min_volume_for_analysis: float = 200.0
min_volume_for_market_making: float = 500.0
min_price_movement: float = 0.02
max_bid_ask_spread: float = 0.15
min_confidence_long_term: float = 0.45

# === COST OPTIMIZATION ===
daily_ai_budget: float = 15.0
max_ai_cost_per_decision: float = 0.12
analysis_cooldown_hours: int = 2
max_analyses_per_market_per_day: int = 6
skip_news_for_low_volume: bool = True
news_search_volume_threshold: float = 1000.0

# === SYSTEM BEHAVIOR ===
beast_mode_enabled: bool = True
fallback_to_legacy: bool = True
log_level: str = "INFO"
performance_monitoring: bool = True

# === ADVANCED FEATURES ===
cross_market_arbitrage: bool = False
multi_model_ensemble: bool = True
sentiment_analysis: bool = True
websocket_streaming: bool = True
options_strategies: bool = False
algorithmic_execution: bool = False


@dataclass
class Settings:
    """Main settings class combining all configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)

    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.api.kalshi_api_key:
            raise ValueError("KALSHI_API_KEY environment variable is required")

        if not self.api.xai_api_key:
            raise ValueError("XAI_API_KEY environment variable is required")

        if self.trading.max_position_size_pct <= 0 or self.trading.max_position_size_pct > 100:
            raise ValueError("max_position_size_pct must be between 0 and 100")

        if self.trading.min_confidence_to_trade <= 0 or self.trading.min_confidence_to_trade > 1:
            raise ValueError("min_confidence_to_trade must be between 0 and 1")

        return True


# Global settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate()
except ValueError as e:
    print(f"Configuration validation error: {e}")
    print("Please check your environment variables and configuration.")
