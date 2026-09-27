"""Runtime settings, read from environment variables (and .env if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return cast(raw)


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", "postgresql+psycopg://localhost/polyarb"))

    scan_interval_sec: int = field(default_factory=lambda: _env("SCAN_INTERVAL_SEC", 60, int))
    rematch_interval_sec: int = field(default_factory=lambda: _env("REMATCH_INTERVAL_SEC", 900, int))

    min_return: float = field(default_factory=lambda: _env("MIN_RETURN", 0.005, float))
    min_match_confidence: float = field(default_factory=lambda: _env("MIN_MATCH_CONFIDENCE", 0.55, float))
    risk_free_rate: float = field(default_factory=lambda: _env("RISK_FREE_RATE", 0.04, float))

    kalshi_base_fee: float = field(default_factory=lambda: _env("KALSHI_BASE_FEE", 0.07, float))
    max_days_to_resolution: float = field(default_factory=lambda: _env("MAX_DAYS_TO_RESOLUTION", 400.0, float))
    min_days_for_annualization: float = field(default_factory=lambda: _env("MIN_DAYS_FOR_ANNUALIZATION", 1.0, float))

    log_all_pairs: bool = field(default_factory=lambda: _env("LOG_ALL_PAIRS", False, bool))
    log_gross_floor: float = field(default_factory=lambda: _env("LOG_GROSS_FLOOR", -0.02, float))

    poly_min_liquidity: float = field(default_factory=lambda: _env("POLY_MIN_LIQUIDITY", 500.0, float))
    kalshi_rps: float = field(default_factory=lambda: _env("KALSHI_RPS", 10.0, float))
    http_timeout_sec: float = field(default_factory=lambda: _env("HTTP_TIMEOUT_SEC", 20.0, float))


def get_settings() -> Settings:
    return Settings()
