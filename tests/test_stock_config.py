"""StockConfig property round-trips."""

from __future__ import annotations

from src.services.stock_config import stock_config


def test_lm_studio_defaults():
    cfg = stock_config()
    assert cfg.lm_studio_base_url.startswith("http")
    assert cfg.lm_studio_timeout_sec >= 5.0


def test_ai_settings_round_trip():
    cfg = stock_config()
    original_enabled = cfg.lm_studio_enabled
    original_model = cfg.lm_studio_model
    try:
        cfg.lm_studio_enabled = True
        cfg.lm_studio_model = "_test_model_"
        assert cfg.lm_studio_enabled is True
        assert cfg.lm_studio_model == "_test_model_"
    finally:
        cfg.lm_studio_enabled = original_enabled
        cfg.lm_studio_model = original_model


def test_market_ticker_strips_and_uppercases():
    cfg = stock_config()
    original = cfg.market_ticker
    try:
        cfg.market_ticker = " spy "
        assert cfg.market_ticker == "SPY"
    finally:
        cfg.market_ticker = original
