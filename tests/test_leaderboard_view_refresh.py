"""Leaderboard tab refresh must not trigger full rescans on tab open."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.analysis.leaderboard import LeaderboardSegment
from src.views.leaderboard_view import LeaderboardView, _universe_key


def _minimal_view(*, cache: pd.DataFrame | None = None) -> LeaderboardView:
    """Build a view instance without running full Flet control tree setup."""
    view = object.__new__(LeaderboardView)
    view.page_ref = MagicMock()
    view._segment = LeaderboardSegment.CANDIDATES
    view._running = False
    view._scored_cache = cache
    view._scored_at = None
    view._scored_universe_key = (
        _universe_key(["AAPL"], full_universe=False) if cache is not None else None
    )
    view._score_full_universe = False
    view.ticker_filter = MagicMock(value="")
    view.limit_field = MagicMock(value="50")
    return view


@pytest.fixture
def view():
    return _minimal_view(
        cache=pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "composite_score": 75.0,
                    "canslim_score": 5,
                    "pattern_quality": 0.5,
                    "rs_pct": 8.0,
                    "volume_ratio": 1.1,
                    "near_high_pct": 88.0,
                    "pass_setup": True,
                    "pass_pattern": False,
                    "risk_flag": "",
                    "latest_price": 180.0,
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                }
            ]
        )
    )


def test_fetch_data_does_not_call_build_leaderboard(view):
    with patch("src.views.leaderboard_view.run_leaderboard_build") as mock_build:
        data = view._fetch_data()
        mock_build.assert_not_called()
    assert data["needs_refresh"] is False
    assert data["df"] is not None


def test_refresh_data_async_without_auto_refresh_uses_fetch(view):
    with patch.object(view, "_start_refresh") as mock_start:
        with patch("src.views.base_view.BaseView.refresh_data_async") as mock_super:
            with patch("src.views.leaderboard_view.stock_config") as mock_cfg:
                mock_cfg.return_value.leaderboard_auto_refresh = False
                view.refresh_data_async(label="tab:Leaderboard")
                mock_start.assert_not_called()
                mock_super.assert_called_once()


def test_segment_change_without_cache_does_not_rescan():
    view = _minimal_view(cache=None)
    with patch.object(view, "_start_refresh") as mock_start:
        with patch.object(view, "_apply_data") as mock_apply:
            event = MagicMock()
            event.control.selected = [LeaderboardSegment.WATCHLIST.value]
            view._on_segment_change(event)
            mock_start.assert_not_called()
            mock_apply.assert_called_once()
            assert mock_apply.call_args[0][0]["needs_refresh"] is True


def test_universe_key_changes_when_focus_changes():
    k1 = _universe_key(["AAPL", "MSFT"], full_universe=False)
    k2 = _universe_key(["AAPL"], full_universe=False)
    assert k1 != k2


def test_refresh_data_delegates_to_async(view):
    with patch.object(view, "refresh_data_async") as mock_async:
        view.refresh_data()
        mock_async.assert_called_once_with(label="leaderboard")
