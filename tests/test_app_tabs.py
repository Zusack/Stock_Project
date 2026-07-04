"""Tab index constants stay aligned with src/app.py."""

from src.services.tab_indices import (
    TAB_ASSISTANT,
    TAB_COMPARE,
    TAB_DASHBOARD,
    TAB_DATA_INGEST,
    TAB_DATA_MANAGEMENT,
    TAB_LEADERBOARD,
    TAB_LIVE,
    TAB_OPTIMIZATION,
    TAB_PORTFOLIO,
    TAB_SETTINGS,
    TAB_SINGLE_TICKER,
    TAB_STOCK_DETAIL,
    TAB_STRATEGY_BACKTESTS,
    TAB_WATCHLIST,
)


def test_tab_indices_order():
    assert TAB_DASHBOARD == 0
    assert TAB_PORTFOLIO == 1
    assert TAB_WATCHLIST == 2
    assert TAB_STOCK_DETAIL == 3
    assert TAB_SINGLE_TICKER == TAB_STOCK_DETAIL
    assert TAB_COMPARE == 4
    assert TAB_STRATEGY_BACKTESTS == 5
    assert TAB_LEADERBOARD == 6
    assert TAB_OPTIMIZATION == 7
    assert TAB_LIVE == 8
    assert TAB_DATA_MANAGEMENT == 9
    assert TAB_ASSISTANT == 10
    assert TAB_DATA_INGEST == TAB_DATA_MANAGEMENT
    assert TAB_SETTINGS == 11
