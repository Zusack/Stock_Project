"""Leaderboard ranking helpers."""

import pandas as pd

from src.analysis.leaderboard import LeaderboardSegment, rows_to_display_df


def test_rows_to_display_empty():
    assert rows_to_display_df(pd.DataFrame()).empty


def test_rows_to_display_rename():
    df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "composite_score": 80.0,
                "canslim_score": 6,
                "pattern_quality": 0.5,
                "rs_pct": 10.0,
                "volume_ratio": 1.2,
                "near_high_pct": 90.0,
                "pass_setup": True,
                "pass_pattern": False,
                "risk_flag": "",
                "latest_price": 150.0,
                "sector": "Tech",
            }
        ]
    )
    out = rows_to_display_df(df)
    assert "Ticker" in out.columns
    assert "Composite" in out.columns
    assert out.iloc[0]["CANSLM"] == "6/6"
    assert out.iloc[0]["Setup"] == "Yes"


def test_leaderboard_segment_values():
    assert LeaderboardSegment.ALL_SCORED.value == "all_scored"
    assert LeaderboardSegment.CANDIDATES.value == "candidates"
