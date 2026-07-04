"""Leaderboard segment filtering on cached scores."""

import pandas as pd

from src.analysis.leaderboard import LeaderboardSegment, filter_leaderboard_segment


def test_filter_all_scored():
    df = pd.DataFrame(
        [
            {"ticker": "A", "composite_score": 90, "pass_setup": True, "pass_pattern": False, "pattern_quality": 0.1, "risk_flag": ""},
            {"ticker": "B", "composite_score": 80, "pass_setup": False, "pass_pattern": True, "pattern_quality": 0.9, "risk_flag": ""},
        ]
    )
    out = filter_leaderboard_segment(df, LeaderboardSegment.ALL_SCORED, ":memory:", limit=None)
    assert len(out) == 2
    assert out.iloc[0]["ticker"] == "A"


def test_filter_candidates():
    df = pd.DataFrame(
        [
            {"ticker": "A", "composite_score": 90, "pass_setup": True, "pass_pattern": False, "pattern_quality": 0.1, "risk_flag": ""},
            {"ticker": "B", "composite_score": 80, "pass_setup": False, "pass_pattern": True, "pattern_quality": 0.9, "risk_flag": ""},
        ]
    )
    out = filter_leaderboard_segment(df, LeaderboardSegment.CANDIDATES, ":memory:", limit=10)
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "A"
