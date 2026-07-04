"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Synthetic cup-like OHLCV (~120 bars)."""
    n = 140
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.concatenate(
        [
            np.linspace(100, 120, 40),
            np.linspace(120, 95, 50),
            np.linspace(95, 118, 30),
            np.linspace(118, 122, 20),
        ]
    )[:n]
    high = close * 1.01
    low = close * 0.99
    volume = np.full(n, 1_000_000.0)
    volume[-5:] = 2_500_000.0
    return pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=dates,
    )
