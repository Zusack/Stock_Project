"""Downsample time series for Flet charts (display only; backtests stay full resolution)."""

from __future__ import annotations

import bisect

from src.analysis.backtest_common import TickerChartSeries, TradeMarker

# Flet LineChart slows sharply past ~500 points per series (curves cost more).
MAX_CHART_POINTS = 384
# Trade/signal markers stay hoverable; only thin very dense technical/hybrid runs.
MAX_CHART_MARKERS = 120
CURVED_LINE_MAX_POINTS = 200


def sample_indices(values: list[float], max_points: int = MAX_CHART_POINTS) -> list[int]:
    """Pick indices that preserve shape (min/max per bucket + endpoints)."""
    n = len(values)
    if n <= max_points:
        return list(range(n))
    if max_points < 3:
        return [0, n - 1] if n > 1 else [0]

    buckets = max(1, (max_points - 2) // 2)
    bucket_size = n / buckets
    picked: set[int] = {0, n - 1}

    for b in range(buckets):
        start = int(b * bucket_size)
        end = int((b + 1) * bucket_size) if b < buckets - 1 else n
        if start >= end:
            continue
        segment = values[start:end]
        lo_off = segment.index(min(segment))
        hi_off = segment.index(max(segment))
        picked.add(start + lo_off)
        picked.add(start + hi_off)

    indices = sorted(picked)
    if len(indices) > max_points:
        stride = max(1, len(indices) // max_points)
        indices = [indices[i] for i in range(0, len(indices), stride)]
        if indices[-1] != n - 1:
            indices.append(n - 1)
        indices = sorted(set(indices))[:max_points]
    return indices


def downsample_series(
    dates: list[str],
    values: list[float],
    *,
    max_points: int = MAX_CHART_POINTS,
) -> tuple[list[str], list[float], list[int]]:
    """Return downsampled dates/values and the original indices kept."""
    if not values:
        return [], [], []
    indices = sample_indices(values, max_points)
    return (
        [dates[i] for i in indices],
        [values[i] for i in indices],
        indices,
    )


def _nearest_sample_position(original_x: int, sampled_indices: list[int]) -> int:
    """Map an original x index to a position in the downsampled array."""
    if not sampled_indices:
        return 0
    pos = bisect.bisect_left(sampled_indices, original_x)
    if pos >= len(sampled_indices):
        pos = len(sampled_indices) - 1
    elif pos > 0 and sampled_indices[pos] != original_x:
        if original_x - sampled_indices[pos - 1] <= sampled_indices[pos] - original_x:
            pos -= 1
    return pos


def remap_markers(
    markers: list[TradeMarker],
    sampled_indices: list[int],
    sampled_values: list[float],
) -> list[TradeMarker]:
    if not sampled_indices:
        return []
    remapped: list[TradeMarker] = []
    for marker in markers:
        pos = _nearest_sample_position(marker.x, sampled_indices)
        remapped.append(
            TradeMarker(
                x=pos,
                y=float(sampled_values[pos]) if pos < len(sampled_values) else marker.y,
                side=marker.side,
                tooltip=marker.tooltip,
            )
        )
    return remapped


def limit_markers(markers: list[TradeMarker], max_markers: int = MAX_CHART_MARKERS) -> list[TradeMarker]:
    """Thin only when necessary; preserves first/last for long signal-heavy runs."""
    if len(markers) <= max_markers:
        return markers
    stride = max(1, len(markers) // max_markers)
    thinned = [markers[i] for i in range(0, len(markers), stride)]
    if markers[0] not in thinned:
        thinned.insert(0, markers[0])
    if markers[-1] not in thinned:
        thinned.append(markers[-1])
    return thinned[:max_markers]


def prepare_ticker_series_for_display(
    series: TickerChartSeries,
    *,
    max_points: int = MAX_CHART_POINTS,
) -> tuple[TickerChartSeries, bool]:
    """Return a display-safe copy; second value is True if data was downsampled."""
    n = len(series.pct_returns)
    if n == 0:
        return series, False

    dates, pct, indices = downsample_series(series.dates, series.pct_returns, max_points=max_points)
    _, portfolio, _ = downsample_series(series.dates, series.portfolio_values, max_points=max_points)
    markers = limit_markers(remap_markers(series.markers, indices, pct))

    downsampled = n > len(pct)
    return (
        TickerChartSeries(
            ticker=series.ticker,
            dates=dates,
            pct_returns=pct,
            portfolio_values=portfolio,
            markers=markers,
        ),
        downsampled,
    )


def chart_note_for_sampling(original_len: int, display_len: int) -> str:
    if display_len >= original_len:
        return ""
    return f"Chart: {display_len} of {original_len} trading days shown for responsiveness."
