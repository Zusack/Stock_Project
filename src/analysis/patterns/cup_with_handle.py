"""Cup-with-handle pattern detection (O'Neil-style, point-in-time)."""



from __future__ import annotations



from dataclasses import dataclass



import numpy as np

import pandas as pd



from src.analysis.canslim_rulebook import CupWithHandleRules, get_rule_set





@dataclass(frozen=True)

class CupWithHandleMatch:

    pivot: float

    cup_low: float

    cup_high: float

    handle_low: float

    handle_high: float

    quality_score: float

    breakout_idx: int

    base_type: str = "cup_with_handle"





def _rolling_argmin_idx(arr: np.ndarray, start: int, end: int) -> int:

    window = arr[start : end + 1]

    if len(window) == 0:

        return start

    return start + int(np.argmin(window))





def scan_cup_with_handle_arrays(

    high: np.ndarray,

    low: np.ndarray,

    close: np.ndarray,

    volume: np.ndarray,

    end_idx: int,

    *,

    rules: CupWithHandleRules | None = None,

) -> CupWithHandleMatch | None:

    """

    Detect cup-with-handle ending at end_idx using only data[0:end_idx+1] (no future data).

    """

    rules = rules or get_rule_set().cup_with_handle

    n = end_idx + 1

    if n < 60:

        return None



    high = high[:n]

    low = low[:n]

    close = close[:n]

    volume = volume[:n]



    cup_min = max(35, rules.cup_min_weeks * 5)

    cup_max = min(n - 10, rules.cup_max_weeks * 5)

    handle_min = rules.handle_min_days



    best: CupWithHandleMatch | None = None

    best_score = 0.0



    for cup_len in range(cup_min, cup_max + 1, 5):

        cup_start = end_idx - cup_len

        if cup_start < 0:

            continue

        left_high = float(np.max(high[cup_start : cup_start + max(5, cup_len // 5)]))

        right_high = float(np.max(high[end_idx - max(5, cup_len // 5) : end_idx]))

        cup_high = max(left_high, right_high)

        if cup_high <= 0:

            continue



        low_start = cup_start + max(5, cup_len // 4)

        low_end = end_idx - handle_min - 5

        if low_end <= low_start:

            continue

        cup_low_idx = _rolling_argmin_idx(low, low_start, low_end)

        cup_low = float(low[cup_low_idx])

        depth = (cup_high - cup_low) / cup_high

        if depth < rules.cup_min_depth_pct or depth > rules.cup_max_depth_pct:

            continue



        mid = (cup_high + cup_low) / 2.0

        if cup_low_idx < cup_start + cup_len // 3 or cup_low_idx > cup_start + 2 * cup_len // 3:

            continue



        handle_start = max(cup_low_idx + 5, end_idx - 25)

        handle_end = end_idx

        if handle_end - handle_start < handle_min:

            continue



        handle_high = float(np.max(high[handle_start:handle_end]))

        handle_low = float(np.min(low[handle_start:handle_end]))

        if handle_high > cup_high * (1.0 + rules.handle_max_pct_below_cup_high):

            continue

        handle_depth = (handle_high - handle_low) / handle_high if handle_high > 0 else 1.0

        if handle_depth > rules.handle_max_depth_pct:

            continue



        handle_mid = (handle_high + handle_low) / 2.0

        if handle_mid < mid:

            continue



        pivot = handle_high * (1.0 + rules.pivot_buffer_pct)

        if close[end_idx] < pivot:

            continue



        vol_avg = float(np.mean(volume[max(0, end_idx - 50) : end_idx])) or 1.0

        vol_ratio = float(volume[end_idx]) / vol_avg

        if vol_ratio < rules.breakout_volume_mult:

            continue



        depth_score = 1.0 - abs(depth - 0.30) / 0.30

        vol_score = min(1.0, (vol_ratio - 1.0) / 1.0)

        handle_score = 1.0 - handle_depth / rules.handle_max_depth_pct

        quality = max(0.0, min(1.0, 0.4 * depth_score + 0.35 * vol_score + 0.25 * handle_score))



        if quality > best_score:

            best_score = quality

            best = CupWithHandleMatch(

                pivot=pivot,

                cup_low=cup_low,

                cup_high=cup_high,

                handle_low=handle_low,

                handle_high=handle_high,

                quality_score=round(quality, 4),

                breakout_idx=end_idx,

            )



    return best





def scan_cup_with_handle(

    ohlcv: pd.DataFrame,

    end_idx: int,

    *,

    rules: CupWithHandleRules | None = None,

) -> CupWithHandleMatch | None:

    """

    Detect cup-with-handle ending at or before end_idx (no future data).

    Expects columns: High, Low, Close, Volume (Adj Close used if no Close).

    """

    if ohlcv is None or len(ohlcv) < 60:

        return None



    close_col = "Close" if "Close" in ohlcv.columns else "Adj Close"

    high = ohlcv["High"].values if "High" in ohlcv.columns else ohlcv[close_col].values

    low = ohlcv["Low"].values if "Low" in ohlcv.columns else ohlcv[close_col].values

    close = ohlcv[close_col].values

    volume = ohlcv["Volume"].values if "Volume" in ohlcv.columns else np.ones(len(ohlcv))



    return scan_cup_with_handle_arrays(high, low, close, volume, end_idx, rules=rules)





def detect_cup_with_handle_series(

    ohlcv: pd.DataFrame,

    *,

    rules: CupWithHandleRules | None = None,

    min_warmup: int = 60,

) -> pd.DataFrame:

    """

    For each bar i, scan pattern using only data[:i+1].

    Returns DataFrame aligned to ohlcv index with Pass_Pattern, Pivot, Pattern_Quality.

    """

    rules = rules or get_rule_set().cup_with_handle

    n = len(ohlcv)

    pass_pat = np.zeros(n, dtype=bool)

    pivot_arr = np.full(n, np.nan)

    quality_arr = np.zeros(n)



    close_col = "Close" if "Close" in ohlcv.columns else "Adj Close"

    high = ohlcv["High"].values if "High" in ohlcv.columns else ohlcv[close_col].values

    low = ohlcv["Low"].values if "Low" in ohlcv.columns else ohlcv[close_col].values

    close = ohlcv[close_col].values

    volume = ohlcv["Volume"].values if "Volume" in ohlcv.columns else np.ones(n)



    for i in range(min_warmup, n):

        match = scan_cup_with_handle_arrays(high, low, close, volume, i, rules=rules)

        if match is not None:

            pass_pat[i] = True

            pivot_arr[i] = match.pivot

            quality_arr[i] = match.quality_score



    return pd.DataFrame(

        {

            "Pass_Pattern": pass_pat,

            "Pivot": pivot_arr,

            "Pattern_Quality": quality_arr,

            "Base_Type": np.where(pass_pat, "cup_with_handle", ""),

        },

        index=ohlcv.index,

    )

