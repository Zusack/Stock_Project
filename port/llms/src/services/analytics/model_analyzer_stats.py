"""
Pure aggregation for the Model Analyzer tab: TPS by category, rating quartiles by category and by prompt.
"""
from __future__ import annotations

import re
import statistics
from typing import Any, Optional

from src.utils.model_display_label import raw_data_model_key


def model_display_name(row: dict) -> str:
    return raw_data_model_key(row).strip()


def _valid_score(score: Any) -> bool:
    return score is not None and isinstance(score, (int, float)) and score >= 0


def summarize_scores(values: list[float]) -> Optional[dict[str, float]]:
    """Return min, q1, median, q3, max, mean, count. None if empty."""
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    n = len(xs)
    mean = statistics.mean(xs)
    mn, mx = xs[0], xs[-1]
    if n == 1:
        return {
            "min": mn,
            "q1": mn,
            "median": mn,
            "q3": mn,
            "max": mx,
            "mean": mean,
            "count": float(n),
        }
    median = statistics.median(xs)
    try:
        qs = statistics.quantiles(xs, n=4, method="inclusive")
        q1, q3 = qs[0], qs[2]
    except (statistics.StatisticsError, ValueError):
        q1 = xs[n // 4]
        q3 = xs[(3 * n) // 4] if n > 1 else xs[-1]
    return {
        "min": mn,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": mx,
        "mean": mean,
        "count": float(n),
    }


def histogram_counts(values: list[float], bin_min: int = 0, bin_max: int = 10) -> list[int]:
    """
    Count values per integer rating bin after rounding to nearest int and clamping to [bin_min, bin_max].
    Returns a list of length (bin_max - bin_min + 1), index 0 = bin_min.
    """
    n_bins = bin_max - bin_min + 1
    counts = [0] * n_bins
    for v in values:
        b = int(round(float(v)))
        b = max(bin_min, min(bin_max, b))
        counts[b - bin_min] += 1
    return counts


def extended_summary(values: list[float]) -> Optional[dict[str, float]]:
    """
    Like summarize_scores plus mode (lowest if multimodal), sample stdev/variance, and IQR.
    stdev/variance are 0.0 when n < 2.
    """
    base = summarize_scores(values)
    if base is None:
        return None
    xs = sorted(float(x) for x in values)
    n = len(xs)
    modes = statistics.multimode(xs)
    mode_val = float(min(modes)) if modes else float(xs[0])
    if n < 2:
        stdev = 0.0
        variance = 0.0
    else:
        stdev = float(statistics.stdev(xs))
        variance = float(statistics.variance(xs))
    iqr = float(base["q3"]) - float(base["q1"])
    out = dict(base)
    out["mode"] = mode_val
    out["stdev"] = stdev
    out["variance"] = variance
    out["iqr"] = iqr
    return out


def rows_for_model(raw_data: list[dict], model: str) -> list[dict]:
    """Rows whose display name matches `model`."""
    out = []
    for row in raw_data:
        d = dict(row) if hasattr(row, "keys") else row
        if model_display_name(d) == model:
            out.append(d)
    return out


def dedupe_tps_by_category_prompt(rows: list[dict]) -> dict[tuple[str, str], Optional[float]]:
    """
    One GenTPS per (Category, PromptName). First non-null wins (evaluator duplicates share the same TPS).
    """
    tps_map: dict[tuple[str, str], Optional[float]] = {}
    for d in rows:
        cat = (d.get("Category") or "").strip() or "—"
        pname = (d.get("PromptName") or "").strip() or "—"
        key = (cat, pname)
        if key not in tps_map:
            g = d.get("GenTPS")
            tps_map[key] = float(g) if g is not None else None
    return tps_map


def category_mean_tps(rows: list[dict]) -> list[tuple[str, float]]:
    """
    For each prompt category, mean TPS across prompts (after per-prompt dedupe).
    Categories with no TPS use 0.0 for the bar (caller may hide).
    """
    tps_map = dedupe_tps_by_category_prompt(rows)
    by_cat: dict[str, list[float]] = {}
    for (cat, _pname), tps in tps_map.items():
        if tps is None:
            continue
        by_cat.setdefault(cat, []).append(tps)
    result = []
    for cat in sorted(by_cat.keys()):
        vals = by_cat[cat]
        result.append((cat, statistics.mean(vals) if vals else 0.0))
    return result


def scores_by_category(rows: list[dict]) -> dict[str, list[float]]:
    """All valid scores per Category (includes multiple evaluator rows)."""
    out: dict[str, list[float]] = {}
    for d in rows:
        s = d.get("Score")
        if not _valid_score(s):
            continue
        cat = (d.get("Category") or "").strip() or "—"
        out.setdefault(cat, []).append(float(s))
    return out


def category_rating_summaries(rows: list[dict]) -> list[tuple[str, Optional[dict[str, float]]]]:
    """Ordered list of (category, summary or None)."""
    by_cat = scores_by_category(rows)
    result = []
    for cat in sorted(by_cat.keys()):
        result.append((cat, summarize_scores(by_cat[cat])))
    return result


def unique_prompt_names_in_category(rows: list[dict], category: str) -> list[str]:
    names = set()
    for d in rows:
        c = (d.get("Category") or "").strip() or "—"
        if c != category:
            continue
        pname = (d.get("PromptName") or "").strip()
        if pname:
            names.add(pname)
    return sorted(names)


def prompt_rating_summaries(rows: list[dict], category: str) -> list[tuple[str, Optional[dict[str, float]]]]:
    """Per-prompt rating summaries within a category."""
    by_prompt: dict[str, list[float]] = {}
    for d in rows:
        c = (d.get("Category") or "").strip() or "—"
        if c != category:
            continue
        s = d.get("Score")
        if not _valid_score(s):
            continue
        pname = (d.get("PromptName") or "").strip() or "—"
        by_prompt.setdefault(pname, []).append(float(s))
    result = []
    for pname in sorted(by_prompt.keys()):
        result.append((pname, summarize_scores(by_prompt[pname])))
    return result


def prompt_tps_values(rows: list[dict], category: str) -> list[tuple[str, float]]:
    """Deduped TPS per prompt name in category (for drill-down consistency)."""
    tps_map = dedupe_tps_by_category_prompt(rows)
    out: list[tuple[str, float]] = []
    for (cat, pname), tps in sorted(tps_map.items()):
        if cat != category or tps is None:
            continue
        out.append((pname, tps))
    return out


def collect_model_names(raw_data: list[dict]) -> list[str]:
    """Unique model display names from raw rows."""
    names: set[str] = set()
    for row in raw_data:
        d = dict(row) if hasattr(row, "keys") else row
        m = model_display_name(d)
        if m:
            names.add(m)
    return sorted(names)


def parse_params_for_sort(param_str: Optional[str]) -> float:
    """Numeric parameter count for sorting (same idea as ResultsReader._parse_params)."""
    if not param_str or param_str == "N/A":
        return 0.0
    try:
        if "x" in param_str.lower():
            parts = re.findall(r"(\d+\.?\d*)", param_str)
            if len(parts) >= 2:
                return float(parts[0]) * float(parts[1])
        nums = re.findall(r"(\d+\.?\d*)", param_str)
        if nums:
            return float(nums[0])
    except (TypeError, ValueError):
        pass
    return 0.0


def visible_segment_total(
    leaderboard: Optional[dict], model_name: str, hidden_categories: Optional[set]
) -> float:
    """Sum of stacked-bar segment scores for categories currently visible on the Leaderboard chart."""
    if not leaderboard:
        return 0.0
    md = (leaderboard.get("model_data") or {}).get(model_name)
    if not md:
        return 0.0
    all_cats = set(leaderboard.get("all_categories") or [])
    hidden = set(hidden_categories or [])
    visible = all_cats - hidden
    total = 0.0
    for seg in md.get("segments") or []:
        cat = seg.get("category")
        if cat in visible:
            total += float(seg.get("score") or 0.0)
    return total


def sort_model_names(
    names: list[str],
    mode: str,
    leaderboard: Optional[dict],
    hidden_categories: Optional[set],
    meta_map: dict[str, dict],
) -> list[str]:
    """
    Sort model display names. Modes:
    - visible_chart: sum of segment scores for visible categories (matches Leaderboard stacked bar order).
    - smart_score: leaderboard table / All-Rounder ordering.
    - alphabetical: A–Z by name.
    - params: parameter count descending (from model metadata).
    - tps, quality, win_rate: from leaderboard model_data.
    """
    if not names:
        return []
    md = (leaderboard or {}).get("model_data") or {}

    def lb_info(name: str) -> dict:
        return md.get(name) or {}

    if mode == "alphabetical":
        return sorted(names, key=lambda n: n.lower())

    if mode == "visible_chart":
        return sorted(
            names,
            key=lambda n: (-visible_segment_total(leaderboard, n, hidden_categories), n.lower()),
        )

    if mode == "smart_score":
        def smart_key(n: str) -> tuple:
            s = lb_info(n).get("smart_score")
            if s is None:
                return (1, 0.0, n.lower())
            return (0, -float(s), n.lower())

        return sorted(names, key=smart_key)

    if mode == "params":

        def param_key(n: str) -> tuple:
            meta = meta_map.get(n) or {}
            p = parse_params_for_sort(meta.get("params_string"))
            return (-p, n.lower())

        return sorted(names, key=param_key)

    if mode == "tps":
        return sorted(
            names,
            key=lambda n: (-(lb_info(n).get("speed") or 0.0), n.lower()),
        )

    if mode == "quality":
        return sorted(
            names,
            key=lambda n: (-(lb_info(n).get("quality") or 0.0), n.lower()),
        )

    if mode == "win_rate":
        return sorted(
            names,
            key=lambda n: (-(lb_info(n).get("win_rate") or 0.0), n.lower()),
        )

    # Unknown mode: fall back to visible chart
    return sort_model_names(names, "visible_chart", leaderboard, hidden_categories, meta_map)


def ordered_model_names(leaderboard: Optional[dict], raw_data: list[dict]) -> list[str]:
    """Backward-compatible: smart_score ordering (leaderboard table)."""
    names = collect_model_names(raw_data)
    return sort_model_names(names, "smart_score", leaderboard, set(), {})
