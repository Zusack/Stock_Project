"""
Aggregation for Prompt Analyzer: per-(category, prompt) TPS across models and rating distributions.

Summary axis (see ``build_summary_axis_columns``):
- **One column per Leaderboard-visible category** (after ``hidden_categories``).
- **Single prompt in category:** TPS = one GenTPS value per model for that prompt; ratings = all
  valid ``Score`` rows for that (category, prompt).
- **Multiple prompts in category:** TPS = for each model, the **mean** of GenTPS across prompts in
  that category where that model has a value (one value per model → box across models). Ratings =
  **pooled** valid scores across all prompts in the category.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Literal, Optional

from . import model_analyzer_stats as mas


def _valid_score(score: Any) -> bool:
    return score is not None and isinstance(score, (int, float)) and score >= 0


def categories_visible(raw_data: list[dict], hidden_categories: set[str] | None) -> list[str]:
    """Sorted category names present in data, excluding hidden."""
    hidden = set(hidden_categories or [])
    cats: set[str] = set()
    for row in raw_data or []:
        d = dict(row) if hasattr(row, "keys") else row
        c = (d.get("Category") or "").strip() or "—"
        if c in hidden:
            continue
        cats.add(c)
    return sorted(cats)


def _tps_lists_by_cat_prompt(raw_data: list[dict]) -> dict[tuple[str, str], list[float]]:
    """
    One GenTPS per (category, prompt, model); aggregate to a list per (category, prompt).
    """
    per_model: dict[tuple[str, str, str], Optional[float]] = {}
    for d in raw_data or []:
        row = dict(d) if hasattr(d, "keys") else d
        cat = (row.get("Category") or "").strip() or "—"
        pname = (row.get("PromptName") or "").strip() or "—"
        model = mas.model_display_name(row)
        if not model:
            continue
        key = (cat, pname, model)
        if key not in per_model:
            g = row.get("GenTPS")
            per_model[key] = float(g) if g is not None else None
    out: dict[tuple[str, str], list[float]] = {}
    for (cat, pname, _m), tps in per_model.items():
        if tps is None:
            continue
        out.setdefault((cat, pname), []).append(tps)
    return out


def _scores_lists_by_cat_prompt(raw_data: list[dict]) -> dict[tuple[str, str], list[float]]:
    """All valid evaluation scores per (category, prompt)."""
    out: dict[tuple[str, str], list[float]] = {}
    for d in raw_data or []:
        row = dict(d) if hasattr(d, "keys") else d
        s = row.get("Score")
        if not _valid_score(s):
            continue
        cat = (row.get("Category") or "").strip() or "—"
        pname = (row.get("PromptName") or "").strip() or "—"
        out.setdefault((cat, pname), []).append(float(s))
    return out


def _scan_raw_maps(
    raw_data: list[dict],
    hidden_categories: set[str] | None,
) -> tuple[
    dict[tuple[str, str], list[float]],
    dict[tuple[str, str], list[float]],
    dict[tuple[str, str, str], float],
]:
    """
    Single pass: TPS lists per (category, prompt), scores per (category, prompt),
    and first non-null GenTPS per (category, prompt, model).
    Rows in hidden categories are skipped.
    """
    hidden = set(hidden_categories or [])
    per_model: dict[tuple[str, str, str], Optional[float]] = {}
    score_map: dict[tuple[str, str], list[float]] = {}

    for d in raw_data or []:
        row = dict(d) if hasattr(d, "keys") else d
        cat = (row.get("Category") or "").strip() or "—"
        if cat in hidden:
            continue
        pname = (row.get("PromptName") or "").strip() or "—"
        model = mas.model_display_name(row)
        if model:
            key = (cat, pname, model)
            if key not in per_model:
                g = row.get("GenTPS")
                per_model[key] = float(g) if g is not None else None
        s = row.get("Score")
        if _valid_score(s):
            score_map.setdefault((cat, pname), []).append(float(s))

    tps_triple: dict[tuple[str, str, str], float] = {}
    tps_map: dict[tuple[str, str], list[float]] = {}
    for (cat, pname, m), tps in per_model.items():
        if tps is None:
            continue
        tps_triple[(cat, pname, m)] = tps
        tps_map.setdefault((cat, pname), []).append(tps)

    return tps_map, score_map, tps_triple


def _mean_tps_per_model_for_category(
    tps_triple: dict[tuple[str, str, str], float],
    category: str,
    prompts: list[str],
) -> list[float]:
    """
    For each model that has any TPS in this category among ``prompts``, mean GenTPS across
    prompts where that model has a value (one number per model).
    """
    pset = set(prompts)
    models: set[str] = set()
    for (c, pname, m) in tps_triple:
        if c == category and pname in pset:
            models.add(m)
    out: list[float] = []
    for m in sorted(models, key=lambda x: x.lower()):
        vals = [
            tps_triple[(category, p, m)]
            for p in prompts
            if (category, p, m) in tps_triple
        ]
        if vals:
            out.append(statistics.mean(vals))
    return out


@dataclass(frozen=True)
class SummaryAxisColumn:
    """One X-axis slot on the Prompt Analyzer Summary (one visible category)."""

    kind: Literal["prompt", "category_agg"]
    category: str
    label: str
    prompt: Optional[str]
    tps_summary: Optional[dict[str, float]]
    rating_summary: Optional[dict[str, float]]


def build_summary_axis_columns(
    raw_data: list[dict],
    hidden_categories: set[str] | None = None,
) -> list[SummaryAxisColumn]:
    """
    Build one column per visible category: either a single prompt or an aggregate for multi-prompt
    categories. Order follows ``categories_visible`` (sorted category names).
    """
    if not raw_data:
        return []

    tps_map, score_map, tps_triple = _scan_raw_maps(raw_data, hidden_categories)
    cats = categories_visible(raw_data, hidden_categories)
    columns: list[SummaryAxisColumn] = []

    for category in cats:
        prompt_names: set[str] = set()
        for (c, pname) in list(tps_map.keys()) + list(score_map.keys()):
            if c == category:
                prompt_names.add(pname)
        prompts = sorted(prompt_names)
        if not prompts:
            continue

        if len(prompts) == 1:
            pname = prompts[0]
            tps_list = tps_map.get((category, pname), [])
            sc_list = score_map.get((category, pname), [])
            t_sum = mas.summarize_scores(tps_list) if tps_list else None
            r_sum = mas.summarize_scores(sc_list) if sc_list else None
            if t_sum is None and r_sum is None:
                continue
            columns.append(
                SummaryAxisColumn(
                    kind="prompt",
                    category=category,
                    label=pname,
                    prompt=pname,
                    tps_summary=t_sum,
                    rating_summary=r_sum,
                )
            )
        else:
            tps_list = _mean_tps_per_model_for_category(tps_triple, category, prompts)
            sc_list: list[float] = []
            for p in prompts:
                sc_list.extend(score_map.get((category, p), []))
            t_sum = mas.summarize_scores(tps_list) if tps_list else None
            r_sum = mas.summarize_scores(sc_list) if sc_list else None
            if t_sum is None and r_sum is None:
                continue
            n = len(prompts)
            label = f"{category} ({n} prompts)"
            columns.append(
                SummaryAxisColumn(
                    kind="category_agg",
                    category=category,
                    label=label,
                    prompt=None,
                    tps_summary=t_sum,
                    rating_summary=r_sum,
                )
            )

    return columns


def scores_for_prompt(raw_data: list[dict], category: str, prompt: str) -> list[float]:
    """All valid evaluation scores for one (category, prompt), same pooling as per-prompt summaries."""
    out: list[float] = []
    for d in raw_data or []:
        row = dict(d) if hasattr(d, "keys") else d
        s = row.get("Score")
        if not _valid_score(s):
            continue
        cat = (row.get("Category") or "").strip() or "—"
        pname = (row.get("PromptName") or "").strip() or "—"
        if cat != category or pname != prompt:
            continue
        out.append(float(s))
    return out


def prompt_score_lists_for_category(
    raw_data: list[dict],
    category: str,
    hidden_categories: set[str] | None = None,
) -> dict[str, list[float]]:
    """Map prompt name -> list of scores for that prompt in ``category`` (hidden category → empty)."""
    hidden = set(hidden_categories or [])
    if category in hidden:
        return {}
    score_map = _scores_lists_by_cat_prompt(raw_data)
    out: dict[str, list[float]] = {}
    for (cat, pname), lst in score_map.items():
        if cat == category:
            out[pname] = list(lst)
    return out


def prompt_summaries_for_category(
    raw_data: list[dict],
    category: str,
    hidden_categories: set[str] | None = None,
) -> list[tuple[str, Optional[dict[str, float]], Optional[dict[str, float]]]]:
    """
    For one category: list of (prompt_name, tps_summary, rating_summary).
    Summaries from mas.summarize_scores; None if no data.
    """
    hidden = set(hidden_categories or [])
    if category in hidden:
        return []

    tps_map = _tps_lists_by_cat_prompt(raw_data)
    score_map = _scores_lists_by_cat_prompt(raw_data)

    prompt_names: set[str] = set()
    for (cat, pname) in list(tps_map.keys()) + list(score_map.keys()):
        if cat != category:
            continue
        prompt_names.add(pname)

    result: list[tuple[str, Optional[dict], Optional[dict]]] = []
    for pname in sorted(prompt_names):
        tps_list = tps_map.get((category, pname), [])
        sc_list = score_map.get((category, pname), [])
        t_sum = mas.summarize_scores(tps_list) if tps_list else None
        r_sum = mas.summarize_scores(sc_list) if sc_list else None
        if t_sum is None and r_sum is None:
            continue
        result.append((pname, t_sum, r_sum))

    return result


def sort_prompt_rows(
    rows: list[tuple[str, Optional[dict[str, float]], Optional[dict[str, float]]]],
    sort_mode: str,
) -> list[tuple[str, Optional[dict[str, float]], Optional[dict[str, float]]]]:
    """
    sort_mode: name | median_tps | median_rating | n_tps | n_ratings
    """
    if sort_mode == "name":
        return sorted(rows, key=lambda x: x[0].lower())

    def med_tps(t: Optional[dict]) -> float:
        if not t:
            return -1.0
        return float(t.get("median", 0.0))

    def med_r(t: Optional[dict]) -> float:
        if not t:
            return -1.0
        return float(t.get("median", 0.0))

    def n_tps(t: Optional[dict]) -> float:
        if not t:
            return -1.0
        return float(t.get("count", 0.0))

    def n_rate(r: Optional[dict]) -> float:
        if not r:
            return -1.0
        return float(r.get("count", 0.0))

    if sort_mode == "median_tps":
        return sorted(rows, key=lambda x: med_tps(x[1]), reverse=True)
    if sort_mode == "median_rating":
        return sorted(rows, key=lambda x: med_r(x[2]), reverse=True)
    if sort_mode == "n_tps":
        return sorted(rows, key=lambda x: n_tps(x[1]), reverse=True)
    if sort_mode == "n_ratings":
        return sorted(rows, key=lambda x: n_rate(x[2]), reverse=True)
    return sorted(rows, key=lambda x: x[0].lower())


def ordered_models_worst_to_best(leaderboard: dict | None, raw_data: list[dict]) -> list[str]:
    """
    Model order for Prompt Analyzer scatter X-axis: lowest overall rank (smart_score) left,
    highest right. Leaderboard ordering is best-first; reverse for left-to-right worst→best.
    """
    if leaderboard and (leaderboard.get("model_data") or {}):
        return list(reversed(mas.ordered_model_names(leaderboard, raw_data)))
    names = mas.collect_model_names(raw_data)
    if not names:
        return []
    return sorted(names, key=lambda n: n.lower())


def _mean_score_per_model_all_prompts(raw_data: list[dict]) -> dict[str, float]:
    """Mean evaluation ``Score`` per model across all rows (all prompts) in ``raw_data``."""
    by_m: dict[str, list[float]] = {}
    for d in raw_data or []:
        row = dict(d) if hasattr(d, "keys") else d
        s = row.get("Score")
        if not _valid_score(s):
            continue
        m = mas.model_display_name(row)
        if not m:
            continue
        by_m.setdefault(m, []).append(float(s))
    return {m: statistics.mean(v) for m, v in by_m.items() if v}


def order_models_by_mean_overall_rating(
    raw_data: list[dict],
    *,
    models_subset: set[str] | None = None,
    worst_to_best: bool = True,
) -> list[str]:
    """
    Sort models by mean evaluation score over **all** prompts in ``raw_data`` (dataset mean).
    When ``worst_to_best`` is True: left = lowest mean, right = highest mean.
    Models that never receive a valid ``Score`` in ``raw_data`` sort **last** (right) with
    ``+inf`` as the sort key so they still appear if listed in ``models_subset``.
    """
    means = _mean_score_per_model_all_prompts(raw_data)
    if models_subset is None:
        names = set(mas.collect_model_names(raw_data))
    else:
        names = set(models_subset)
    if not names:
        return []

    def sort_key(name: str) -> tuple[float, str]:
        if name in means:
            return (means[name], name.lower())
        return (float("inf"), name.lower())

    ordered = sorted(names, key=sort_key)
    if not worst_to_best:
        ordered = list(reversed(ordered))
    return ordered


def model_tps_for_prompt(
    raw_data: list[dict], category: str, prompt: str
) -> dict[str, float]:
    """One GenTPS per model for this category+prompt (first non-null per model)."""
    per_model: dict[str, Optional[float]] = {}
    for d in raw_data or []:
        row = dict(d) if hasattr(d, "keys") else d
        if (row.get("Category") or "").strip() or "—" != category:
            continue
        if (row.get("PromptName") or "").strip() or "—" != prompt:
            continue
        m = mas.model_display_name(row)
        if not m:
            continue
        if m not in per_model:
            g = row.get("GenTPS")
            per_model[m] = float(g) if g is not None else None
    return {k: v for k, v in per_model.items() if v is not None}


def model_mean_scores_for_prompt(
    raw_data: list[dict], category: str, prompt: str
) -> dict[str, float]:
    """Mean evaluation score per model for this category+prompt."""
    by_m: dict[str, list[float]] = {}
    for d in raw_data or []:
        row = dict(d) if hasattr(d, "keys") else d
        s = row.get("Score")
        if not _valid_score(s):
            continue
        if (row.get("Category") or "").strip() or "—" != category:
            continue
        if (row.get("PromptName") or "").strip() or "—" != prompt:
            continue
        m = mas.model_display_name(row)
        if not m:
            continue
        by_m.setdefault(m, []).append(float(s))
    return {m: statistics.mean(v) for m, v in by_m.items() if v}


def rank_by_mean_score_on_prompt(mean_by_model: dict[str, float]) -> dict[str, int]:
    """Rank 1 = highest mean score among models; ties get sequential ranks."""
    if not mean_by_model:
        return {}
    ordered = sorted(mean_by_model.items(), key=lambda x: (-x[1], x[0].lower()))
    ranks: dict[str, int] = {}
    for i, (name, _s) in enumerate(ordered):
        ranks[name] = i + 1
    return ranks


def linear_regression_r2(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """Return (slope, intercept, r_squared) or None."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    ss_xx = sum((x - mx) ** 2 for x in xs)
    ss_yy = sum((y - my) ** 2 for y in ys)
    if ss_xx < 1e-18 or ss_yy < 1e-18:
        return None
    ss_xy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    slope = ss_xy / ss_xx
    intercept = my - slope * mx
    r = ss_xy / ((ss_xx * ss_yy) ** 0.5)
    r2 = r * r
    return (slope, intercept, r2)
