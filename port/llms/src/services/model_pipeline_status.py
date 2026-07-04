"""Pipeline completion labels for Model Manager (Run / Judge vs active prompts, provider detection)."""
from __future__ import annotations

from typing import Any, Dict, List, Set

from src.database.results import is_real_benchmark_error
from src.utils.model_display_label import is_provider_detected


def _missing_bench_llm_ids(missing_responses: List[tuple]) -> Set[int]:
    return {int(row[0]) for row in missing_responses}


def _missing_eval_respondent_ids(missing_evals: List[tuple]) -> Set[int]:
    # get_missing_evaluations: index 8 = respondent_llm_id
    return {int(row[8]) for row in missing_evals if len(row) > 8}


def build_pipeline_status(
    models: List[Dict[str, Any]],
    summary: Dict[tuple, Any],
    active_prompt_ids: Set[int],
    missing_responses: List[tuple],
    missing_evals: List[tuple],
) -> Dict[int, Dict[str, Any]]:
    miss_b = _missing_bench_llm_ids(missing_responses)
    miss_e = _missing_eval_respondent_ids(missing_evals)
    n_prompts = len(active_prompt_ids)

    out: Dict[int, Dict[str, Any]] = {}
    for m in models:
        lid = int(m["llm_id"])
        is_resp = bool(m.get("sys_is_respondent") or m.get("is_respondent"))
        det = is_provider_detected(m)

        completed_good = 0
        if n_prompts and is_resp:
            for pid in active_prompt_ids:
                key = (lid, pid)
                if key not in summary:
                    continue
                if is_real_benchmark_error(summary.get(key)):
                    continue
                completed_good += 1

        bench_pct = None
        if is_resp and n_prompts > 0:
            bench_pct = min(1.0, completed_good / n_prompts)

        has_miss_b = lid in miss_b if is_resp else False
        has_miss_e = lid in miss_e if is_resp else False

        # Not Started: undetected (no live provider / missing GGUF), or not on Run roster,
        # or on roster but no successful completions yet (see below).
        if not det or not is_resp:
            pipeline = "not_started"
        elif not has_miss_b and not has_miss_e:
            pipeline = "complete"
        elif completed_good == 0:
            pipeline = "not_started"
        else:
            pipeline = "in_progress"

        out[lid] = {
            "pipeline": pipeline,
            "bench_pct": bench_pct,
            "has_missing_bench": has_miss_b,
            "has_missing_eval": has_miss_e,
            "completed_good": completed_good,
        }
    return out
