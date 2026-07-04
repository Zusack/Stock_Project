from src.utils.model_display_label import is_provider_detected


class ProgressCalculator:
    """
    Central logic for calculating benchmark and sequence progress.
    Ensures consistency between RunnerView and EvaluatorView.
    """
    
    @staticmethod
    def calculate_targets(all_llms: list[dict], total_prompts: int, all_llms_as_evaluators: bool = False, num_standardized_prompts: int = 0):
        """
        Calculates the total number of operations required for a full run.

        Args:
            all_llms: List of dicts representing models (must contain system flags)
            total_prompts: Count of active prompts
            all_llms_as_evaluators: If True, count every available LLM as an evaluator (for full evaluation queue progress)
            num_standardized_prompts: Count of active prompts that are standardized (BLEU/ROUGE/BERT). These are
                "evaluation complete" when the benchmark runs; no LLM evaluator run. Reduces total_eval_ops.

        Returns:
            tuple: (total_bench_ops, total_eval_ops)
        """
        # Filter for models active ON THIS SYSTEM
        available_models = []
        for m in all_llms:
            is_avail = is_provider_detected(m)
            if is_avail:
                available_models.append(m)

        # 1. Benchmark Operations (Respondents * Prompts)
        num_respondents = 0
        for m in available_models:
            is_resp = bool(m.get('sys_is_respondent')) or bool(m.get('is_respondent'))
            if is_resp:
                num_respondents += 1

        total_bench_ops = num_respondents * total_prompts

        # 2. Evaluation Operations
        # For every Evaluator, judge every Response from every OTHER model.
        # If all_llms_as_evaluators: count every available LLM; else only those with is_evaluator.
        total_eval_ops = 0
        num_evaluators = 0
        num_evaluators_who_are_respondents = 0
        for m in available_models:
            is_eval = all_llms_as_evaluators or bool(m.get('sys_is_evaluator')) or bool(m.get('is_evaluator'))
            if is_eval:
                num_evaluators += 1
                is_resp = bool(m.get('sys_is_respondent')) or bool(m.get('is_respondent'))
                if is_resp:
                    num_evaluators_who_are_respondents += 1
                ops = total_bench_ops
                if is_resp:
                    ops -= total_prompts
                total_eval_ops += ops
        # Standardized prompts are "eval complete" when benchmark runs; no evaluator run. Subtract those slots
        # and add one "op" per (respondent, standardized prompt) so progress = LLM_evals + standardized_count.
        if num_standardized_prompts > 0 and total_eval_ops > 0:
            slots_standardized = num_standardized_prompts * (num_evaluators * num_respondents - num_evaluators_who_are_respondents)
            total_eval_ops = total_eval_ops - slots_standardized + (num_respondents * num_standardized_prompts)
            total_eval_ops = max(0, total_eval_ops)

        return total_bench_ops, total_eval_ops

    @staticmethod
    def calculate_progress_percent(done: int, total: int) -> float:
        if total <= 0: return 0.0
        pct = done / total
        return min(1.0, max(0.0, pct))