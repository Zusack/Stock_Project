from typing import Optional

from .writer import ResultsWriter
from .reader import ResultsReader
from .cleanup import ResultsCleanup

# Stop reason used when model lacks vision/tool-use for a prompt that requires it
ERROR_INCOMPATIBLE = "Error_Incompatible"

# Stop reasons that mean "incompatible" — pre-run, model lacks capability (vision/tool-use) per LM Studio metadata.
# Only Error_Incompatible gets purple/TODO; Error_Vision and Error_Tool are run failures for capable models → ERROR, evaluated.
INCOMPATIBLE_STOP_REASONS = frozenset({
    "Error_Incompatible",
})


def is_real_benchmark_error(stop_reason: Optional[str]) -> bool:
    """True if the stop_reason indicates a real run failure (load/crash/generic error).
    False for incompatible/skipped (vision/tool capability mismatch) — those should show as TODO/grey, not ERROR."""
    if not stop_reason:
        return False
    if stop_reason in INCOMPATIBLE_STOP_REASONS:
        return False
    return "Error" in stop_reason or stop_reason == "Error_Load"


class ResultsRepository(ResultsWriter, ResultsReader, ResultsCleanup):
    """
    Aggregates all result-related database operations.
    Inherited by DatabaseManager.
    """

    def ensure_incompatible_responses_recorded(self, system_id: int = 1) -> None:
        """For each (llm, prompt) pair that is respondent x active but not capability-matched
        and has no response, insert a response with stop_reason=Error_Incompatible and
        response_text='Incompatible with prompt'. Capability is from LM Studio metadata in DB:
        vision (Vis column) for image_comprehension prompts; trained_for_tool_use (Tools column) for tool_use/mcp."""
        for llm_id, prompt_id in self.get_incompatible_response_pairs(system_id):
            self.add_response(
                llm_id,
                prompt_id,
                {
                    "response_text": "Incompatible with prompt",
                    "stop_reason": ERROR_INCOMPATIBLE,
                },
                system_id=system_id,
            )

    def ensure_incompatible_evaluations_recorded(self, system_id: int = 1) -> None:
        """For each incompatible response (stop_reason=Error_Incompatible) that lacks an evaluation,
        insert evaluation with rating=0 so evaluators skip these (nothing to evaluate)."""
        if not self.connection:
            return
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT r.response_id, r.llm_id as respondent_llm_id, p.name as prompt_category
            FROM responses r
            JOIN prompts p ON r.prompt_id = p.prompt_id AND p.is_active = 1
            WHERE r.system_id = ? AND r.stop_reason = ?
        """, (system_id, ERROR_INCOMPATIBLE))
        incompatible_responses = cursor.fetchall()
        cursor.execute("""
            SELECT llm_id FROM llm_system_metrics
            WHERE system_id = ? AND is_available = 1
        """, (system_id,))
        evaluator_ids = [row["llm_id"] for row in cursor.fetchall()]
        rationale = "Incompatible with prompt (model lacks required vision/tool-use capability)."
        for row in incompatible_responses:
            response_id = row["response_id"]
            respondent_llm_id = row["respondent_llm_id"]
            prompt_category = row["prompt_category"]
            for evaluator_llm_id in evaluator_ids:
                if evaluator_llm_id == respondent_llm_id:
                    continue  # Skip self-evaluation
                cursor.execute(
                    "SELECT 1 FROM evaluations WHERE response_id = ? AND evaluator_llm_id = ?",
                    (response_id, evaluator_llm_id),
                )
                if cursor.fetchone() is None:
                    self.add_evaluation(response_id, evaluator_llm_id, {
                        "rating": 0,
                        "rationale": rationale,
                        "prompt_category": prompt_category,
                    })