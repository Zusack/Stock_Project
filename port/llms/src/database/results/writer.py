import sqlite3
from typing import Optional

REF_METRIC_KEYS = frozenset(("bleu_score", "rouge_score", "bert_score"))


def _filter_ref_metrics_for_prompt(connection, prompt_id: int, response_data: dict) -> dict:
    """Keep only the prompt's reference metric for standardized prompts; strip all for non-standardized."""
    filtered = dict(response_data)
    try:
        row = connection.execute(
            "SELECT benchmark_type, reference_metric FROM prompts WHERE prompt_id = ?",
            (prompt_id,),
        ).fetchone()
    except sqlite3.Error:
        # On error, strip all ref metrics to be safe
        for k in REF_METRIC_KEYS:
            filtered.pop(k, None)
        return filtered
    if not row:
        for k in REF_METRIC_KEYS:
            filtered.pop(k, None)
        return filtered
    bt = (row[0] or "text").strip().lower()
    ref_metric = (row[1] or "").strip().lower()
    if bt != "standardized" or ref_metric not in ("bleu", "rouge", "bert"):
        for k in REF_METRIC_KEYS:
            filtered.pop(k, None)
        return filtered
    # Keep only the metric that matches this prompt's reference_metric
    allowed_key = f"{ref_metric}_score"
    for k in REF_METRIC_KEYS:
        if k != allowed_key:
            filtered.pop(k, None)
    return filtered


class ResultsWriter:
    def add_response(self, llm_id: int, prompt_id: int, response_data: dict, system_id: int = 1) -> None:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        # Strip BLEU/ROUGE/BERT so they are only stored for standardized prompts with matching reference_metric
        response_data = _filter_ref_metrics_for_prompt(self.connection, prompt_id, response_data)
        columns = ["llm_id", "prompt_id", "system_id"]
        values = [llm_id, prompt_id, system_id]
        for key, val in response_data.items():
            columns.append(key)
            values.append(val)
        placeholders = ", ".join(["?"] * len(columns))
        column_names = ", ".join(columns)
        try:
            self.connection.execute(f"INSERT OR REPLACE INTO responses ({column_names}) VALUES ({placeholders})", tuple(values))
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error adding response: {e}")
            self.connection.rollback()

    def add_evaluation(self, response_id: int, evaluator_llm_id: int, eval_data: dict) -> None:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        # evaluation_time_sec may be None when merging from older DBs or for error paths; column accepts NULL.
        query = """
        INSERT OR REPLACE INTO evaluations 
        (response_id, evaluator_llm_id, prompt_category, rating, rationale, raw_response, evaluation_time_sec, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        try:
            self.connection.execute(query, (
                response_id,
                evaluator_llm_id,
                eval_data.get('prompt_category'),
                eval_data.get('rating'),
                eval_data.get('rationale'),
                eval_data.get('raw_response'),
                eval_data.get('evaluation_time_sec'),
            ))
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error adding evaluation: {e}")
            self.connection.rollback()

    def log_error(self, system_id: int, llm_id: Optional[int], process_step: str, message: str):
        if not self.connection: 
            try: self.connect()
            except: return 
        try:
            self.connection.execute("INSERT INTO error_logs (system_id, llm_id, process_step, error_message) VALUES (?, ?, ?, ?)", (system_id, llm_id, process_step, message))
            self.connection.commit()
        except sqlite3.Error: pass