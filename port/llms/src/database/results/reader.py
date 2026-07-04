import sqlite3
import re
import math
from typing import Optional, Any

class ResultsReader:
    # --- Internal Helpers ---
    def _parse_params(self, param_str):
        if not param_str or param_str == "N/A": return 0.0
        try:
            if 'x' in param_str.lower():
                parts = re.findall(r"(\d+\.?\d*)", param_str)
                if len(parts) >= 2: return float(parts[0]) * float(parts[1])
            nums = re.findall(r"(\d+\.?\d*)", param_str)
            if nums: return float(nums[0])
        except: pass
        return 0.0

    # --- Fetch Data ---
    
    def get_system_counts(self, system_id: int) -> dict:
        """Count only responses/evaluations for active prompts so runner progress matches active prompt set."""
        if not self.connection: raise sqlite3.Error("DB not open")
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM responses r JOIN prompts p ON r.prompt_id = p.prompt_id AND p.is_active = 1 WHERE r.system_id = ?",
            (system_id,),
        )
        resp_count = cursor.fetchone()[0]
        cursor.execute(
            "SELECT COUNT(e.evaluation_id) FROM evaluations e JOIN responses r ON e.response_id = r.response_id JOIN prompts p ON r.prompt_id = p.prompt_id AND p.is_active = 1 WHERE r.system_id = ?",
            (system_id,),
        )
        eval_count = cursor.fetchone()[0]
        return {'responses': resp_count, 'evaluations': eval_count}

    def get_evaluations_count_for_roster(
        self, system_id: int, evaluator_llm_ids: list[int], respondent_llm_ids: list[int]
    ) -> int:
        """Count evaluations only for enabled evaluators and respondents (active prompts). Used for Evaluator tab progress."""
        if not self.connection:
            raise sqlite3.Error("DB not open")
        if not evaluator_llm_ids or not respondent_llm_ids:
            return 0
        placeholders_e = ",".join("?" * len(evaluator_llm_ids))
        placeholders_r = ",".join("?" * len(respondent_llm_ids))
        query = f"""
        SELECT COUNT(e.evaluation_id) FROM evaluations e
        JOIN responses r ON e.response_id = r.response_id
        JOIN prompts p ON r.prompt_id = p.prompt_id AND p.is_active = 1
        WHERE r.system_id = ? AND e.evaluator_llm_id IN ({placeholders_e}) AND r.llm_id IN ({placeholders_r})
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id, *evaluator_llm_ids, *respondent_llm_ids))
        return cursor.fetchone()[0]

    def get_all_responses_summary(self, system_id: int = 1) -> dict[tuple[int, int], str]:
        """Returns (llm_id, prompt_id) -> stop_reason only for responses whose prompt is active, so runner/inspector completion counts match active prompts."""
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT r.llm_id, r.prompt_id, r.stop_reason FROM responses r JOIN prompts p ON r.prompt_id = p.prompt_id AND p.is_active = 1 WHERE r.system_id = ?",
            (system_id,),
        )
        return {(row['llm_id'], row['prompt_id']): row['stop_reason'] for row in cursor.fetchall()}

    def get_response_statuses_for_llm(self, llm_id: int, system_id: int = 1) -> dict[int, str]:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        cursor.execute("SELECT prompt_id, stop_reason FROM responses WHERE llm_id = ? AND system_id = ?", (llm_id, system_id))
        return {row['prompt_id']: row['stop_reason'] for row in cursor.fetchall()}

    def get_response(self, llm_id: int, prompt_id: int, system_id: int = 1) -> Optional[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        query = """
        SELECT r.*, m.sys_ram_avail_pre_gb, m.sys_ram_avail_post_gb, m.vram_avail_pre_gb, m.vram_avail_post_gb, m.load_time_sec AS metrics_load_time_sec, p.prompt_text, p.benchmark_type, p.reference_metric
        FROM responses r
        JOIN llms l ON r.llm_id = l.llm_id
        JOIN prompts p ON r.prompt_id = p.prompt_id
        LEFT JOIN llm_system_metrics m ON l.llm_id = m.llm_id AND m.system_id = ?
        WHERE r.llm_id = ? AND r.prompt_id = ? AND r.system_id = ?
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id, llm_id, prompt_id, system_id))
        return cursor.fetchone()
        
    def get_evaluation_detail(self, evaluator_id: int, respondent_id: int, prompt_id: int, system_id: int = 1) -> Optional[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("DB not open")
        
        cursor = self.connection.cursor()
        
        # Select evaluation fields including optional time/category (merge-safe: columns may be missing in old DBs)
        try:
            query = """
            SELECT p.prompt_text, p.expected_response, p.assessment_text, p.scoring_criteria, r.response_text,
                e.rating, e.rationale, e.raw_response, e.prompt_category, e.timestamp AS evaluation_timestamp, e.evaluation_time_sec
            FROM evaluations e
            JOIN responses r ON e.response_id = r.response_id
            JOIN prompts p ON r.prompt_id = p.prompt_id
            WHERE e.evaluator_llm_id = ? AND r.llm_id = ? AND r.prompt_id = ? AND r.system_id = ?
            """
            cursor.execute(query, (evaluator_id, respondent_id, prompt_id, system_id))
            return cursor.fetchone()
        except sqlite3.OperationalError as e:
            # If optional columns don't exist (old DB), fall back to minimal columns
            if "no such column" in str(e).lower():
                query = """
                SELECT p.prompt_text, p.expected_response, p.assessment_text, p.scoring_criteria, r.response_text,
                    e.rating, e.rationale, NULL AS raw_response, e.prompt_category, e.timestamp AS evaluation_timestamp, NULL AS evaluation_time_sec
                FROM evaluations e
                JOIN responses r ON e.response_id = r.response_id
                JOIN prompts p ON r.prompt_id = p.prompt_id
                WHERE e.evaluator_llm_id = ? AND r.llm_id = ? AND r.prompt_id = ? AND r.system_id = ?
                """
                cursor = self.connection.cursor()
                cursor.execute(query, (evaluator_id, respondent_id, prompt_id, system_id))
                return cursor.fetchone()
            else:
                raise

    def get_standardized_evaluation_detail(self, respondent_id: int, prompt_id: int, system_id: int = 1) -> Optional[sqlite3.Row]:
        """For standardized prompts: return prompt_text, expected_response, response_text, rating (0-10), assessment_text, scoring_criteria, rationale.
        When the benchmark errored (respondent-side), rating is 0 and rationale indicates benchmark error."""
        if not self.connection:
            raise sqlite3.Error("DB not open")
        query = """
        SELECT p.prompt_text, p.expected_response, p.assessment_text, p.scoring_criteria, r.response_text,
            CASE
                WHEN p.reference_metric = 'bleu' AND r.bleu_score IS NOT NULL THEN ROUND(r.bleu_score * 10, 2)
                WHEN p.reference_metric = 'rouge' AND r.rouge_score IS NOT NULL THEN ROUND(r.rouge_score * 10, 2)
                WHEN p.reference_metric = 'bert' AND r.bert_score IS NOT NULL THEN ROUND(r.bert_score * 10, 2)
                WHEN r.stop_reason IS NOT NULL AND r.stop_reason != 'Error_Incompatible'
                  AND (r.stop_reason LIKE '%Error%' OR r.stop_reason = 'Error_Load')
                THEN 0
                ELSE NULL
            END AS rating,
            CASE
                WHEN p.reference_metric = 'bleu' AND r.bleu_score IS NOT NULL THEN 'Standardized (BLEU)'
                WHEN p.reference_metric = 'rouge' AND r.rouge_score IS NOT NULL THEN 'Standardized (ROUGE)'
                WHEN p.reference_metric = 'bert' AND r.bert_score IS NOT NULL THEN 'Standardized (BERT)'
                WHEN r.stop_reason IS NOT NULL AND r.stop_reason != 'Error_Incompatible'
                  AND (r.stop_reason LIKE '%Error%' OR r.stop_reason = 'Error_Load')
                THEN 'Standardized (' || UPPER(COALESCE(p.reference_metric, '')) || ') – benchmark error (0)'
                ELSE 'Standardized (' || UPPER(COALESCE(p.reference_metric, '')) || ')'
            END AS rationale
        FROM responses r
        JOIN prompts p ON r.prompt_id = p.prompt_id
        WHERE r.llm_id = ? AND r.prompt_id = ? AND r.system_id = ?
          AND COALESCE(p.benchmark_type, 'text') = 'standardized'
          AND p.reference_metric IN ('bleu', 'rouge', 'bert')
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (respondent_id, prompt_id, system_id))
        row = cursor.fetchone()
        if row is None:
            return None
        if row['rating'] is None:
            return None
        return row

    def get_standardized_responses_count(self, system_id: int, respondent_llm_ids: list[int]) -> int:
        """Count (respondent, prompt) pairs that are standardized and have a ref-metric score or benchmark error (0)."""
        if not self.connection or not respondent_llm_ids:
            return 0
        placeholders = ",".join("?" * len(respondent_llm_ids))
        query = f"""
        SELECT COUNT(*) FROM responses r
        JOIN prompts p ON r.prompt_id = p.prompt_id AND p.is_active = 1
        WHERE r.system_id = ? AND r.llm_id IN ({placeholders})
          AND COALESCE(p.benchmark_type, 'text') = 'standardized'
          AND p.reference_metric IN ('bleu', 'rouge', 'bert')
          AND (
              (p.reference_metric = 'bleu' AND r.bleu_score IS NOT NULL)
              OR (p.reference_metric = 'rouge' AND r.rouge_score IS NOT NULL)
              OR (p.reference_metric = 'bert' AND r.bert_score IS NOT NULL)
              OR (r.stop_reason IS NOT NULL AND r.stop_reason != 'Error_Incompatible'
                  AND (r.stop_reason LIKE '%Error%' OR r.stop_reason = 'Error_Load'))
          )
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id, *respondent_llm_ids))
        return cursor.fetchone()[0]

    def get_model_averaged_metrics(self, llm_id: int, system_id: int = 1) -> Optional[dict]:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        query = """
        SELECT MIN(r.timestamp) as first_run, AVG(r.time_to_first_token_sec) as avg_ttft, AVG(r.generation_time_sec) as avg_gen_time,
            AVG(r.total_time_sec) as avg_total_time, AVG(r.tokens_generated) as avg_tokens, AVG(r.total_tps) as avg_tps,
            AVG(r.avg_cpu_usage) as avg_cpu_load, AVG(r.avg_gpu_usage) as avg_gpu_load, AVG(r.avg_cpu_temp) as avg_cpu_temp,
            AVG(r.avg_gpu_temp) as avg_gpu_temp, MAX(r.max_cpu_temp) as peak_cpu_temp, MAX(r.max_gpu_temp) as peak_gpu_temp,
            AVG(CASE WHEN COALESCE(p.benchmark_type, 'text') = 'standardized' THEN r.bleu_score END) as avg_bleu,
            AVG(CASE WHEN COALESCE(p.benchmark_type, 'text') = 'standardized' THEN r.rouge_score END) as avg_rouge,
            AVG(CASE WHEN COALESCE(p.benchmark_type, 'text') = 'standardized' THEN r.bert_score END) as avg_bert,
            COUNT(r.response_id) as count,
            m.sys_ram_avail_pre_gb, m.sys_ram_avail_post_gb, m.vram_avail_pre_gb, m.vram_avail_post_gb, m.load_time_sec
        FROM responses r
        JOIN llms l ON r.llm_id = l.llm_id
        JOIN prompts p ON r.prompt_id = p.prompt_id
        LEFT JOIN llm_system_metrics m ON l.llm_id = m.llm_id AND m.system_id = ?
        WHERE r.llm_id = ? AND r.stop_reason NOT LIKE '%Error%' AND r.system_id = ?
        GROUP BY r.llm_id
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id, llm_id, system_id))
        row = cursor.fetchone()
        return dict(row) if row and row['count'] > 0 else None

    def get_respondent_scorecard(self, system_id: int = 1) -> list[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        query = """
        SELECT l.llm_identifier, l.display_name, AVG(e.rating) as avg_score, COUNT(e.rating) as num_evals,
            (AVG(e.rating * e.rating) - AVG(e.rating) * AVG(e.rating)) as variance
        FROM evaluations e
        JOIN responses r ON e.response_id = r.response_id
        JOIN llms l ON r.llm_id = l.llm_id
        JOIN llm_system_metrics m ON l.llm_id = m.llm_id AND m.system_id = r.system_id
        JOIN llms evaluator ON e.evaluator_llm_id = evaluator.llm_id
        JOIN llm_system_metrics m_eval ON evaluator.llm_id = m_eval.llm_id AND m_eval.system_id = r.system_id
        WHERE e.rating >= 0 AND m.is_respondent = 1 AND m_eval.is_evaluator = 1 AND r.system_id = ?
        GROUP BY l.llm_id ORDER BY avg_score DESC
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id,))
        return cursor.fetchall()
        
    def get_model_category_scores(self, system_id: int = 1) -> list[dict]:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        
        # Quality Scores
        query_evals = """
        SELECT l.llm_identifier, l.display_name, p.category as category, AVG(e.rating) as avg_score
        FROM evaluations e
        JOIN responses r ON e.response_id = r.response_id
        JOIN llms l ON r.llm_id = l.llm_id
        JOIN prompts p ON r.prompt_id = p.prompt_id AND p.is_active = 1
        JOIN llm_system_metrics m ON l.llm_id = m.llm_id AND m.system_id = r.system_id
        WHERE e.rating >= 0 AND r.system_id = ? AND m.use_results = 1
        GROUP BY l.llm_id, p.category
        """
        cursor.execute(query_evals, (system_id,))
        results = [dict(row) for row in cursor.fetchall()]
        
        # Speed Scores
        query_tps = """
        SELECT l.llm_identifier, l.display_name, AVG(r.total_tps) as avg_tps
        FROM responses r
        JOIN llms l ON r.llm_id = l.llm_id
        JOIN llm_system_metrics m ON l.llm_id = m.llm_id AND m.system_id = r.system_id
        WHERE r.total_tps > 0 AND r.system_id = ? AND m.use_results = 1
        GROUP BY l.llm_id
        """
        cursor.execute(query_tps, (system_id,))
        tps_rows = cursor.fetchall()
        for row in tps_rows:
            avg_tps = row['avg_tps']
            if avg_tps > 0:
                try:
                    raw_score = 10 * (math.log10(avg_tps) / math.log10(50))
                    speed_score = max(0, min(10, raw_score))
                except ValueError: speed_score = 0
                results.append({'llm_identifier': row['llm_identifier'], 'display_name': row['display_name'], 'category': 'Speed (TPS)', 'avg_score': speed_score})
        return results

    def get_evaluator_statistics(self, system_id: int = 1) -> list[sqlite3.Row]:
        """Per-evaluator counts including only evaluations of responses from respondents that are enabled (Run checked in Model Manager)."""
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        query = """
        SELECT l.llm_identifier, l.display_name, l.llm_id, COUNT(e.evaluation_id) as total_attempts,
            SUM(CASE WHEN e.rating = -1 THEN 1 ELSE 0 END) as failed_attempts
        FROM evaluations e
        JOIN llms l ON e.evaluator_llm_id = l.llm_id
        JOIN responses r ON e.response_id = r.response_id
        JOIN llm_system_metrics m_resp ON r.llm_id = m_resp.llm_id AND m_resp.system_id = r.system_id AND m_resp.is_respondent = 1
        WHERE r.system_id = ?
        GROUP BY l.llm_id ORDER BY total_attempts DESC
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id,))
        return cursor.fetchall()

    def get_respondent_evaluation_stats(self, system_id: int = 1) -> list[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("DB not open")
        query = "SELECT r.llm_id, COUNT(e.evaluation_id) as total_evals, SUM(CASE WHEN e.rating = -1 THEN 1 ELSE 0 END) as error_count FROM responses r JOIN evaluations e ON r.response_id = e.response_id WHERE r.system_id = ? GROUP BY r.llm_id"
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id,))
        return cursor.fetchall()

    def get_missing_responses(self, system_id: int = 1, backend_type: Optional[str] = None) -> list[tuple[int, int, str, str, str, str]]:
        """Missing (llm, prompt) jobs for selected respondents (Run checkbox). Only includes jobs where the model
        has the required capability for the prompt's benchmark_type. Returns (llm_id, prompt_id, llm_identifier,
        prompt_text, prompt_category, backend_type). When backend_type is set, only returns models from that backend
        (legacy filter); normally omit to run all selected models regardless of backend."""
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        query = """
        SELECT l.llm_id, p.prompt_id, l.llm_identifier, p.prompt_text, p.name AS prompt_category,
               COALESCE(l.backend_type, 'lmstudio') AS backend_type,
               l.architecture, l.params_string, l.display_name
        FROM llms AS l
        CROSS JOIN prompts AS p
        JOIN llm_system_metrics m ON l.llm_id = m.llm_id AND m.system_id = ?
        LEFT JOIN responses AS r ON r.llm_id = l.llm_id AND r.prompt_id = p.prompt_id AND r.system_id = ?
        WHERE r.response_id IS NULL AND p.is_active = 1 AND m.is_available = 1 AND m.is_respondent = 1
        AND (
            COALESCE(p.benchmark_type, 'text') IN ('text', 'standardized')
            OR (p.benchmark_type = 'image_comprehension' AND l.vision = 1)
            OR (p.benchmark_type IN ('tool_use', 'mcp') AND l.trained_for_tool_use = 1)
        )
        """
        params: list = [system_id, system_id]
        if backend_type is not None and backend_type != "":
            query += " AND COALESCE(l.backend_type, 'lmstudio') = ?"
            params.append(backend_type)
        cursor = self.connection.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        def sort_key(row): return ((row['architecture'] or "").lower(), self._parse_params(row['params_string']), (row['display_name'] or row['llm_identifier']).lower(), (row['prompt_category'] or "").lower())
        rows.sort(key=sort_key)
        return [(row['llm_id'], row['prompt_id'], row['llm_identifier'], row['prompt_text'], row['prompt_category'], row['backend_type']) for row in rows]

    def get_incompatible_response_pairs(self, system_id: int = 1) -> list[tuple[int, int]]:
        """Returns (llm_id, prompt_id) pairs that are respondent x active prompt but not capability-matched
        and have no response yet. Capability uses llms.vision (Vis) and llms.trained_for_tool_use (Tools)
        from LM Studio metadata. Caller records these with 'Incompatible with prompt' → purple/TODO cards."""
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        query = """
        SELECT l.llm_id, p.prompt_id
        FROM llms AS l
        CROSS JOIN prompts AS p
        JOIN llm_system_metrics m ON l.llm_id = m.llm_id AND m.system_id = ?
        LEFT JOIN responses AS r ON r.llm_id = l.llm_id AND r.prompt_id = p.prompt_id AND r.system_id = ?
        WHERE r.response_id IS NULL AND p.is_active = 1 AND m.is_available = 1 AND m.is_respondent = 1
        AND (
            (COALESCE(p.benchmark_type, 'text') = 'image_comprehension' AND (l.vision = 0 OR l.vision IS NULL))
            OR (COALESCE(p.benchmark_type, 'text') IN ('tool_use', 'mcp') AND (l.trained_for_tool_use = 0 OR l.trained_for_tool_use IS NULL))
        )
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id, system_id))
        return [(row['llm_id'], row['prompt_id']) for row in cursor.fetchall()]

    def get_missing_evaluations(self, system_id: int = 1, include_non_evaluators: bool = False) -> list[tuple[Any, ...]]:
        """Missing evaluation jobs. Only includes responses from respondents that are enabled (Run checked in Model Manager).
        Excludes incompatible responses (Error_Incompatible) — those get auto-rated 0 via ensure_incompatible_evaluations_recorded.
        Default: only Judge-selected models (m_eval.is_evaluator = 1) as evaluators. If include_non_evaluators=True,
        also appends jobs where evaluators are not on the Judge panel (legacy / not used by the evaluation engine)."""
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        base_select = """
        SELECT r.response_id, evaluator.llm_id, respondent.llm_identifier, evaluator.llm_identifier, 
               r.response_text, p.name as category, p.rubric_text, p.prompt_text,
               evaluator.architecture, evaluator.params_string, evaluator.display_name,
               r.llm_id as respondent_llm_id, r.prompt_id,
               COALESCE(p.category, 'Ungrouped') as prompt_category_group
        FROM responses AS r
        JOIN prompts AS p ON r.prompt_id = p.prompt_id AND p.is_active = 1
        JOIN llms AS respondent ON r.llm_id = respondent.llm_id
        JOIN llm_system_metrics m_resp ON respondent.llm_id = m_resp.llm_id AND m_resp.system_id = ? AND m_resp.is_respondent = 1
        CROSS JOIN llms AS evaluator
        JOIN llm_system_metrics m_eval ON evaluator.llm_id = m_eval.llm_id AND m_eval.system_id = ?
        LEFT JOIN evaluations AS e ON e.response_id = r.response_id AND e.evaluator_llm_id = evaluator.llm_id
        WHERE r.system_id = ? AND e.evaluation_id IS NULL AND respondent.llm_id != evaluator.llm_id AND m_eval.is_available = 1
        AND (r.stop_reason IS NULL OR r.stop_reason != 'Error_Incompatible')
        AND COALESCE(p.benchmark_type, 'text') != 'standardized'
        """
        def sort_key(row): return ((row['architecture'] or "").lower(), self._parse_params(row['params_string']), (row['display_name'] or row['llm_identifier']).lower(), row['llm_identifier'].lower())

        cursor = self.connection.cursor()
        params = (system_id, system_id, system_id)
        # Evaluator-panel jobs first
        query_panel = base_select + " AND m_eval.is_evaluator = 1"
        cursor.execute(query_panel, params)
        rows_panel = cursor.fetchall()
        rows_panel.sort(key=sort_key)
        result = [tuple(row[:8]) + (row['respondent_llm_id'], row['prompt_id'], row['prompt_category_group']) for row in rows_panel]

        if include_non_evaluators:
            query_non = base_select + " AND (m_eval.is_evaluator = 0 OR m_eval.is_evaluator IS NULL)"
            cursor.execute(query_non, params)
            rows_non = cursor.fetchall()
            rows_non.sort(key=sort_key)
            result.extend([tuple(row[:8]) + (row['respondent_llm_id'], row['prompt_id'], row['prompt_category_group']) for row in rows_non])

        return result

    def get_category_evaluation_existing_count(
        self, system_id: int, category: str, include_non_evaluators: bool = False
    ) -> int:
        """Count of evaluations already in DB for this prompt category.
        Uses same filters as get_missing_evaluations (respondent enabled, evaluator available, etc.)
        so existing + missing = true total for the category."""
        if not self.connection:
            raise sqlite3.Error("Database connection is not open.")
        base_conditions = """
        FROM evaluations e
        JOIN responses r ON e.response_id = r.response_id
        JOIN prompts p ON r.prompt_id = p.prompt_id AND p.is_active = 1
        JOIN llms respondent ON r.llm_id = respondent.llm_id
        JOIN llm_system_metrics m_resp ON respondent.llm_id = m_resp.llm_id AND m_resp.system_id = ? AND m_resp.is_respondent = 1
        JOIN llm_system_metrics m_eval ON e.evaluator_llm_id = m_eval.llm_id AND m_eval.system_id = ?
        WHERE r.system_id = ?
          AND COALESCE(p.category, 'Ungrouped') = ?
          AND respondent.llm_id != e.evaluator_llm_id
          AND m_eval.is_available = 1
          AND (r.stop_reason IS NULL OR r.stop_reason != 'Error_Incompatible')
          AND COALESCE(p.benchmark_type, 'text') != 'standardized'
        """
        cursor = self.connection.cursor()
        params = (system_id, system_id, system_id, category)
        if include_non_evaluators:
            query = f"SELECT COUNT(*) {base_conditions}"
            cursor.execute(query, params)
        else:
            query = f"SELECT COUNT(*) {base_conditions} AND m_eval.is_evaluator = 1"
            cursor.execute(query, params)
        return cursor.fetchone()[0]

    def get_full_benchmark_results(self, system_id: int = 1) -> list[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        query = """
        SELECT l.llm_identifier AS Model, l.display_name AS DisplayName,
            COALESCE(l.backend_type, 'lmstudio') AS BackendType,
            l.architecture AS Architecture, l.size_bytes AS SizeBytes,
            l.params_string AS Params, p.category AS Category, p.name AS PromptName, r.response_text AS Response, r.tokens_generated AS Tokens,
            r.total_time_sec AS TotalTime, r.time_to_first_token_sec AS TTFT, r.generation_time_sec AS GenTime, r.total_tps AS TotalTPS,
            r.streaming_tps AS GenTPS, r.context_length AS Context, r.max_cpu_temp AS PeakCPU, r.max_gpu_temp AS PeakGPU, 
            r.peak_vram_usage_gb, r.peak_ram_usage_gb, r.thermal_throttling_count, r.power_throttling_count,
            r.bleu_score AS BLEU, r.rouge_score AS ROUGE, r.bert_score AS BERT, r.stop_reason AS StopReason, 
            CASE
                WHEN COALESCE(p.benchmark_type, 'text') = 'standardized' AND p.reference_metric = 'bleu' AND r.bleu_score IS NOT NULL THEN ROUND(r.bleu_score * 10, 2)
                WHEN COALESCE(p.benchmark_type, 'text') = 'standardized' AND p.reference_metric = 'rouge' AND r.rouge_score IS NOT NULL THEN ROUND(r.rouge_score * 10, 2)
                WHEN COALESCE(p.benchmark_type, 'text') = 'standardized' AND p.reference_metric = 'bert' AND r.bert_score IS NOT NULL THEN ROUND(r.bert_score * 10, 2)
                WHEN COALESCE(p.benchmark_type, 'text') = 'standardized' AND p.reference_metric IN ('bleu', 'rouge', 'bert')
                  AND (
                    (p.reference_metric = 'bleu' AND r.bleu_score IS NULL) OR
                    (p.reference_metric = 'rouge' AND r.rouge_score IS NULL) OR
                    (p.reference_metric = 'bert' AND r.bert_score IS NULL)
                  )
                  AND r.stop_reason IS NOT NULL AND r.stop_reason != 'Error_Incompatible'
                  AND (r.stop_reason LIKE '%Error%' OR r.stop_reason = 'Error_Load')
                THEN 0
                ELSE e.rating
            END AS Score,
            CASE
                WHEN e.evaluation_id IS NOT NULL THEN evaluator.llm_identifier
                WHEN COALESCE(p.benchmark_type, 'text') = 'standardized' THEN COALESCE(p.reference_metric, '')
                ELSE NULL
            END AS Judge,
            e.rationale AS Rationale
        FROM responses r
        JOIN llms l ON r.llm_id = l.llm_id
        JOIN prompts p ON r.prompt_id = p.prompt_id
        JOIN llm_system_metrics m ON l.llm_id = m.llm_id AND m.system_id = r.system_id
        LEFT JOIN evaluations e ON r.response_id = e.response_id
        LEFT JOIN llms evaluator ON e.evaluator_llm_id = evaluator.llm_id
        LEFT JOIN llm_system_metrics m_eval ON e.evaluator_llm_id = m_eval.llm_id AND m_eval.system_id = r.system_id
        WHERE r.system_id = ? AND m.use_results = 1 AND p.is_active = 1
          AND (e.evaluation_id IS NULL OR m_eval.is_evaluator = 1)
        ORDER BY l.llm_identifier, p.category
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id,))
        return cursor.fetchall()

    def get_evaluations_matrix(self, system_id: int = 1) -> list[tuple]:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        query = """
        SELECT e.response_id, e.evaluator_llm_id, e.rating 
        FROM evaluations e 
        JOIN responses r ON e.response_id = r.response_id
        JOIN prompts p ON r.prompt_id = p.prompt_id AND p.is_active = 1
        JOIN llm_system_metrics m ON r.llm_id = m.llm_id AND m.system_id = r.system_id
        JOIN llm_system_metrics m_eval ON e.evaluator_llm_id = m_eval.llm_id AND m_eval.system_id = r.system_id AND m_eval.is_evaluator = 1
        WHERE r.system_id = ? AND e.rating >= 0 AND m.use_results = 1
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id,))
        return cursor.fetchall()

    def get_judge_bias_stats(self, system_id: int = 1) -> list[dict]:
        if not self.connection: raise sqlite3.Error("DB not open")
        query = """
        SELECT l.display_name, l.llm_identifier, AVG(e.rating) as avg_score, COUNT(e.rating) as count
        FROM evaluations e
        JOIN llms l ON e.evaluator_llm_id = l.llm_id
        JOIN responses r ON e.response_id = r.response_id
        JOIN prompts p ON r.prompt_id = p.prompt_id AND p.is_active = 1
        JOIN llm_system_metrics m ON r.llm_id = m.llm_id AND m.system_id = r.system_id
        JOIN llm_system_metrics m_eval ON e.evaluator_llm_id = m_eval.llm_id AND m_eval.system_id = r.system_id AND m_eval.is_evaluator = 1
        WHERE r.system_id = ? AND e.rating >= 0 AND m.use_results = 1
        GROUP BY l.llm_id ORDER BY avg_score DESC
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id,))
        return [{'name': r['display_name'] or r['llm_identifier'].split('/')[-1], 'avg': r['avg_score'], 'count': r['count']} for r in cursor.fetchall()]

    def get_all_evaluations_detailed(self, system_id: int = 1) -> list[dict]:
        if not self.connection: raise sqlite3.Error("DB not open")
        query = """
        SELECT r.response_id, p.name as category, resp_llm.display_name as respondent, resp_llm.llm_identifier as respondent_id,
            eval_llm.display_name as evaluator, e.rating, e.rationale
        FROM evaluations e
        JOIN responses r ON e.response_id = r.response_id
        JOIN prompts p ON r.prompt_id = p.prompt_id AND p.is_active = 1
        JOIN llms resp_llm ON r.llm_id = resp_llm.llm_id
        JOIN llms eval_llm ON e.evaluator_llm_id = eval_llm.llm_id
        JOIN llm_system_metrics m ON r.llm_id = m.llm_id AND m.system_id = r.system_id
        JOIN llm_system_metrics m_eval ON e.evaluator_llm_id = m_eval.llm_id AND m_eval.system_id = r.system_id AND m_eval.is_evaluator = 1
        WHERE e.rating >= 0 AND r.system_id = ? AND m.use_results = 1
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id,))
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_system_global_stats(self, system_ids: list[int]) -> dict:
        if not self.connection: raise sqlite3.Error("DB not open")
        if not system_ids: return {}
        placeholders = ",".join("?" * len(system_ids))
        query = f"SELECT system_id, AVG(avg_cpu_usage) as cpu_util, AVG(avg_gpu_usage) as gpu_util, AVG(avg_cpu_temp) as cpu_temp, AVG(avg_gpu_temp) as gpu_temp FROM responses WHERE system_id IN ({placeholders}) GROUP BY system_id"
        cursor = self.connection.cursor()
        cursor.execute(query, tuple(system_ids))
        return {row['system_id']: dict(row) for row in cursor.fetchall()}

    def get_system_comparison_stats(self, system_ids: list[int]) -> list[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("DB not open")
        if not system_ids: return []
        placeholders = ",".join("?" * len(system_ids))
        query = f"""
        SELECT r.system_id, l.llm_identifier, l.display_name, l.size_bytes, r.context_length,
            AVG(r.total_tps) as avg_tps, AVG(r.streaming_tps) as avg_gen_tps,
            AVG(r.time_to_first_token_sec) as avg_ttft, AVG(r.generation_time_sec) as avg_gen_time, AVG(r.peak_vram_usage_gb) as avg_peak_vram,
            AVG(r.peak_ram_usage_gb) as avg_peak_ram, COUNT(r.response_id) as run_count
        FROM responses r
        JOIN llms l ON r.llm_id = l.llm_id
        JOIN llm_system_metrics m ON l.llm_id = m.llm_id AND m.system_id = r.system_id
        WHERE r.system_id IN ({placeholders}) AND r.stop_reason NOT LIKE '%Error%' AND m.use_results = 1
        GROUP BY r.system_id, l.llm_identifier, r.context_length ORDER BY l.llm_identifier, r.context_length, r.system_id
        """
        cursor = self.connection.cursor()
        cursor.execute(query, tuple(system_ids))
        return cursor.fetchall()

    def has_any_evaluations(self) -> bool:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        cursor.execute("SELECT 1 FROM evaluations LIMIT 1")
        return cursor.fetchone() is not None

    def get_error_log(self, limit: int = 100) -> list[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("DB not open")
        query = "SELECT e.*, l.llm_identifier, l.display_name FROM error_logs e LEFT JOIN llms l ON e.llm_id = l.llm_id ORDER BY e.timestamp DESC LIMIT ?"
        cursor = self.connection.cursor()
        cursor.execute(query, (limit,))
        return cursor.fetchall()

    def get_failed_evaluations_log(self, limit: int = 50) -> list[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("DB not open")
        query = "SELECT e.timestamp, e.rationale as error_message, l.llm_identifier, l.display_name FROM evaluations e JOIN llms l ON e.evaluator_llm_id = l.llm_id WHERE e.rating = -1 ORDER BY e.timestamp DESC LIMIT ?"
        cursor = self.connection.cursor()
        cursor.execute(query, (limit,))
        return cursor.fetchall()

    def get_all_evaluator_stats_unfiltered(self, system_id: int = 1) -> list[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        query = "SELECT l.llm_identifier, l.display_name, l.llm_id, COUNT(e.evaluation_id) as total_attempts, SUM(CASE WHEN e.rating = -1 THEN 1 ELSE 0 END) as failed_attempts FROM evaluations e JOIN llms l ON e.evaluator_llm_id = l.llm_id JOIN responses r ON e.response_id = r.response_id WHERE r.system_id = ? GROUP BY l.llm_id ORDER BY total_attempts DESC"
        cursor = self.connection.cursor()
        cursor.execute(query, (system_id,))
        return cursor.fetchall()

    def has_responses_for_prompt(self, prompt_id: int) -> bool:
        if not self.connection: raise sqlite3.Error("DB not open")
        cursor = self.connection.cursor()
        cursor.execute("SELECT 1 FROM responses WHERE prompt_id = ? LIMIT 1", (prompt_id,))
        return cursor.fetchone() is not None