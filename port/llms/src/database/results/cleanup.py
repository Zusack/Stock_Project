import sqlite3

# Must match database.results.INCOMPATIBLE_STOP_REASONS — avoid circular import.
# Only Error_Incompatible is preserved; Error_Vision/Error_Tool are run failures, cleared as real errors.
_INCOMPATIBLE_STOP_REASONS = frozenset({"Error_Incompatible"})


class ResultsCleanup:
    def clear_all_evaluations(self, system_id: int):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("DELETE FROM evaluations WHERE response_id IN (SELECT response_id FROM responses WHERE system_id = ?)", (system_id,))
        self.connection.commit()

    def clear_all_responses(self, system_id: int):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("DELETE FROM responses WHERE system_id = ?", (system_id,))
        self.connection.commit()

    def clear_all_memory_metrics(self, system_id: int):
        """
        Resets memory stats to NULL without deleting the row.
        """
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("""
            UPDATE llm_system_metrics 
            SET sys_ram_total_gb=NULL, vram_total_gb=NULL, sys_ram_avail_pre_gb=NULL, vram_avail_pre_gb=NULL,
                sys_ram_avail_post_gb=NULL, vram_avail_post_gb=NULL, load_time_sec=NULL, last_updated=NULL
            WHERE system_id = ?
        """, (system_id,))
        self.connection.commit()

    def clear_model_memory_metrics(self, llm_id: int, system_id: int):
        """
        Resets memory stats to NULL without deleting the row.
        """
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("""
            UPDATE llm_system_metrics 
            SET sys_ram_total_gb=NULL, vram_total_gb=NULL, sys_ram_avail_pre_gb=NULL, vram_avail_pre_gb=NULL,
                sys_ram_avail_post_gb=NULL, vram_avail_post_gb=NULL, load_time_sec=NULL, last_updated=NULL
            WHERE llm_id = ? AND system_id = ?
        """, (llm_id, system_id))
        self.connection.commit()

    def delete_model_evaluations(self, llm_id: int, system_id: int):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("DELETE FROM evaluations WHERE response_id IN (SELECT response_id FROM responses WHERE llm_id = ? AND system_id = ?)", (llm_id, system_id))
        self.connection.execute("DELETE FROM evaluations WHERE evaluator_llm_id = ? AND response_id IN (SELECT response_id FROM responses WHERE system_id = ?)", (llm_id, system_id))
        self.connection.commit()

    def delete_model_responses(self, llm_id: int, system_id: int):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("DELETE FROM responses WHERE llm_id = ? AND system_id = ?", (llm_id, system_id))
        self.connection.commit()

    def clear_error_responses(self, system_id: int = 1) -> int:
        """Delete only real error responses (load/crash/generic). Keeps incompatible/skipped placeholders."""
        if not self.connection:
            try: self.connect()
            except: raise sqlite3.Error("Could not connect to database.")
        try:
            cursor = self.connection.cursor()
            placeholders = ",".join("?" * len(_INCOMPATIBLE_STOP_REASONS))
            cursor.execute(
                "DELETE FROM responses WHERE system_id = ? AND stop_reason LIKE '%Error%' AND stop_reason NOT IN (" + placeholders + ")",
                (system_id,) + tuple(_INCOMPATIBLE_STOP_REASONS)
            )
            deleted_count = cursor.rowcount
            self.connection.commit()
            return deleted_count
        except sqlite3.Error: self.connection.rollback(); return 0

    def clear_error_responses_for_model(self, llm_id: int, system_id: int) -> int:
        """Delete only real error responses for a single model. Keeps incompatible/skipped placeholders."""
        if not self.connection:
            try: self.connect()
            except: raise sqlite3.Error("Could not connect to database.")
        try:
            cursor = self.connection.cursor()
            placeholders = ",".join("?" * len(_INCOMPATIBLE_STOP_REASONS))
            cursor.execute(
                "DELETE FROM responses WHERE llm_id = ? AND system_id = ? AND stop_reason LIKE '%Error%' AND stop_reason NOT IN (" + placeholders + ")",
                (llm_id, system_id) + tuple(_INCOMPATIBLE_STOP_REASONS)
            )
            deleted_count = cursor.rowcount
            self.connection.commit()
            return deleted_count
        except sqlite3.Error: self.connection.rollback(); return 0

    def clear_failed_evaluations(self, system_id: int = 1) -> int:
        if not self.connection:
            try: self.connect()
            except: raise sqlite3.Error("Could not connect to database.")
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM evaluations WHERE rating = -1 AND response_id IN (SELECT response_id FROM responses WHERE system_id = ?)", (system_id,))
            deleted_count = cursor.rowcount
            self.connection.commit()
            return deleted_count
        except sqlite3.Error: self.connection.rollback(); return 0

    def clear_failed_evaluations_for_model(self, evaluator_llm_id: int, system_id: int) -> int:
        """Delete only failed (rating = -1) evaluations for a single evaluator model."""
        if not self.connection:
            try: self.connect()
            except: raise sqlite3.Error("Could not connect to database.")
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "DELETE FROM evaluations WHERE evaluator_llm_id = ? AND rating = -1 AND response_id IN (SELECT response_id FROM responses WHERE system_id = ?)",
                (evaluator_llm_id, system_id)
            )
            deleted_count = cursor.rowcount
            self.connection.commit()
            return deleted_count
        except sqlite3.Error: self.connection.rollback(); return 0

    def delete_evaluator_history(self, evaluator_llm_id: int, system_id: int = 1) -> int:
        if not self.connection:
            try: self.connect()
            except: raise sqlite3.Error("Could not connect to database.")
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM evaluations WHERE evaluator_llm_id = ? AND response_id IN (SELECT response_id FROM responses WHERE system_id = ?)", (evaluator_llm_id, system_id))
            deleted_count = cursor.rowcount
            self.connection.commit()
            return deleted_count
        except sqlite3.Error: self.connection.rollback(); return 0

    def clear_error_log(self):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("DELETE FROM error_logs")
        self.connection.commit()

    def delete_responses_for_prompt(self, prompt_id: int):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("DELETE FROM responses WHERE prompt_id = ?", (prompt_id,))
        self.connection.commit()

    def delete_single_response(self, llm_id: int, prompt_id: int, system_id: int) -> None:
        """Delete the single response for the given LLM, prompt, and system. Cascades to evaluations."""
        if not self.connection:
            raise sqlite3.Error("DB not open")
        self.connection.execute(
            "DELETE FROM responses WHERE llm_id = ? AND prompt_id = ? AND system_id = ?",
            (llm_id, prompt_id, system_id),
        )
        self.connection.commit()

    def delete_single_evaluation(self, evaluator_llm_id: int, respondent_llm_id: int, prompt_id: int, system_id: int) -> None:
        """Delete the single evaluation for the given evaluator, respondent, prompt, and system."""
        if not self.connection:
            raise sqlite3.Error("DB not open")
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT response_id FROM responses WHERE llm_id = ? AND prompt_id = ? AND system_id = ?",
            (respondent_llm_id, prompt_id, system_id),
        )
        row = cursor.fetchone()
        if row is None:
            return  # No response, nothing to delete
        response_id = row[0]
        self.connection.execute(
            "DELETE FROM evaluations WHERE response_id = ? AND evaluator_llm_id = ?",
            (response_id, evaluator_llm_id),
        )
        self.connection.commit()