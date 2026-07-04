import sqlite3

class LLMRepository:
    # --- SYSTEM SPECIFIC UPDATES ---
    
    def ensure_system_metric_record(self, llm_id: int, system_id: int):
        """Ensures a record exists in llm_system_metrics for this pair."""
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("""
            INSERT OR IGNORE INTO llm_system_metrics (llm_id, system_id) VALUES (?, ?)
        """, (llm_id, system_id))

    def set_all_llms_availability(self, system_id: int, is_available: bool):
        """Sets availability for ALL models on THIS system."""
        if not self.connection: raise sqlite3.Error("DB not open")
        val = 1 if is_available else 0
        self.connection.execute("UPDATE llm_system_metrics SET is_available = ? WHERE system_id = ?", (val, system_id))
        self.connection.commit()

    def mark_llm_available(self, llm_id: int, system_id: int):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.ensure_system_metric_record(llm_id, system_id)
        self.connection.execute("""
            UPDATE llm_system_metrics 
            SET is_available = 1 
            WHERE llm_id = ? AND system_id = ?
        """, (llm_id, system_id))
        self.connection.commit()

    def update_respondent_status(self, llm_id: int, system_id: int, is_respondent: bool):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.ensure_system_metric_record(llm_id, system_id)
        val = 1 if is_respondent else 0
        self.connection.execute("""
            UPDATE llm_system_metrics SET is_respondent = ? WHERE llm_id = ? AND system_id = ?
        """, (val, llm_id, system_id))
        self.connection.commit()

    def update_evaluator_status(self, llm_id: int, system_id: int, is_evaluator: bool):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.ensure_system_metric_record(llm_id, system_id)
        val = 1 if is_evaluator else 0
        self.connection.execute("""
            UPDATE llm_system_metrics SET is_evaluator = ? WHERE llm_id = ? AND system_id = ?
        """, (val, llm_id, system_id))
        self.connection.commit()
        
    def update_use_results_status(self, llm_id: int, system_id: int, use_results: bool):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.ensure_system_metric_record(llm_id, system_id)
        val = 1 if use_results else 0
        self.connection.execute("""
            UPDATE llm_system_metrics SET use_results = ? WHERE llm_id = ? AND system_id = ?
        """, (val, llm_id, system_id))
        self.connection.commit()

    def set_all_respondents(self, system_id: int, is_respondent: bool):
        if not self.connection: raise sqlite3.Error("DB not open")
        val = 1 if is_respondent else 0
        self.connection.execute("""
            UPDATE llm_system_metrics 
            SET is_respondent = ? 
            WHERE system_id = ? AND is_available = 1 AND COALESCE(is_disabled, 0) = 0
        """, (val, system_id))
        self.connection.commit()
        
    def set_all_use_results(self, system_id: int, use_results: bool):
        if not self.connection: raise sqlite3.Error("DB not open")
        val = 1 if use_results else 0
        self.connection.execute("""
            UPDATE llm_system_metrics 
            SET use_results = ? 
            WHERE system_id = ? AND is_available = 1 AND COALESCE(is_disabled, 0) = 0
        """, (val, system_id))
        self.connection.commit()

    def update_disabled_status(self, llm_id: int, system_id: int, is_disabled: bool):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.ensure_system_metric_record(llm_id, system_id)
        val = 1 if is_disabled else 0
        if is_disabled:
            self.connection.execute("""
                UPDATE llm_system_metrics 
                SET is_disabled = ?, is_respondent = 0, is_evaluator = 0, use_results = 0
                WHERE llm_id = ? AND system_id = ?
            """, (val, llm_id, system_id))
        else:
            self.connection.execute("""
                UPDATE llm_system_metrics SET is_disabled = ? WHERE llm_id = ? AND system_id = ?
            """, (val, llm_id, system_id))
        self.connection.commit()

    def set_disabled_by_backend(self, system_id: int, backend_type: str, is_disabled: bool):
        """Disable or enable all models from a specific backend."""
        if not self.connection: raise sqlite3.Error("DB not open")
        val = 1 if is_disabled else 0
        if is_disabled:
            self.connection.execute("""
                UPDATE llm_system_metrics 
                SET is_disabled = ?, is_respondent = 0, is_evaluator = 0, use_results = 0
                WHERE system_id = ? AND llm_id IN (
                    SELECT llm_id FROM llms WHERE backend_type = ?
                )
            """, (val, system_id, backend_type))
        else:
            self.connection.execute("""
                UPDATE llm_system_metrics SET is_disabled = ?
                WHERE system_id = ? AND llm_id IN (
                    SELECT llm_id FROM llms WHERE backend_type = ?
                )
            """, (val, system_id, backend_type))
        self.connection.commit()

    def set_respondents_by_backend(self, system_id: int, backend_type: str, is_respondent: bool):
        """Set Run (benchmark roster) for all models from a specific backend."""
        if not self.connection: raise sqlite3.Error("DB not open")
        val = 1 if is_respondent else 0
        self.connection.execute("""
            UPDATE llm_system_metrics
            SET is_respondent = ?
            WHERE system_id = ? AND COALESCE(is_disabled, 0) = 0 AND is_available = 1
            AND llm_id IN (SELECT llm_id FROM llms WHERE backend_type = ?)
        """, (val, system_id, backend_type))
        self.connection.commit()

    def set_evaluators_by_backend(self, system_id: int, backend_type: str, is_evaluator: bool):
        """Set Judge (evaluator panel) for all models from a specific backend."""
        if not self.connection: raise sqlite3.Error("DB not open")
        val = 1 if is_evaluator else 0
        self.connection.execute("""
            UPDATE llm_system_metrics
            SET is_evaluator = ?
            WHERE system_id = ? AND COALESCE(is_disabled, 0) = 0 AND is_available = 1
            AND llm_id IN (SELECT llm_id FROM llms WHERE backend_type = ?)
        """, (val, system_id, backend_type))
        self.connection.commit()

    def set_use_results_by_backend(self, system_id: int, backend_type: str, use_results: bool):
        """Set Use Res. for all models from a specific backend."""
        if not self.connection: raise sqlite3.Error("DB not open")
        val = 1 if use_results else 0
        self.connection.execute("""
            UPDATE llm_system_metrics
            SET use_results = ?
            WHERE system_id = ? AND COALESCE(is_disabled, 0) = 0 AND is_available = 1
            AND llm_id IN (SELECT llm_id FROM llms WHERE backend_type = ?)
        """, (val, system_id, backend_type))
        self.connection.commit()

    def reset_all_evaluators(self, system_id: int):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("UPDATE llm_system_metrics SET is_evaluator = 0 WHERE system_id = ?", (system_id,))
        self.connection.commit()

    def sync_gguf_available_on_disk(self) -> None:
        """Refresh available_on_disk for GGUF rows from file existence."""
        if not self.connection:
            raise sqlite3.Error("DB not open")
        import os
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT llm_id, model_path FROM llms WHERE LOWER(COALESCE(backend_type, '')) = 'gguf'"
        )
        for row in cursor.fetchall():
            path = (row["model_path"] or "").strip()
            ok = bool(path and os.path.isfile(path))
            cursor.execute(
                "UPDATE llms SET available_on_disk = ? WHERE llm_id = ?",
                (1 if ok else 0, row["llm_id"]),
            )
        self.connection.commit()

    # --- GLOBAL / GENERIC ---

    def delete_llm(self, llm_id: int):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("DELETE FROM llms WHERE llm_id = ?", (llm_id,))
        self.connection.commit()

    def get_backend_type(self, llm_id: int) -> str:
        """Return backend_type for the given llm_id (e.g. 'gguf', 'ollama'). Defaults to 'lmstudio' if missing."""
        if not self.connection:
            raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        cursor.execute("SELECT COALESCE(backend_type, 'lmstudio') FROM llms WHERE llm_id = ?", (llm_id,))
        row = cursor.fetchone()
        return (row[0] or "lmstudio") if row else "lmstudio"

    def get_or_create_llm(self, llm_identifier: str, backend_type: str = "lmstudio") -> int:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        # Try to match on (identifier, backend_type) first for multi-backend support
        cursor.execute(
            "SELECT llm_id FROM llms WHERE llm_identifier = ? AND backend_type = ?",
            (llm_identifier, backend_type),
        )
        row = cursor.fetchone()
        if row: return row['llm_id']
        # Fallback: check without backend_type (legacy rows)
        cursor.execute("SELECT llm_id FROM llms WHERE llm_identifier = ? AND (backend_type IS NULL OR backend_type = '')", (llm_identifier,))
        row = cursor.fetchone()
        if row:
            # Update legacy row with backend_type
            cursor.execute("UPDATE llms SET backend_type = ? WHERE llm_id = ?", (backend_type, row['llm_id']))
            self.connection.commit()
            return row['llm_id']
        # Insert new
        try:
            cursor.execute("INSERT INTO llms (llm_identifier, backend_type) VALUES (?, ?)", (llm_identifier, backend_type))
            self.connection.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor.execute("SELECT llm_id FROM llms WHERE llm_identifier = ? AND backend_type = ?", (llm_identifier, backend_type))
            row = cursor.fetchone()
            if row: return row['llm_id']
            raise

    def update_llm_metadata(self, llm_id: int, metadata: dict) -> None:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        audit_keys = {'sys_ram_total_gb', 'vram_total_gb', 'sys_ram_avail_pre_gb', 'vram_avail_pre_gb', 'sys_ram_avail_post_gb', 'vram_avail_post_gb', 'load_time_sec'}
        clean_meta = {k: v for k, v in metadata.items() if k not in audit_keys}
        if not clean_meta: return

        set_clause = ", ".join([f"{key} = ?" for key in clean_meta.keys()])
        values = list(clean_meta.values())
        values.append(llm_id)
        try:
            self.connection.execute(f"UPDATE llms SET {set_clause}, last_updated = CURRENT_TIMESTAMP WHERE llm_id = ?", tuple(values))
            self.connection.commit()
        except sqlite3.Error: self.connection.rollback()

    def update_load_time_sec(self, llm_id: int, system_id: int, load_time_sec: float) -> None:
        """Update only load_time_sec for this llm/system (e.g. after benchmark load). Does not overwrite other metrics.
        When merging DBs from older versions, load_time_sec may be missing in the import; merge uses .get() so NULL is accepted.
        TODO: Once all systems' DBs are synchronized, the merge comment in system_summary_view can be removed."""
        if not self.connection: raise sqlite3.Error("DB not open")
        self.ensure_system_metric_record(llm_id, system_id)
        try:
            self.connection.execute("""
                UPDATE llm_system_metrics SET load_time_sec = ?, last_updated = CURRENT_TIMESTAMP
                WHERE llm_id = ? AND system_id = ?
            """, (load_time_sec, llm_id, system_id))
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error updating load time: {e}")
            self.connection.rollback()

    def save_audit_metrics(self, llm_id: int, system_id: int, metrics: dict) -> None:
        if not self.connection: raise sqlite3.Error("DB not open")
        
        query = """
        INSERT INTO llm_system_metrics 
        (llm_id, system_id, sys_ram_total_gb, vram_total_gb, sys_ram_avail_pre_gb, vram_avail_pre_gb, sys_ram_avail_post_gb, vram_avail_post_gb, load_time_sec, audit_context_length, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(llm_id, system_id) DO UPDATE SET
            sys_ram_total_gb=excluded.sys_ram_total_gb,
            vram_total_gb=excluded.vram_total_gb,
            sys_ram_avail_pre_gb=excluded.sys_ram_avail_pre_gb,
            vram_avail_pre_gb=excluded.vram_avail_pre_gb,
            sys_ram_avail_post_gb=excluded.sys_ram_avail_post_gb,
            vram_avail_post_gb=excluded.vram_avail_post_gb,
            load_time_sec=excluded.load_time_sec,
            audit_context_length=excluded.audit_context_length,
            last_updated=CURRENT_TIMESTAMP
        """
        
        values = (
            llm_id, system_id,
            metrics.get('sys_ram_total_gb'), metrics.get('vram_total_gb'),
            metrics.get('sys_ram_avail_pre_gb'), metrics.get('vram_avail_pre_gb'),
            metrics.get('sys_ram_avail_post_gb'), metrics.get('vram_avail_post_gb'),
            metrics.get('load_time_sec'),
            metrics.get('audit_context_length')
        )
        try:
            self.connection.execute(query, values)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error saving audit metrics: {e}")
            self.connection.rollback()

    def get_all_llms(self) -> list[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("DB not open")
        return self.connection.execute("SELECT * FROM llms ORDER BY llm_id").fetchall()
        
    def get_model_summary(self, system_id: int = 1) -> list[sqlite3.Row]:
        """
        Fetches model data joined with system-specific metrics for the CURRENT system.
        """
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        
        # FIX: Added l.available_on_disk to the SELECT clause
        query = """
        SELECT 
            l.llm_id, l.llm_identifier, l.display_name, l.model_key, l.format, 
            l.size_bytes, l.vision, l.trained_for_tool_use, l.max_context_length, 
            l.architecture, l.model_path, l.params_string,
            l.available_on_disk,
            l.backend_type,
            
            m.is_available AS sys_is_available,
            m.is_respondent AS sys_is_respondent,
            m.is_evaluator AS sys_is_evaluator,
            m.use_results AS sys_use_results,
            m.is_disabled AS sys_is_disabled,
            
            m.sys_ram_avail_post_gb AS system_ram_post,
            m.vram_avail_post_gb AS system_vram_post,
            m.sys_ram_total_gb AS system_ram_total,
            m.vram_total_gb AS system_vram_total,
            (m.sys_ram_total_gb - m.sys_ram_avail_post_gb) as max_ram_used,
            (m.vram_total_gb - m.vram_avail_post_gb) as max_vram_used
        FROM llms l
        INNER JOIN llm_system_metrics m ON l.llm_id = m.llm_id
        WHERE m.system_id = ?
        ORDER BY l.llm_identifier
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (system_id,))
            return cursor.fetchall()
        except sqlite3.OperationalError as e:
            if "no such column" in str(e).lower() and hasattr(self, "create_tables"):
                # Legacy DB missing backend_type/available_on_disk - run migrations and retry
                self.create_tables()
                cursor.execute(query, (system_id,))
                return cursor.fetchall()
            raise

    def get_model_summary_row(self, system_id: int, llm_id: int) -> sqlite3.Row | None:
        """Single-row variant of get_model_summary for one llm_id (e.g. toggle guards)."""
        if not self.connection:
            raise sqlite3.Error("Database connection is not open.")
        query = """
        SELECT 
            l.llm_id, l.llm_identifier, l.display_name, l.model_key, l.format, 
            l.size_bytes, l.vision, l.trained_for_tool_use, l.max_context_length, 
            l.architecture, l.model_path, l.params_string,
            l.available_on_disk,
            l.backend_type,
            
            m.is_available AS sys_is_available,
            m.is_respondent AS sys_is_respondent,
            m.is_evaluator AS sys_is_evaluator,
            m.use_results AS sys_use_results,
            m.is_disabled AS sys_is_disabled,
            
            m.sys_ram_avail_post_gb AS system_ram_post,
            m.vram_avail_post_gb AS system_vram_post,
            m.sys_ram_total_gb AS system_ram_total,
            m.vram_total_gb AS system_vram_total,
            (m.sys_ram_total_gb - m.sys_ram_avail_post_gb) as max_ram_used,
            (m.vram_total_gb - m.vram_avail_post_gb) as max_vram_used
        FROM llms l
        INNER JOIN llm_system_metrics m ON l.llm_id = m.llm_id
        WHERE m.system_id = ? AND l.llm_id = ?
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (system_id, llm_id))
            return cursor.fetchone()
        except sqlite3.OperationalError as e:
            if "no such column" in str(e).lower() and hasattr(self, "create_tables"):
                self.create_tables()
                cursor.execute(query, (system_id, llm_id))
                return cursor.fetchone()
            raise

    def get_cross_system_models(self) -> list[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("DB not open")
        query = """
        SELECT 
            s.name as system_name,
            l.llm_identifier,
            l.display_name,
            l.model_key,
            l.max_context_length,
            l.params_string,
            l.architecture,
            l.model_path,
            m.sys_ram_total_gb as total_ram
        FROM llm_system_metrics m
        JOIN llms l ON m.llm_id = l.llm_id
        JOIN systems s ON m.system_id = s.system_id
        ORDER BY s.name, l.llm_identifier
        """
        cursor = self.connection.cursor()
        cursor.execute(query)
        return cursor.fetchall()