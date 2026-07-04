import sqlite3

class DatabaseSchema:
    def create_tables(self) -> None:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        try:
            # 1. Systems
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS systems (
                system_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                hardware_hash TEXT UNIQUE,
                fuzzy_signature TEXT,
                os_info TEXT,
                cpu_info TEXT,
                ram_info TEXT,
                gpu_info TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Migration check
            cursor.execute("PRAGMA table_info(systems)")
            cols = [row['name'] for row in cursor.fetchall()]
            if 'fuzzy_signature' not in cols:
                cursor.execute("ALTER TABLE systems ADD COLUMN fuzzy_signature TEXT")

            # 2. LLMs (Identity = llm_identifier + backend_type so same model from different backends are distinct)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS llms (
                llm_id INTEGER PRIMARY KEY AUTOINCREMENT,
                llm_identifier TEXT NOT NULL,
                backend_type TEXT NOT NULL DEFAULT 'lmstudio',
                params_string TEXT,
                display_name TEXT,
                model_key TEXT,
                format TEXT,
                size_bytes INTEGER,
                vision BOOLEAN DEFAULT 0,
                trained_for_tool_use BOOLEAN DEFAULT 0,
                max_context_length INTEGER,
                architecture TEXT,
                model_path TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                available_on_disk BOOLEAN DEFAULT 1,
                UNIQUE (llm_identifier, backend_type)
            )
            """)

            # 3. Prompts
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE, 
                category TEXT DEFAULT 'Ungrouped',
                prompt_text TEXT NOT NULL,
                expected_response TEXT,
                assessment_text TEXT,
                scoring_criteria TEXT,
                rubric_text TEXT, 
                is_active BOOLEAN DEFAULT 1
            )
            """)

            # 4. Responses
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                response_id INTEGER PRIMARY KEY AUTOINCREMENT,
                llm_id INTEGER NOT NULL,
                prompt_id INTEGER NOT NULL,
                system_id INTEGER, 
                response_text TEXT,
                tokens_generated INTEGER,
                stop_reason TEXT,
                total_time_sec REAL,
                time_to_first_token_sec REAL,
                generation_time_sec REAL,
                streaming_tps REAL,
                total_tps REAL,
                context_length INTEGER,
                
                avg_cpu_usage REAL,
                avg_gpu_usage REAL,
                avg_cpu_temp REAL,
                avg_gpu_temp REAL,
                max_cpu_temp REAL,
                max_gpu_temp REAL,
                
                peak_vram_usage_gb REAL,
                peak_ram_usage_gb REAL,
                thermal_throttling_count INTEGER DEFAULT 0,
                power_throttling_count INTEGER DEFAULT 0,
                
                bleu_score REAL,
                rouge_score REAL,
                bert_score REAL,
                
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (llm_id) REFERENCES llms (llm_id) ON DELETE CASCADE,
                FOREIGN KEY (prompt_id) REFERENCES prompts (prompt_id) ON DELETE CASCADE,
                FOREIGN KEY (system_id) REFERENCES systems (system_id) ON DELETE CASCADE,
                UNIQUE (llm_id, prompt_id, system_id)
            )
            """)

            # 5. Evaluations
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id INTEGER NOT NULL,
                evaluator_llm_id INTEGER NOT NULL,
                prompt_category TEXT,
                rating INTEGER,
                rationale TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (response_id) REFERENCES responses (response_id) ON DELETE CASCADE,
                FOREIGN KEY (evaluator_llm_id) REFERENCES llms (llm_id) ON DELETE CASCADE,
                UNIQUE (response_id, evaluator_llm_id)
            )
            """)

            # 6. Error Logs
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_id INTEGER,
                llm_id INTEGER,
                process_step TEXT,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (llm_id) REFERENCES llms (llm_id) ON DELETE SET NULL,
                FOREIGN KEY (system_id) REFERENCES systems (system_id) ON DELETE CASCADE
            )
            """)

            # 7. App Settings
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """)
            
            # 8. LLM System Metrics & Status (Combined)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_system_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                llm_id INTEGER NOT NULL,
                system_id INTEGER NOT NULL,
                
                -- Status Flags (System Specific)
                is_available BOOLEAN DEFAULT 0,
                is_respondent BOOLEAN DEFAULT 0,
                is_evaluator BOOLEAN DEFAULT 0,
                use_results BOOLEAN DEFAULT 0,
                is_disabled BOOLEAN DEFAULT 0,
                
                -- Hardware Metrics
                sys_ram_avail_post_gb REAL,
                vram_avail_post_gb REAL,
                sys_ram_total_gb REAL,
                vram_total_gb REAL,
                sys_ram_avail_pre_gb REAL,
                vram_avail_pre_gb REAL,
                load_time_sec REAL,
                audit_context_length INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (llm_id) REFERENCES llms (llm_id) ON DELETE CASCADE,
                FOREIGN KEY (system_id) REFERENCES systems (system_id) ON DELETE CASCADE,
                UNIQUE (llm_id, system_id)
            )
            """)

            # 9. Deep Audit Profiles (New)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_audit_profiles (
                profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                llm_id INTEGER NOT NULL,
                system_id INTEGER NOT NULL,
                
                -- The Linear Regression: Memory = (Context * slope) + intercept
                mem_slope_mb_per_token REAL,
                mem_intercept_gb REAL,
                r_squared REAL,
                
                -- Offload Impact
                vram_at_max_offload_gb REAL,
                ram_at_zero_offload_gb REAL,
                
                -- Quantization & Batching
                kv_cache_savings_factor REAL DEFAULT 1.0, -- e.g. 0.65 means 35% savings
                batch_overhead_gb_per_512 REAL,
                
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (llm_id) REFERENCES llms (llm_id) ON DELETE CASCADE,
                FOREIGN KEY (system_id) REFERENCES systems (system_id) ON DELETE CASCADE,
                UNIQUE (llm_id, system_id)
            )
            """)
            
            # Run Migrations
            self._migrate_prompts_table(cursor)
            self._migrate_responses_table(cursor)
            self._migrate_llms_table(cursor)
            self._migrate_systems_structure(cursor)
            self._migrate_responses_metrics(cursor)
            self._migrate_prompts_category_structure(cursor)
            self._migrate_responses_hardware_stats(cursor)
            self._migrate_system_specific_flags(cursor)
            self._migrate_evaluations_table(cursor)
            self._migrate_audit_profiles_error_tracking(cursor)
            self._migrate_evaluations_duration(cursor)
            self._migrate_audit_profiles_duration(cursor)
            self._migrate_prompts_benchmark_type(cursor)
            self._migrate_prompts_reference_metric(cursor)
            self._migrate_prompts_tool_script_path(cursor)
            self._migrate_responses_tool_call_log(cursor)
            self._migrate_llms_backend_type(cursor)
            self._migrate_llms_composite_unique(cursor)
            self._migrate_is_disabled(cursor)
            self._migrate_audit_context_length(cursor)
            
            self.connection.commit()

        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")
            self.connection.rollback()
            raise

    # --- Migration Helpers ---
    def _migrate_prompts_table(self, cursor):
        cursor.execute("PRAGMA table_info(prompts)")
        columns = [row['name'] for row in cursor.fetchall()]
        new_cols = {'expected_response': 'TEXT', 'assessment_text': 'TEXT', 'scoring_criteria': 'TEXT', 'is_active': 'BOOLEAN DEFAULT 1'}
        for col, dtype in new_cols.items():
            if col not in columns: cursor.execute(f"ALTER TABLE prompts ADD COLUMN {col} {dtype}")

    def _migrate_responses_table(self, cursor):
        cursor.execute("PRAGMA table_info(responses)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'context_length' not in columns: cursor.execute("ALTER TABLE responses ADD COLUMN context_length INTEGER")
        if 'streaming_tps' not in columns: cursor.execute("ALTER TABLE responses ADD COLUMN streaming_tps REAL DEFAULT 0")
        if 'total_tps' not in columns: cursor.execute("ALTER TABLE responses ADD COLUMN total_tps REAL DEFAULT 0")
        new_cols = {
            'avg_cpu_usage': 'REAL', 'avg_gpu_usage': 'REAL',
            'avg_cpu_temp': 'REAL', 'avg_gpu_temp': 'REAL',
            'max_cpu_temp': 'REAL', 'max_gpu_temp': 'REAL'
        }
        for col, dtype in new_cols.items():
            if col not in columns: cursor.execute(f"ALTER TABLE responses ADD COLUMN {col} {dtype}")

    def _migrate_llms_table(self, cursor):
        # Note: We are moving away from global status flags in 'llms'.
        # We only ensure 'available_on_disk' exists here.
        cursor.execute("PRAGMA table_info(llms)")
        columns = [row['name'] for row in cursor.fetchall()]
        
        if 'available_on_disk' not in columns: 
            cursor.execute("ALTER TABLE llms ADD COLUMN available_on_disk BOOLEAN DEFAULT 1")

    def _migrate_systems_structure(self, cursor):
        cursor.execute("PRAGMA table_info(systems)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'notes' not in columns: cursor.execute("ALTER TABLE systems ADD COLUMN notes TEXT")
        if 'last_updated' not in columns:
            # Use constant default for older SQLite (no CURRENT_TIMESTAMP in ALTER TABLE)
            cursor.execute("ALTER TABLE systems ADD COLUMN last_updated TIMESTAMP")
            cursor.execute("UPDATE systems SET last_updated = CURRENT_TIMESTAMP WHERE last_updated IS NULL")
        if 'platform_info' not in columns:
            cursor.execute("ALTER TABLE systems ADD COLUMN platform_info TEXT")
    
    def _migrate_evaluations_table(self, cursor):
        """Add raw_response column to evaluations table for debugging"""
        cursor.execute("PRAGMA table_info(evaluations)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'raw_response' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN raw_response TEXT")
        cursor.execute("PRAGMA table_info(responses)")
        resp_columns = [row['name'] for row in cursor.fetchall()]
        if 'system_id' not in resp_columns: cursor.execute("ALTER TABLE responses ADD COLUMN system_id INTEGER REFERENCES systems(system_id)")
    
    def _migrate_audit_profiles_error_tracking(self, cursor):
        """Add error tracking columns to model_audit_profiles table"""
        cursor.execute("PRAGMA table_info(model_audit_profiles)")
        columns = [row['name'] for row in cursor.fetchall()]
        
        if 'has_errors' not in columns:
            print("Migrating DB: Adding has_errors to model_audit_profiles...")
            cursor.execute("ALTER TABLE model_audit_profiles ADD COLUMN has_errors BOOLEAN DEFAULT 0")
        
        if 'error_count' not in columns:
            print("Migrating DB: Adding error_count to model_audit_profiles...")
            cursor.execute("ALTER TABLE model_audit_profiles ADD COLUMN error_count INTEGER DEFAULT 0")

    def _migrate_evaluations_duration(self, cursor):
        """Add evaluation_time_sec for task time estimation. Merged DBs from older versions may have NULL here."""
        cursor.execute("PRAGMA table_info(evaluations)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'evaluation_time_sec' not in columns:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN evaluation_time_sec REAL")

    def _migrate_audit_profiles_duration(self, cursor):
        """Add total_audit_duration_sec for task time estimation. Merged DBs from older versions may have NULL here."""
        cursor.execute("PRAGMA table_info(model_audit_profiles)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'total_audit_duration_sec' not in columns:
            cursor.execute("ALTER TABLE model_audit_profiles ADD COLUMN total_audit_duration_sec REAL")

    def _migrate_responses_metrics(self, cursor):
        cursor.execute("PRAGMA table_info(responses)")
        columns = [row['name'] for row in cursor.fetchall()]
        new_cols = {'bleu_score': 'REAL', 'rouge_score': 'REAL', 'bert_score': 'REAL'}
        for col, dtype in new_cols.items():
            if col not in columns: cursor.execute(f"ALTER TABLE responses ADD COLUMN {col} {dtype}")

    def _migrate_prompts_category_structure(self, cursor):
        cursor.execute("PRAGMA table_info(prompts)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'category' in columns and 'name' not in columns:
            try: cursor.execute("ALTER TABLE prompts RENAME COLUMN category TO name")
            except: pass
        cursor.execute("PRAGMA table_info(prompts)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'category' not in columns: cursor.execute("ALTER TABLE prompts ADD COLUMN category TEXT DEFAULT 'Ungrouped'")

    def _migrate_responses_hardware_stats(self, cursor):
        cursor.execute("PRAGMA table_info(responses)")
        columns = [row['name'] for row in cursor.fetchall()]
        new_cols = {
            'peak_vram_usage_gb': 'REAL',
            'peak_ram_usage_gb': 'REAL',
            'thermal_throttling_count': 'INTEGER DEFAULT 0',
            'power_throttling_count': 'INTEGER DEFAULT 0',
        }
        for col, dtype in new_cols.items():
            if col not in columns:
                cursor.execute(f"ALTER TABLE responses ADD COLUMN {col} {dtype}")
                
    def _migrate_system_specific_flags(self, cursor):
        """Moves status flags to the system-specific table."""
        cursor.execute("PRAGMA table_info(llm_system_metrics)")
        columns = [row['name'] for row in cursor.fetchall()]
        
        flags = {
            'is_available': 'BOOLEAN DEFAULT 0',
            'is_respondent': 'BOOLEAN DEFAULT 0',
            'is_evaluator': 'BOOLEAN DEFAULT 0',
            'use_results': 'BOOLEAN DEFAULT 0'
        }
        
        for col, dtype in flags.items():
            if col not in columns:
                print(f"Migrating DB: Adding {col} to system metrics...")
                cursor.execute(f"ALTER TABLE llm_system_metrics ADD COLUMN {col} {dtype}")

    def _migrate_prompts_benchmark_type(self, cursor):
        """Add benchmark_type, attachment_paths, tool_definition_json to prompts."""
        cursor.execute("PRAGMA table_info(prompts)")
        columns = [row['name'] for row in cursor.fetchall()]
        new_cols = {
            'benchmark_type': 'TEXT DEFAULT "text"',
            'attachment_paths': 'TEXT',
            'tool_definition_json': 'TEXT',
        }
        for col, dtype in new_cols.items():
            if col not in columns:
                cursor.execute(f"ALTER TABLE prompts ADD COLUMN {col} {dtype}")

    def _migrate_prompts_reference_metric(self, cursor):
        """Add reference_metric for standardized benchmarks (bleu, rouge, bert)."""
        cursor.execute("PRAGMA table_info(prompts)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'reference_metric' not in columns:
            cursor.execute("ALTER TABLE prompts ADD COLUMN reference_metric TEXT")

    def _migrate_prompts_tool_script_path(self, cursor):
        """Add tool_script_path for custom tool modules (Phase 4)."""
        cursor.execute("PRAGMA table_info(prompts)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'tool_script_path' not in columns:
            cursor.execute("ALTER TABLE prompts ADD COLUMN tool_script_path TEXT")

    def _migrate_responses_tool_call_log(self, cursor):
        """Add tool_call_log to responses for tool/MCP runs."""
        cursor.execute("PRAGMA table_info(responses)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'tool_call_log' not in columns:
            cursor.execute("ALTER TABLE responses ADD COLUMN tool_call_log TEXT")

    def _migrate_llms_backend_type(self, cursor):
        """Add backend_type column to llms table and update UNIQUE constraint."""
        cursor.execute("PRAGMA table_info(llms)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'backend_type' not in columns:
            print("Migrating DB: Adding backend_type to llms...")
            cursor.execute("ALTER TABLE llms ADD COLUMN backend_type TEXT DEFAULT 'lmstudio'")
            # Update existing rows
            cursor.execute("UPDATE llms SET backend_type = 'lmstudio' WHERE backend_type IS NULL")
            # Drop old unique index and create new one with backend_type
            # SQLite doesn't support DROP INDEX IF EXISTS directly via PRAGMA,
            # so we check for existing indexes first
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='llms'")
            existing_indexes = [row['name'] for row in cursor.fetchall()]
            for idx_name in existing_indexes:
                if 'llm_identifier' in idx_name.lower():
                    try:
                        cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")
                    except Exception:
                        pass
            # Create new unique index on (llm_identifier, backend_type)
            try:
                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_llms_identifier_backend
                    ON llms (llm_identifier, backend_type)
                """)
            except Exception as e:
                print(f"Note: Could not create unique index on (llm_identifier, backend_type): {e}")
                # This can happen if there are duplicate llm_identifiers already
                # The constraint will still be enforced at the application level

    def _migrate_llms_composite_unique(self, cursor):
        """Recreate llms so (llm_identifier, backend_type) is unique instead of llm_identifier alone.
        Allows the same model name from different backends (LM Studio, Ollama, etc.) to have separate rows."""
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='llms'")
        row = cursor.fetchone()
        if not row:
            return
        create_sql = (row[0] or "").lower()
        # Already migrated: new schema has composite unique in the table definition
        if "unique (llm_identifier, backend_type)" in create_sql:
            return
        # Old schema: column-level "llm_identifier ... UNIQUE" blocks multiple backends; recreate table
        if "llm_identifier" not in create_sql:
            return
        print("Migrating DB: Recreating llms table for (llm_identifier, backend_type) uniqueness...")
        cursor.execute("""
            CREATE TABLE llms_new (
                llm_id INTEGER PRIMARY KEY AUTOINCREMENT,
                llm_identifier TEXT NOT NULL,
                backend_type TEXT NOT NULL DEFAULT 'lmstudio',
                params_string TEXT,
                display_name TEXT,
                model_key TEXT,
                format TEXT,
                size_bytes INTEGER,
                vision BOOLEAN DEFAULT 0,
                trained_for_tool_use BOOLEAN DEFAULT 0,
                max_context_length INTEGER,
                architecture TEXT,
                model_path TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                available_on_disk BOOLEAN DEFAULT 1,
                UNIQUE (llm_identifier, backend_type)
            )
        """)
        # Preserve llm_id for foreign key references; include available_on_disk if present in old table
        cursor.execute("PRAGMA table_info(llms)")
        old_cols = [r["name"] for r in cursor.fetchall()]
        has_available = "available_on_disk" in old_cols
        if has_available:
            cursor.execute("""
                INSERT INTO llms_new (llm_id, llm_identifier, backend_type, params_string, display_name, model_key, format, size_bytes, vision, trained_for_tool_use, max_context_length, architecture, model_path, last_updated, available_on_disk)
                SELECT llm_id, llm_identifier, COALESCE(NULLIF(backend_type, ''), 'lmstudio'), params_string, display_name, model_key, format, size_bytes, vision, trained_for_tool_use, max_context_length, architecture, model_path, last_updated, COALESCE(available_on_disk, 1) FROM llms
            """)
        else:
            cursor.execute("""
                INSERT INTO llms_new (llm_id, llm_identifier, backend_type, params_string, display_name, model_key, format, size_bytes, vision, trained_for_tool_use, max_context_length, architecture, model_path, last_updated)
                SELECT llm_id, llm_identifier, COALESCE(NULLIF(backend_type, ''), 'lmstudio'), params_string, display_name, model_key, format, size_bytes, vision, trained_for_tool_use, max_context_length, architecture, model_path, last_updated FROM llms
            """)
        cursor.execute("DROP TABLE llms")
        cursor.execute("ALTER TABLE llms_new RENAME TO llms")

    def _migrate_is_disabled(self, cursor):
        """Add is_disabled column to llm_system_metrics."""
        cursor.execute("PRAGMA table_info(llm_system_metrics)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'is_disabled' not in columns:
            print("Migrating DB: Adding is_disabled to llm_system_metrics...")
            cursor.execute("ALTER TABLE llm_system_metrics ADD COLUMN is_disabled BOOLEAN DEFAULT 0")

    def _migrate_audit_context_length(self, cursor):
        """Add audit_context_length column to llm_system_metrics."""
        cursor.execute("PRAGMA table_info(llm_system_metrics)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'audit_context_length' not in columns:
            print("Migrating DB: Adding audit_context_length to llm_system_metrics...")
            cursor.execute("ALTER TABLE llm_system_metrics ADD COLUMN audit_context_length INTEGER")