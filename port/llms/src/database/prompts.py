import sqlite3

class PromptRepository:
    def delete_prompt(self, prompt_id: int):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("DELETE FROM prompts WHERE prompt_id = ?", (prompt_id,))
        self.connection.commit()

    def update_prompt_details(self, prompt_id: int, details: dict) -> None:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        set_clause = ", ".join([f"{key} = ?" for key in details.keys()])
        values = list(details.values())
        values.append(prompt_id)
        try:
            self.connection.execute(f"UPDATE prompts SET {set_clause} WHERE prompt_id = ?", tuple(values))
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error updating prompt details: {e}")
            self.connection.rollback()

    def rename_prompt_category(self, old_category: str, new_category: str):
        """Renames a category across all prompts that use it."""
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        try:
            self.connection.execute(
                "UPDATE prompts SET category = ? WHERE category = ?", 
                (new_category, old_category)
            )
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error renaming category: {e}")
            self.connection.rollback()

    def get_prompt_by_id(self, prompt_id: int) -> sqlite3.Row:
        """Fetches a single prompt row by ID."""
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM prompts WHERE prompt_id = ?", (prompt_id,))
        return cursor.fetchone()

    def get_or_create_prompt(self, name: str, category: str, prompt_text: str, rubric_text: str = "",
                             expected_response: str = "", assessment_text: str = "", scoring_criteria: str = "",
                             benchmark_type: str = "text", attachment_paths: str = "", tool_definition_json: str = "",
                             tool_script_path: str = "", reference_metric: str = "") -> int:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()

        # Check by NAME (Unique Identifier)
        cursor.execute("SELECT prompt_id FROM prompts WHERE name = ?", (name,))
        row = cursor.fetchone()

        details = {
            'category': category,
            'prompt_text': prompt_text,
            'rubric_text': rubric_text,
            'expected_response': expected_response,
            'assessment_text': assessment_text,
            'scoring_criteria': scoring_criteria,
            'benchmark_type': benchmark_type or "text",
            'attachment_paths': attachment_paths or "",
            'tool_definition_json': tool_definition_json or "",
            'tool_script_path': tool_script_path or "",
            'reference_metric': reference_metric or "",
        }
        if row:
            prompt_id = row['prompt_id']
            self.update_prompt_details(prompt_id, details)
            return prompt_id
        else:
            try:
                cursor.execute("""
                    INSERT INTO prompts (name, category, prompt_text, rubric_text, expected_response, assessment_text, scoring_criteria, benchmark_type, attachment_paths, tool_definition_json, tool_script_path, reference_metric)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (name, category, prompt_text, rubric_text, expected_response, assessment_text, scoring_criteria,
                      details['benchmark_type'], details['attachment_paths'], details['tool_definition_json'], details['tool_script_path'], details['reference_metric']))
                self.connection.commit()
                return cursor.lastrowid
            except sqlite3.Error as e:
                print(f"Error creating prompt: {e}")
                self.connection.rollback()
                raise

    def get_all_prompts(self) -> list[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        # Sort by Category, then by Name
        cursor.execute("SELECT * FROM prompts ORDER BY category ASC, name ASC")
        return cursor.fetchall()

    def get_active_prompts(self) -> list[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM prompts WHERE is_active = 1 ORDER BY category ASC, name ASC")
        return cursor.fetchall()