import sqlite3
import json
from typing import Optional

class SystemsRepository:
    def update_system_name(self, system_id: int, new_name: str):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("UPDATE systems SET name = ? WHERE system_id = ?", (new_name, system_id))
        self.connection.commit()

    def update_system_details(self, system_id: int, cpu_json: str, ram_json: str, gpu_json: str, notes: str, platform_info: Optional[str] = None):
        if not self.connection: raise sqlite3.Error("DB not open")
        if platform_info is not None:
            self.connection.execute("""
                UPDATE systems 
                SET cpu_info = ?, ram_info = ?, gpu_info = ?, notes = ?, platform_info = ?
                WHERE system_id = ?
            """, (cpu_json, ram_json, gpu_json, notes, platform_info, system_id))
        else:
            self.connection.execute("""
                UPDATE systems 
                SET cpu_info = ?, ram_info = ?, gpu_info = ?, notes = ?
                WHERE system_id = ?
            """, (cpu_json, ram_json, gpu_json, notes, system_id))
        self.connection.commit()

    def delete_system(self, system_id: int):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("DELETE FROM systems WHERE system_id = ?", (system_id,))
        self.connection.commit()

    def get_all_systems(self) -> list[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM systems ORDER BY created_at DESC")
        return cursor.fetchall()

    def get_system_by_hash(self, hardware_hash: str) -> Optional[sqlite3.Row]:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM systems WHERE hardware_hash = ?", (hardware_hash,))
        return cursor.fetchone()

    def get_or_create_system(self, name: str, hardware_hash: str, 
                             os_info: dict, cpu_info: dict, ram_info: dict, gpu_info: list) -> int:
        if not self.connection: raise sqlite3.Error("Database connection is not open.")
        cursor = self.connection.cursor()
        cursor.execute("SELECT system_id FROM systems WHERE hardware_hash = ?", (hardware_hash,))
        row = cursor.fetchone()
        if row: return row['system_id']
        else:
            try:
                cursor.execute("""
                    INSERT INTO systems (name, hardware_hash, os_info, cpu_info, ram_info, gpu_info)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (name, hardware_hash, json.dumps(os_info), json.dumps(cpu_info), json.dumps(ram_info), json.dumps(gpu_info)))
                self.connection.commit()
                return cursor.lastrowid
            except sqlite3.Error as e:
                self.connection.rollback()
                return -1

    def get_system_by_fuzzy_match(self, signature: str) -> list[sqlite3.Row]:
        """Finds systems with matching hardware specs but different hashes."""
        if not self.connection: raise sqlite3.Error("DB not open")
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM systems WHERE fuzzy_signature = ?", (signature,))
        return cursor.fetchall()

    def rebind_system_hash(self, system_id: int, new_hash: str, new_details: dict):
        """Updates an existing system record with the current hardware hash/info."""
        if not self.connection: raise sqlite3.Error("DB not open")
        
        self.connection.execute("""
            UPDATE systems 
            SET hardware_hash = ?, 
                os_info = ?, cpu_info = ?, ram_info = ?, gpu_info = ?,
                last_updated = CURRENT_TIMESTAMP
            WHERE system_id = ?
        """, (
            new_hash, 
            json.dumps(new_details['os']), 
            json.dumps(new_details['cpu']),
            json.dumps(new_details['ram']),
            json.dumps(new_details['gpus']),
            system_id
        ))
        self.connection.commit()