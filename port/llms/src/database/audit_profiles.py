import sqlite3
from typing import Optional

class AuditProfilesRepository:
    def save_audit_profile(self, llm_id: int, system_id: int, data: dict):
        if not self.connection: raise sqlite3.Error("DB not open")
        # total_audit_duration_sec may be None when loading data from DBs merged from older versions; column accepts NULL.
        query = """
        INSERT INTO model_audit_profiles 
        (llm_id, system_id, mem_slope_mb_per_token, mem_intercept_gb, r_squared, 
         vram_at_max_offload_gb, ram_at_zero_offload_gb, kv_cache_savings_factor, 
         batch_overhead_gb_per_512, has_errors, error_count, total_audit_duration_sec, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(llm_id, system_id) DO UPDATE SET
            mem_slope_mb_per_token=excluded.mem_slope_mb_per_token,
            mem_intercept_gb=excluded.mem_intercept_gb,
            r_squared=excluded.r_squared,
            vram_at_max_offload_gb=excluded.vram_at_max_offload_gb,
            ram_at_zero_offload_gb=excluded.ram_at_zero_offload_gb,
            kv_cache_savings_factor=excluded.kv_cache_savings_factor,
            batch_overhead_gb_per_512=excluded.batch_overhead_gb_per_512,
            has_errors=excluded.has_errors,
            error_count=excluded.error_count,
            total_audit_duration_sec=excluded.total_audit_duration_sec,
            last_updated=CURRENT_TIMESTAMP
        """
        
        vals = (
            llm_id, system_id,
            data.get('slope', 0), data.get('intercept', 0), data.get('r2', 0),
            data.get('max_vram', 0), data.get('max_ram', 0),
            data.get('kv_factor', 1.0), data.get('batch_overhead', 0),
            data.get('has_errors', False), data.get('error_count', 0),
            data.get('total_audit_duration_sec'),
        )
        
        self.connection.execute(query, vals)
        self.connection.commit()

    def get_audit_profile(self, llm_id: int, system_id: int) -> Optional[dict]:
        if not self.connection: raise sqlite3.Error("DB not open")
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM model_audit_profiles WHERE llm_id = ? AND system_id = ?", (llm_id, system_id))
        row = cursor.fetchone()
        return dict(row) if row else None