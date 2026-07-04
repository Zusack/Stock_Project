"""
Data fetching for analytics.
Handles all database queries for analytics data.
"""
from src.database.manager import DatabaseManager, DEFAULT_DB_FILE


class AnalyticsDataFetcher:
    """Handles fetching raw data from the database"""
    
    def __init__(self, db_path=DEFAULT_DB_FILE):
        self.db_path = db_path
    
    def get_raw_data(self, system_id=1):
        """
        Fetches raw data for the specific system ID provided.
        Returns list of dictionaries.
        """
        with DatabaseManager(self.db_path) as db:
            raw_data = db.get_full_benchmark_results(system_id)
        return [dict(row) for row in raw_data]
    
    def get_judge_bias_data(self, system_id=1):
        """Get judge bias statistics"""
        with DatabaseManager(self.db_path) as db:
            return db.get_judge_bias_stats(system_id)
    
    def get_all_evaluations_detailed(self, system_id=1):
        """Get all evaluations with details"""
        with DatabaseManager(self.db_path) as db:
            raw = db.get_all_evaluations_detailed(system_id)
        return [dict(row) for row in raw] if raw else []
    
    def get_evaluations_matrix(self, system_id=1):
        """Get evaluations matrix for inter-rater reliability"""
        with DatabaseManager(self.db_path) as db:
            return db.get_evaluations_matrix(system_id)
    
    def get_all_systems(self):
        """Get all registered systems"""
        with DatabaseManager(self.db_path) as db:
            systems = db.get_all_systems()
        return [dict(row) for row in systems] if systems else []
    
    def get_system_global_stats(self, system_ids):
        """Get global stats for multiple systems"""
        with DatabaseManager(self.db_path) as db:
            return db.get_system_global_stats(system_ids)
    
    def get_system_comparison_stats(self, system_ids):
        """Get comparison stats for multiple systems"""
        with DatabaseManager(self.db_path) as db:
            raw = db.get_system_comparison_stats(system_ids)
        return [dict(row) for row in raw] if raw else []

