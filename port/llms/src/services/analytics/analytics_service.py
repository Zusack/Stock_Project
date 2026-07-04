"""
Main Analytics Service - Facade that coordinates data fetching, processing, and color management.
Maintains backward compatibility with existing code.
"""
from .data_fetcher import AnalyticsDataFetcher
from .data_processor import AnalyticsDataProcessor
from .color_manager import AnalyticsColorManager
from src.database.manager import DEFAULT_DB_FILE


class AnalyticsService:
    """
    Main analytics service that coordinates data fetching, processing, and visualization.
    Acts as a facade for the specialized analytics classes.
    """
    
    def __init__(self, db_path=DEFAULT_DB_FILE):
        self.db_path = db_path
        self.fetcher = AnalyticsDataFetcher(db_path)
        self.processor = AnalyticsDataProcessor()
        self.color_manager = AnalyticsColorManager()
    
    # --- Color Management (delegated) ---
    def get_color_for_category(self, category):
        """Get color for a category"""
        return self.color_manager.get_color_for_category(category)
    
    # --- Data Fetching (delegated) ---
    def get_raw_data(self, system_id=1):
        """Get raw benchmark data"""
        return self.fetcher.get_raw_data(system_id)
    
    # --- Data Processing (delegated with convenience methods) ---
    def get_leaderboard_data(
        self,
        system_id=1,
        include_speed=True,
        include_win_rate=True,
        include_load_time=True,
        include_token_efficiency=True,
        raw_data=None,
    ):
        """
        Get leaderboard data. Accepts raw_data to avoid redundant queries.
        """
        if raw_data is None:
            raw_data = self.get_raw_data(system_id)
        return self.processor.get_leaderboard_data(
            raw_data, include_speed, include_win_rate, include_load_time, include_token_efficiency
        )
    
    def calculate_krippendorff_alpha(self, system_id=1):
        """Calculate Krippendorff's alpha for inter-rater reliability"""
        raw_matrix = self.fetcher.get_evaluations_matrix(system_id)
        return self.processor.calculate_krippendorff_alpha(raw_matrix)
    
    def get_judge_bias_data(self, system_id=1):
        """Get judge bias statistics"""
        return self.fetcher.get_judge_bias_data(system_id)
    
    def get_conflict_items(self, system_id=1):
        """Get evaluation conflicts"""
        raw_evaluations = self.fetcher.get_all_evaluations_detailed(system_id)
        return self.processor.get_conflict_items(raw_evaluations)
    
    def get_consistency_stats(self, raw_data):
        """Get consistency statistics"""
        return self.processor.get_consistency_stats(raw_data)
    
    def get_category_trends(self, raw_data):
        """Get category trends"""
        return self.processor.get_category_trends(raw_data)
    
    def get_hardware_stats(self, raw_data):
        """Get hardware statistics"""
        return self.processor.get_hardware_stats(raw_data)
    
    def get_best_in_class(self, raw_data, category):
        """Get best model in a category"""
        return self.processor.get_best_in_class(raw_data, category)
    
    def get_performance_scatter_data(self, raw_data):
        """Get performance scatter data"""
        return self.processor.get_performance_scatter_data(raw_data)

    def get_model_analyzer_ordered_models(self, raw_data, leaderboard):
        """Model display names for Model Analyzer tab, ordered like the leaderboard."""
        return self.processor.model_analyzer_ordered_models(raw_data, leaderboard)

    def get_prompt_analyzer_categories(self, raw_data, hidden_categories=None):
        return self.processor.prompt_analyzer_categories(raw_data, hidden_categories)

    def get_prompt_analyzer_rows(
        self, raw_data, category: str, hidden_categories=None, sort_mode: str = "name"
    ):
        return self.processor.prompt_analyzer_sorted_rows(
            raw_data, category, hidden_categories, sort_mode
        )

    def process_comparison_data(self, system_ids, strict_mode=True):
        """
        Process comparison data for multiple systems.
        Maintains backward compatibility with existing signature.
        """
        # Fetch data
        all_systems = self.fetcher.get_all_systems()
        system_map = {s['system_id']: s['name'] for s in all_systems if s['system_id'] in system_ids}
        global_stats = self.fetcher.get_system_global_stats(system_ids)
        raw_data = self.fetcher.get_system_comparison_stats(system_ids)
        
        # Process data
        return self.processor.process_comparison_data(raw_data, system_ids, system_map, global_stats, strict_mode)

