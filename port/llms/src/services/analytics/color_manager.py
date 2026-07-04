"""
Color management for analytics visualizations.
Handles color assignment for categories and metrics.
"""
import flet as ft


class AnalyticsColorManager:
    """Manages color assignments for analytics charts"""
    
    def __init__(self):
        self.base_palette = [
            "#EF5350", "#42A5F5", "#66BB6A", 
            "#FFA726", "#AB47BC", "#26A69A",
            "#EC407A", "#26C6DA", "#FFCA28", 
            "#D4E157", "#5C6BC0", "#8D6E63"
        ]
        self.category_color_map = {}
    
    def get_color_for_category(self, category):
        """
        Get color for a category.
        Special categories get fixed colors, others get assigned from palette.
        """
        if category == "Speed (TPS)":
            return ft.Colors.BLUE_GREY_400
        if category == "Load Time":
            return ft.Colors.TEAL_400
        if category == "Win Rate":
            return ft.Colors.AMBER_600

        # Assign from palette if not already assigned
        if category not in self.category_color_map:
            idx = len(self.category_color_map) % len(self.base_palette)
            self.category_color_map[category] = self.base_palette[idx]
        
        return self.category_color_map[category]
    
    def reset_color_map(self):
        """Reset the color mapping (useful for testing)"""
        self.category_color_map.clear()

