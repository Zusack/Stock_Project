"""
Compatibility shim for analytics_service.py
This file maintains backward compatibility while the actual implementation
has been moved to src/services/analytics/ module.
"""
# Import from the new module structure
from src.services.analytics import AnalyticsService

# Re-export for backward compatibility
__all__ = ['AnalyticsService']
