"""Local AI adapters (LM Studio-ready).

Only the text-analysis adapter is re-exported here to avoid import cycles:
guidance -> news_signals -> ai.adapter must not pull in advisor/context.
"""

from src.analysis.ai.adapter import AIAnalysisResult, analyze_headline, get_ai_adapter

__all__ = ["AIAnalysisResult", "analyze_headline", "get_ai_adapter"]
