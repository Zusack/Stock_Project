"""Analysis modules for the Stock Analyzer Flet app."""

from src.analysis.canslim import CanslimResult, analyze_canslim
from src.analysis.canslim_backtest import CanslimBacktestResult, run_canslim_backtest
from src.analysis.focused import FocusedResult, run_focused_optimization
from src.analysis.general import GeneralResult, evaluate_general_rules
from src.analysis.hybrid import HybridResult, run_hybrid_analysis
from src.analysis.technical import TechnicalResult, run_technical_analysis
from src.analysis.volatility import VolatilityResult, run_volatility_analysis
from src.analysis.ingest import ingest_stock_data, init_db

__all__ = [
    "CanslimResult",
    "analyze_canslim",
    "CanslimBacktestResult",
    "run_canslim_backtest",
    "FocusedResult",
    "run_focused_optimization",
    "GeneralResult",
    "evaluate_general_rules",
    "HybridResult",
    "run_hybrid_analysis",
    "TechnicalResult",
    "run_technical_analysis",
    "VolatilityResult",
    "run_volatility_analysis",
    "ingest_stock_data",
    "init_db",
]
