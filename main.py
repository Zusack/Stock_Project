"""
Entry point. Sets up sys.path, cache directories, baseline diagnostics,
and launches the Flet app via apps.desktop.main.run().

Run with:
    python main.py
"""
from __future__ import annotations

import os
import sys


def _configure_linux_rendering() -> None:
    """Opt-in environment knobs for tricky Linux GL stacks.

    All overrides are off by default. Set the corresponding env var to "1"
    before launching to enable. Mirrors the parent project's knobs.
    """
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("FLET_FORCE_SOFTWARE_GL") == "1":
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
        os.environ["GALLIUM_DRIVER"] = "llvmpipe"
    if os.environ.get("FLET_FORCE_GTK_CAIRO") == "1":
        os.environ["GSK_RENDERER"] = "cairo"
    if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("FLET_FORCE_GDK_X11") == "1":
        os.environ.setdefault("GDK_BACKEND", "x11")


def _configure_cache_dirs() -> None:
    """Keep transient downloads (HF / NLTK / transformers) inside the app folder.

    Harmless when those libraries aren't installed; the env vars simply go
    unused. This keeps the app self-contained and easy to uninstall.
    """
    app_base = (
        os.path.dirname(sys.executable)
        if getattr(sys, "frozen", False)
        else os.path.dirname(os.path.abspath(__file__))
    )
    cache = os.path.join(app_base, "cache")
    os.environ.setdefault("NLTK_DATA", os.path.join(cache, "nltk_data"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache, "transformers"))
    os.environ.setdefault("HF_HOME", os.path.join(cache, "huggingface"))


_configure_linux_rendering()
_configure_cache_dirs()

# Make absolute "src.*" imports resolvable when run from the project root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from apps.desktop.main import run  # noqa: E402
from src.analysis.db import ensure_db_ready  # noqa: E402
from src.analysis.backtest_schema import ensure_backtest_schema  # noqa: E402
from src.analysis.intelligence_schema import ensure_intelligence_schema  # noqa: E402
from src.analysis.llm_schema import ensure_llm_schema  # noqa: E402
from src.services.stock_config import stock_config  # noqa: E402
from src.utils.logger_utils import app_logger  # noqa: E402


if __name__ == "__main__":
    try:
        app_logger.enable_logging(level="INFO")
        app_logger.log(
            "BOOT",
            "App starting.",
            level="INFO",
            platform=sys.platform,
            python=sys.version.split(" ")[0],
        )
    except Exception as ex:
        print(f"Startup logger init failed: {ex}")

    db_status = ensure_db_ready(stock_config().db_path)
    if db_status.get("exists") and not db_status.get("ready"):
        app_logger.log(
            "BOOT",
            "Database WAL recovery reported an issue.",
            level="WARN",
            error=db_status.get("error"),
        )
    elif db_status.get("ready"):
        app_logger.log(
            "BOOT",
            "Database ready.",
            level="DEBUG",
            journal_mode=db_status.get("journal_mode"),
        )

    if db_status.get("exists"):
        try:
            ensure_intelligence_schema(stock_config().db_path)
            ensure_backtest_schema(stock_config().db_path)
            ensure_llm_schema(stock_config().db_path)
        except Exception as ex:
            app_logger.log(
                "BOOT",
                "Intelligence schema init failed.",
                level="WARN",
                error=str(ex),
            )
        try:
            from src.analysis.quotes_schema import ensure_quotes_schema
            from src.analysis.watchlist_schema import (
                ensure_watchlist_schema,
                sync_focus_to_default_watchlist,
            )
            from src.analysis.portfolio_schema import ensure_portfolio_schema

            db = stock_config().db_path
            ensure_quotes_schema(db)
            ensure_watchlist_schema(db)
            sync_focus_to_default_watchlist(db)
            ensure_portfolio_schema(db)
        except Exception as ex:
            app_logger.log(
                "BOOT",
                "UX schema init failed.",
                level="WARN",
                error=str(ex),
            )

    run()
