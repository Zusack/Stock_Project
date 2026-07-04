import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .core import DatabaseCore
from .schema import DatabaseSchema
from .systems import SystemsRepository
from .llms import LLMRepository
from .prompts import PromptRepository
# Note: ResultsRepository now points to the package we created
from .results import ResultsRepository 
from .settings import SettingsRepository
from .audit_profiles import AuditProfilesRepository # <--- NEW IMPORT

# Single source of truth for application base dir (frozen = exe dir, dev = project root)
if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
    ROOT_DIR = os.path.dirname(sys.executable)
else:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DB_FILE = os.path.join(ROOT_DIR, "data", "benchmark.db")

# Default values for backend settings (single source for UI and backends)
BACKEND_SETTINGS_DEFAULTS = {
    "lmstudio_base_url": "http://localhost:1234",
    "ollama_host": "http://localhost:11434",
    "gguf_model_dir": "",
    "gguf_runtime_target": "auto",
    "gguf_include_subfolders": "1",
    "gguf_n_gpu_layers": "-1",
    "gguf_main_gpu": "0",
    # Android: "native" = libllama_bench_native (NDK); "python" = legacy subprocess + llama-cpp-python (desktop-only wheels).
    "gguf_android_runtime": "native",
    "vllm_base_url": "http://localhost:8000",
    "vllm_api_key": "",
}


class DatabaseManager(
    DatabaseCore,
    DatabaseSchema,
    SystemsRepository,
    LLMRepository,
    PromptRepository,
    ResultsRepository,
    SettingsRepository,
    AuditProfilesRepository # <--- INHERIT
):
    def __init__(self, db_file: str = DEFAULT_DB_FILE):
        super().__init__(db_file)

    def get_backend_settings(self) -> dict:
        """Return backend connection settings from database. Single source for controller, views, and pre_load_check."""
        return {
            key: self.get_setting(key, default)
            for key, default in BACKEND_SETTINGS_DEFAULTS.items()
        }

    def connect(self) -> None:
        """Connect and ensure schema migrations are applied (fixes legacy DBs missing backend_type, etc.)."""
        super().connect()
        try:
            self.create_tables()
        except Exception as e:
            print(f"Warning: Schema migration failed. Some features may not work. Error: {e}")