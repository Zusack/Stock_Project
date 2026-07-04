# src/utils/metadata_utils.py
"""
Utilities for retrieving LLM metadata.
Now delegates to the backend abstraction layer.
Kept for backward compatibility with views that call get_all_downloaded_llms().
"""
from typing import List, Optional, Tuple

from src.llm.backends import BackendFactory, BackendType


def get_all_downloaded_llms(backend_type: str = "lmstudio", settings: dict = None) -> Tuple[List[dict], Optional[str]]:
    """
    Gets a list of all downloaded/available LLM identifiers from the specified backend
    and extracts detailed metadata for the Model Summary tab.

    Args:
        backend_type: String name of the backend (e.g. 'lmstudio', 'ollama', 'gguf', 'vllm').
        settings: Backend-specific settings dict.

    Returns:
        (llm_list, error_message):
        - On success: (list of model dicts, None). Empty list when no models.
        - On backend unreachable: ([], error_message).
        - On other error: ([], "Error listing models: {detail}").
    """
    print(f"[METADATA] Connecting to {backend_type} backend...")
    try:
        backend = BackendFactory.create_from_string(backend_type, settings or {})
        backend.connect()
        try:
            models = backend.list_available_models()
            if not models:
                return ([], None)

            llm_list = []
            for model in models:
                if not model.identifier:
                    continue
                data = {
                    "identifier": model.identifier,
                    "model_key": model.model_key or model.identifier,
                    "display_name": model.display_name or "N/A",
                    "format": model.format or "N/A",
                    "size_bytes": model.size_bytes or 0,
                    "vision": model.vision,
                    "trained_for_tool_use": model.trained_for_tool_use,
                    "max_context_length": model.max_context_length or 0,
                    "params_string": model.params_string or "N/A",
                    "architecture": model.architecture or "N/A",
                    "model_path": model.model_path or "N/A",
                    "backend_type": model.backend_type.value,
                }
                llm_list.append(data)
            return (llm_list, None)
        finally:
            backend.disconnect()

    except Exception as e:
        print(f"Error: Could not list models from {backend_type}. \nDetail: {e}")
        err_msg = str(e)
        if "not reachable" in err_msg.lower() or "not running" in err_msg.lower() or "connection refused" in err_msg.lower():
            return ([], err_msg)
        return ([], f"Error listing models: {e}")


if __name__ == "__main__":
    models, err = get_all_downloaded_llms()
    if err:
        print(f"Error: {err}")
    else:
        print(f"Found {len(models)} models.")
        if models:
            print("First model sample:", models[0])
