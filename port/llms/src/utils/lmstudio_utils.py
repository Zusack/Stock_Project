# src/utils/lmstudio_utils.py
"""
Utilities for safely interacting with LM Studio models.
Handles proper cleanup and error handling during model unloading.
"""
import time
import lmstudio as lms

# Message used when LM Studio is unreachable (for consistent user-facing text)
LM_STUDIO_UNREACHABLE_MSG = "LM Studio is not reachable. Please start LM Studio and try again."

# Message when load fails due to model already loaded
LM_STUDIO_MODEL_ALREADY_LOADED_MSG = (
    "A model is already loaded in LM Studio. Unload it before loading another, or use "
    "Settings > LM Studio Manager > 'Unload all models before process' to auto-unload."
)


def get_loaded_llm_identifiers() -> tuple[list[str], str | None]:
    """
    Returns identifiers of LLMs currently loaded in LM Studio memory.

    Returns:
        (identifiers_list, error_message):
        - On success: ([...model_keys...], None). Empty list when no LLMs loaded.
        - On error: ([], "error message").
    """
    try:
        loaded = lms.list_loaded_models("llm")
    except AttributeError:
        try:
            with lms.Client() as client:
                loaded = client.llm.list_loaded()
        except Exception as e:
            return _fallback_get_loaded(e)
    except Exception as e:
        return _fallback_get_loaded(e)

    identifiers = []
    for model in (loaded or []):
        # LM Studio SDK uses 'identifier' on loaded model handles (LLM objects)
        key = getattr(model, "identifier", None) or getattr(model, "model_key", None) or getattr(model, "key", None)
        if key:
            identifiers.append(str(key))
    return (identifiers, None)


def _fallback_get_loaded(prior_error: Exception) -> tuple[list[str], str | None]:
    """
    Fallback when list_loaded_models is unavailable: try lms.llm() with no args.
    If it returns a model, we have at least one loaded (but we can't get its key easily).
    """
    try:
        model = lms.llm()
        if model is not None:
            key = getattr(model, "identifier", None) or getattr(model, "model_key", None) or getattr(model, "key", None)
            if key:
                return ([str(key)], None)
            return (["(model loaded)"], None)  # At least one loaded
    except Exception:
        pass
    return ([], str(prior_error))


def unload_all_models() -> tuple[bool, str | None]:
    """
    Unloads all LLMs currently in LM Studio memory.
    Returns (True, None) on success, (False, error_message) on failure.
    """
    try:
        loaded = lms.list_loaded_models("llm")
    except AttributeError:
        try:
            with lms.Client() as client:
                loaded = client.llm.list_loaded()
        except Exception as e:
            return _fallback_unload_all(e)
    except Exception as e:
        return _fallback_unload_all(e)

    errors = []
    for model in (loaded or []):
        try:
            if hasattr(model, "unload"):
                model.unload()
                time.sleep(0.2)
        except Exception as e:
            errors.append(str(e))

    return (len(errors) == 0, "; ".join(errors) if errors else None)


def _fallback_unload_all(prior_error: Exception) -> tuple[bool, str | None]:
    """
    Fallback when list_loaded_models is unavailable: repeatedly get lms.llm() and unload
    until no model is returned (or we hit a limit).
    """
    max_attempts = 5
    for _ in range(max_attempts):
        try:
            model = lms.llm()
            if model is None:
                return (True, None)
            if hasattr(model, "unload"):
                model.unload()
                time.sleep(0.3)
        except Exception as e:
            err_str = str(e).lower()
            if "no model" in err_str or "not loaded" in err_str or "model not found" in err_str:
                return (True, None)
            return (False, str(prior_error))
    return (True, None)


def is_model_load_blocked_error(error) -> bool:
    """
    Returns True if the error indicates loading failed because another model
    is already loaded (or similar resource conflict).
    """
    msg = str(error).lower() if error else ""
    triggers = [
        "failed to load model",
        "error loading model",
        "model has unloaded or crashed",
        "model already loaded",
        "cannot load",
        "operation canceled",
    ]
    return any(t in msg for t in triggers)


def is_lm_studio_unreachable_error(arg) -> bool:
    """
    Returns True if the given exception or error message indicates LM Studio
    is not reachable (e.g. not running, connection refused).
    Accepts Exception or str.
    """
    msg = str(arg).lower() if arg else ""
    triggers = [
        "connection refused",
        "econnrefused",
        "failed to connect",
        "cannot connect",
        "not reachable",
        "connectionrefusederror",
        "connection error",
        "lm studio is not reachable",
        "no connection",
        "connect econnrefused",
        "actively refused",
        "connection reset",
        "network is unreachable",
    ]
    return any(t in msg for t in triggers)


def safe_unload_model(llm, stream=None, wait_after_close=0.8):
    """
    Safely unloads an LM Studio model, ensuring all streams are closed
    and pending operations complete before unloading.
    
    Args:
        llm: The LM Studio LLM object to unload (can be None)
        stream: Optional stream object to close before unloading
        wait_after_close: Time in seconds to wait after closing stream (default: 0.8)
    
    Returns:
        bool: True if unload was successful or model was already unloaded, False on error
    """
    if llm is None:
        return True
    
    # Step 1: Close any active streams first with proper error handling
    if stream is not None:
        try:
            # Try to close the stream gracefully
            # Some streams may have a cancel method
            if hasattr(stream, 'cancel'):
                try:
                    stream.cancel()
                except:
                    pass
            
            # Close the stream
            stream.close()
            
            # Give LM Studio more time to process the close and finish any pending operations
            # This helps prevent "channel already closed" errors
            time.sleep(wait_after_close)
        except Exception as e:
            # Stream might already be closed or invalid, that's okay
            # We still want to try unloading the model
            pass
    
    # Step 2: Additional wait to ensure all channel operations complete
    # This helps prevent "unhandled message for already closed channel" errors
    time.sleep(0.2)
    
    # Step 3: Attempt to unload the model
    try:
        llm.unload()
        # Small delay to allow LM Studio to complete the unload operation
        time.sleep(0.15)
        return True
    except Exception as e:
        error_msg = str(e).lower()
        # Check if the error indicates the model is already unloaded
        # This is not a failure case - the model is in the desired state
        if any(phrase in error_msg for phrase in [
            "model unloaded",
            "no model loaded",
            "model not found",
            "not loaded",
            "channel",
            "already closed"
        ]):
            # Model is already unloaded or channel is closed, which is fine
            return True
        
        # For other errors, log but don't raise - we've done our best
        # The model might be in an inconsistent state, but we can't do more
        return False


def safe_unload_model_with_retry(llm, stream=None, max_retries=2, wait_between_retries=0.5):
    """
    Attempts to safely unload a model with retry logic.
    
    Args:
        llm: The LM Studio LLM object to unload (can be None)
        stream: Optional stream object to close before unloading
        max_retries: Maximum number of retry attempts (default: 2)
        wait_between_retries: Time to wait between retries in seconds (default: 0.5)
    
    Returns:
        bool: True if unload was successful, False if all retries failed
    """
    if llm is None:
        return True
    
    for attempt in range(max_retries):
        success = safe_unload_model(llm, stream=stream if attempt == 0 else None)
        if success:
            return True
        
        # Wait before retrying (except on last attempt)
        if attempt < max_retries - 1:
            time.sleep(wait_between_retries)
    
    return False

