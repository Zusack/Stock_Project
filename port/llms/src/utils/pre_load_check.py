# src/utils/pre_load_check.py
"""
Pre-service check: ensure LLM backends used by selected models are ready for model loading
before starting Benchmark, Evaluation, Memory Audit, Sequence, or Deep Audit.
Backends are determined by selected models (Run/Judge) in Model Manager, not a single Settings backend.
When process_type is provided, backends are derived from the pending work queue in the database.
"""
import flet as ft
from typing import Callable, Optional

from src.database.manager import DatabaseManager
from src.services.global_control_service import GlobalControlService
from src.utils.system_utils import SystemScanner
from src.backends import BackendFactory
from src.backends.types import BackendType


def _get_system_id(db: DatabaseManager) -> int:
    """Return the current system_id for the running machine."""
    try:
        scanner = SystemScanner()
        sys_row = db.get_system_by_hash(scanner.system_id)
        return sys_row["system_id"] if sys_row else 1
    except Exception:
        return 1


def _get_backend_types_for_pending_work(db: DatabaseManager, system_id: int, process_type: str) -> list:
    """
    Return distinct backend types required for the pending work queue.
    For 'generation': backends from get_missing_responses.
    For 'evaluation': backends from evaluator LLMs in get_missing_evaluations.
    Returns empty list if no pending work (caller may fall back to system selection).
    """
    try:
        if process_type == "generation":
            work = db.get_missing_responses(system_id=system_id)
            if not work:
                return []
            return list(dict.fromkeys(job[5] for job in work if len(job) > 5))
        if process_type == "evaluation":
            work = db.get_missing_evaluations(system_id=system_id, include_non_evaluators=False)
            if not work:
                return []
            evaluator_ids = list(dict.fromkeys(job[1] for job in work))
            return list(dict.fromkeys(db.get_backend_type(eid) or "lmstudio" for eid in evaluator_ids))
    except Exception:
        return []
    return []


def _get_backend_types_for_system(db: DatabaseManager) -> list:
    """Return distinct backend types for models selected (Run or Judge) on the current system."""
    try:
        scanner = SystemScanner()
        sys_row = db.get_system_by_hash(scanner.system_id)
        system_id = sys_row["system_id"] if sys_row else 1
    except Exception:
        return []
    try:
        cursor = db.connection.cursor()
        cursor.execute("""
            SELECT DISTINCT COALESCE(l.backend_type, 'lmstudio') AS backend_type
            FROM llms l
            JOIN llm_system_metrics m ON l.llm_id = m.llm_id AND m.system_id = ? AND m.is_available = 1
            WHERE m.is_respondent = 1 OR m.is_evaluator = 1
        """, (system_id,))
        rows = cursor.fetchall()
        return [row[0] for row in rows if row[0]]
    except Exception:
        return []


def _get_backends_for_system(db_path: str) -> list:
    """Create and connect backend instances for each backend type used by selected models."""
    with DatabaseManager(db_path) as db:
        db.connect()
        backend_types = _get_backend_types_for_system(db)
        if not backend_types:
            backend_types = ["lmstudio"]
        settings = db.get_backend_settings()
    backends = []
    for bt in backend_types:
        try:
            b = BackendFactory.create_from_string(bt, settings)
            b.connect()
            backends.append(b)
        except Exception:
            pass
    return backends


def ensure_ready_for_model_load(
    page: ft.Page,
    db_path: str,
    on_ready: Callable[[], None],
    on_cancel: Optional[Callable[[], None]] = None,
    on_connection_error: Optional[Callable[[str, str], None]] = None,
    process_type: Optional[str] = None,
):
    """
    Checks if models are loaded and either unloads (per setting) or prompts user.
    Calls on_ready() when it is safe to start the process. Calls on_cancel() if user cancels.

    Args:
        page: Flet page reference
        db_path: Path to database for settings
        on_ready: Callable with no args - invoked when ready to proceed
        on_cancel: Optional callable - invoked when user cancels (modal Cancel)
        on_connection_error: Optional callable(message, backend_name) - invoked when a backend
            cannot be connected; the view can set status banner and SnackBar.
        process_type: Optional "generation" or "evaluation" - when set, backends are derived
            from the pending work queue; otherwise from selected models.
    """
    with DatabaseManager(db_path) as db:
        db.connect()
        unload_all = db.get_setting("unload_all_before_process", "1") == "1"
        if process_type == "evaluation":
            system_id = GlobalControlService().active_system_id or _get_system_id(db)
        else:
            system_id = _get_system_id(db)
        if process_type:
            backend_types = _get_backend_types_for_pending_work(db, system_id, process_type)
            if not backend_types:
                on_ready()
                return
        else:
            backend_types = _get_backend_types_for_system(db)
        if not backend_types:
            backend_types = ["lmstudio"]
        settings = db.get_backend_settings()

    backends = []
    for bt in backend_types:
        backend_name = "backend"
        try:
            b = BackendFactory.create_from_string(bt, settings)
            backend_name = b.display_name
            b.connect()
            backends.append(b)
        except Exception as e:
            msg = str(e)
            try:
                backend_name = BackendType.from_string(bt).display_name()
            except Exception:
                pass
            if on_connection_error:
                on_connection_error(msg, backend_name)
            elif page:
                page.snack_bar = ft.SnackBar(ft.Text(f"Could not connect to {backend_name}: {msg}"))
                page.snack_bar.open = True
                page.update()
            return

    try:
        if unload_all:
            any_err = None
            for b in backends:
                ok, err = b.unload_all_models()
                if not ok and err:
                    any_err = err
            if any_err and page:
                page.snack_bar = ft.SnackBar(ft.Text(f"Failed to unload models: {any_err}"))
                page.snack_bar.open = True
                page.update()
                return
            on_ready()
            return

        any_loaded = False
        for b in backends:
            ids = b.list_loaded_models()
            if ids:
                any_loaded = True
                break
        if not any_loaded:
            on_ready()
            return
    except Exception as e:
        if page:
            page.snack_bar = ft.SnackBar(ft.Text(f"Could not check backends: {e}"))
            page.snack_bar.open = True
            page.update()
        return
    finally:
        for b in backends:
            try:
                b.disconnect()
            except Exception:
                pass

    # Models loaded and setting is No - show modal
    dlg_ref = [None]

    def _close_and_run(callback):
        if dlg_ref[0] and page:
            page.pop_dialog()
            page.update()
        if callback:
            callback()

    def on_cancel_click(e):
        _close_and_run(on_cancel)

    def on_yes_click(e):
        _close_and_run(None)
        GlobalControlService().load_as_new_instance = False
        backends_yes = _get_backends_for_system(db_path)
        any_err = None
        for b in backends_yes:
            try:
                ok, err = b.unload_all_models()
                if not ok and err:
                    any_err = err
            finally:
                try:
                    b.disconnect()
                except Exception:
                    pass
        if any_err and page:
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to unload: {any_err}"))
            page.snack_bar.open = True
        else:
            on_ready()
        if page:
            page.update()

    def on_no_click(e):
        _close_and_run(None)
        GlobalControlService().load_as_new_instance = True
        on_ready()
        if page:
            page.update()

    dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text("Model Load Warning"),
        content=ft.Text(
            "WARNING: Model detected already loaded in memory. Loading multiple models into "
            "memory may significantly affect performance. Should this model(s) be unloaded "
            "before proceeding with this process?"
        ),
        actions=[
            ft.TextButton("Cancel", on_click=on_cancel_click),
            ft.TextButton("Yes", on_click=on_yes_click),
            ft.TextButton("No", on_click=on_no_click),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )
    dlg_ref[0] = dlg
    page.open(dlg)
    page.update()
