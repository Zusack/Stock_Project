import threading
from typing import Optional

import flet as ft

class GlobalControlService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalControlService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized: return
        self._initialized = True
        
        self.page_ref = None
        
        # Threading Events
        self.run_event = threading.Event()
        self.run_event.set() 
        self.stop_event = threading.Event()
        self.skip_event = threading.Event()
        self.error_call_event = threading.Event()
        
        # Process State
        self.user_paused = False
        self.thermal_paused = False
        self.power_paused = False
        self.output_view_paused = False  # Paused due to expanded output window
        self.soft_stop_active = False  # Track if we're in soft-stop mode (auto-restart pending)
        self.active_process_name = None
        
        # --- NEW: Fleet Management State ---
        self.physical_system_id = None # The ID of the hardware we are running on
        self.active_system_id = None   # The ID of the system we are VIEWING
        self.admin_mode = False        # Override flag

        # When True, engines should use load_new_instance() so a model loads alongside any already-loaded model (user chose "No" in modal).
        self.load_as_new_instance = False

        # Inspector selection (from Model Manager / Prompt Manager)
        self.selected_model_for_inspector = None   # llm_id or None
        self.selected_prompt_for_inspector = None  # prompt_id or None

        # Active tab tracking (set by app.py on_tab_change)
        self.active_tab_index = 0

        # Top tab strip loading indicator (registered from app.py; used for tab switches + heavy tab loads)
        self._tab_strip_busy_depth = 0
        self._tab_strip_loading_message = ""
        self._tab_strip_ring: Optional[ft.ProgressRing] = None
        self._tab_strip_label: Optional[ft.Text] = None

    def register_tab_strip_loading_controls(
        self,
        ring: ft.ProgressRing,
        label: Optional[ft.Text] = None,
    ) -> None:
        """Wire the main-window tab row spinner (app.py). Safe to call once at startup."""
        self._tab_strip_ring = ring
        self._tab_strip_label = label
        self._apply_tab_strip_loading()

    def _apply_tab_strip_loading(self) -> None:
        busy = self._tab_strip_busy_depth > 0
        msg = (self._tab_strip_loading_message or "").strip()
        if self._tab_strip_ring is not None:
            self._tab_strip_ring.visible = busy
        if self._tab_strip_label is not None:
            self._tab_strip_label.visible = busy
            self._tab_strip_label.value = msg if msg else "Loading…"

    def push_tab_strip_loading(self, message: str = "", *, flush_page: bool = True) -> None:
        """Mark global tab-row loading busy (reference counted). Call only from the UI event loop."""
        self._tab_strip_busy_depth += 1
        if message:
            self._tab_strip_loading_message = message
        self._apply_tab_strip_loading()
        if flush_page and self.page_ref:
            try:
                self.page_ref.update()
            except Exception:
                pass

    def pop_tab_strip_loading(self, *, flush_page: bool = True) -> None:
        """Release one tab-row loading level. When depth hits zero, the spinner hides."""
        self._tab_strip_busy_depth = max(0, self._tab_strip_busy_depth - 1)
        if self._tab_strip_busy_depth == 0:
            self._tab_strip_loading_message = ""
        self._apply_tab_strip_loading()
        if flush_page and self.page_ref:
            try:
                self.page_ref.update()
            except Exception:
                pass

    def set_model_selected_for_inspector(self, llm_id):
        """Set the model selected in Model Manager for sync to Benchmark/Evaluation Inspector."""
        self.selected_model_for_inspector = llm_id

    def set_prompt_selected_for_inspector(self, prompt_id):
        """Set the prompt selected in Prompt Manager for sync to Benchmark/Evaluation Inspector."""
        self.selected_prompt_for_inspector = prompt_id

    def register_page(self, page: ft.Page):
        self.page_ref = page

    # --- System Context Methods ---
    def set_system_context(self, physical_id: int):
        """Called on app startup to register the physical hardware ID."""
        self.physical_system_id = physical_id
        if self.active_system_id is None:
            self.active_system_id = physical_id

    def select_system(self, system_id: int):
        """Switches the view to a different system."""
        self.active_system_id = system_id
        self._broadcast_state_change()

    def toggle_admin_mode(self, enabled: bool):
        """Enables/Disables Admin override."""
        self.admin_mode = enabled
        self._broadcast_state_change()

    def _broadcast_state_change(self):
        if self.page_ref:
            self.page_ref.pubsub.send_all("refresh_results")
            self.page_ref.pubsub.send_all("system_changed")

    # --- Process State (single source of truth for Stop All button) ---
    @property
    def is_process_running(self) -> bool:
        """True if any benchmark/evaluation/audit process is running or paused. Use to enable Stop All button."""
        return (not self.run_event.is_set()) or (self.active_process_name is not None)

    # --- Permission Properties ---
    @property
    def is_viewing_remote_system(self) -> bool:
        """True when active system differs from the physical machine."""
        if self.active_system_id is None or self.physical_system_id is None:
            return False
        return self.active_system_id != self.physical_system_id

    @property
    def is_remote_read_only_mode(self) -> bool:
        """True when viewing a remote system without admin override."""
        return self.is_viewing_remote_system and (not self.admin_mode)

    @property
    def is_interaction_allowed(self) -> bool:
        """
        Can the user Create/Edit/Delete data?
        Allowed if:
        1. No process is running.
        2. AND (We are on local system OR Admin Mode is ON).
        """
        if self.is_process_running:
            return False
        if self.admin_mode:
            return True
        return self.active_system_id == self.physical_system_id

    @property
    def is_execution_allowed(self) -> bool:
        """
        Can the user Start Benchmarks/Evaluations?
        STRICTLY allowed only if Active System == Physical System.
        Admin mode does NOT override this (prevents data pollution).
        """
        if self.is_process_running:
            return False
        
        return self.active_system_id == self.physical_system_id

    # --- Process Control Methods ---
    def set_active_process(self, name: str):
        self.active_process_name = name

    def request_start(self):
        self.stop_event.clear()
        self.skip_event.clear()
        self.error_call_event.clear()
        self.run_event.set()
        self.user_paused = False
        self.thermal_paused = False
        self.power_paused = False
        self.soft_stop_active = False

    def request_stop(self):
        print("[GLOBAL] Stop Requested.")
        self.stop_event.set()
        self.run_event.set() 
        self.active_process_name = None
        if self.page_ref:
            self.page_ref.pubsub.send_all("process_stopped")
            self.page_ref.pubsub.send_all("sequence_stopped")

    def notify_process_finished(self):
        self.active_process_name = None
        self.load_as_new_instance = False  # Reset so next run doesn't use new-instance unless user chooses again
        if self.page_ref:
            self.page_ref.pubsub.send_all("process_stopped")

    def request_pause(self, source="user"):
        print(f"[GLOBAL] Pause Requested by {source}.")
        if source == "user": self.user_paused = True
        elif source == "thermal": self.thermal_paused = True
        elif source == "power": self.power_paused = True
        elif source == "output_view": self.output_view_paused = True
        self.run_event.clear()
        if self.page_ref and source == "user":
             self.page_ref.pubsub.send_all({"type": "status_update", "message": "Paused by User", "is_processing": False})

    def request_resume(self, source="user"):
        print(f"[GLOBAL] Resume Requested by {source}.")
        if source == "user": self.user_paused = False
        elif source == "thermal": self.thermal_paused = False
        elif source == "power": self.power_paused = False
        elif source == "output_view": self.output_view_paused = False
        if not self.user_paused and not self.thermal_paused and not self.power_paused and not self.output_view_paused:
            self.run_event.set()
            if self.page_ref:
                self.page_ref.pubsub.send_all({"type": "status_update", "message": "Resumed. Running...", "is_processing": True})

    def request_skip(self):
        print("[GLOBAL] Skip Requested.")
        self.skip_event.set()
        self.run_event.set()
        
    def reset_skip(self):
        self.skip_event.clear()

    def request_error_call(self):
        """Record current prompt/evaluation as error and move on. Saves partial output with Error marker."""
        print("[GLOBAL] Error Call Requested.")
        self.error_call_event.set()
        self.run_event.set()

    def reset_error_call(self):
        self.error_call_event.clear()