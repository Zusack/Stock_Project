import threading
import time
import os
import psutil

from src.database.manager import DatabaseManager, DEFAULT_DB_FILE
import src.utils.metadata_utils as metadata_utils
from src.utils.logger_utils import app_logger 
from src.services.event_bus import event_bus

# Services
from src.services.global_control_service import GlobalControlService
from src.services.thermal_monitor import ThermalMonitor
from src.services.generation_engine import GenerationEngine
from src.services.evaluation_engine import EvaluationEngine
from src.services.audit_engine import AuditEngine
from src.services.deep_audit_engine import DeepAuditEngine

# Backend abstraction
from src.backends import BackendFactory, BackendType, LLMBackend

class BenchmarkController(threading.Thread):
    def __init__(self, context_length: int = 8192, model_id: int = None, model_identifier: str = None, audit_target_llm_ids: list = None):
        super().__init__()
        
        # CRITICAL FIX: Daemon threads die automatically when the main program exits.
        self.daemon = True 
        
        self.db = None 
        self.global_control = GlobalControlService()
        
        # Link flags
        self.stop_event = self.global_control.stop_event 
        self.run_event = self.global_control.run_event
        self.skip_llm_event = self.global_control.skip_event

        # Config
        self.context_length = context_length 
        self.mode = "GENERATION" 
        self.inference_config = {
            "temperature": 0.7,
            "max_tokens": context_length
        }
        
        # Target for Deep Audit
        self.target_model_id = model_id
        self.target_model_identifier = model_identifier
        
        # Optional: for Memory Audit Re-Audit, only run on these llm_ids
        self.audit_target_llm_ids = audit_target_llm_ids
        
        # Sub-Services
        self.thermal_monitor = None
        self.generation_engine = None
        self.evaluation_engine = None
        self.audit_engine = None
        self.deep_audit_engine = None
        
        # Backends: created per backend_type from selected models (no single active_backend from Settings)
        self._backends: dict[str, LLMBackend] = {}
        
        # Track if we've already logged stop detection to avoid spam
        self._stop_logged = False
        
        psutil.cpu_percent(interval=None)
        self._log(f"Initialized with Context: {context_length}")

    def _get_backend_settings(self) -> dict:
        """Return backend settings from database (used when creating backends per model)."""
        if not self.db:
            return {}
        return self.db.get_backend_settings()

    def get_backend(self, backend_type: str) -> LLMBackend:
        """Return (and cache) the backend for the given type. Backend is determined by selected models, not Settings."""
        bt = (backend_type or "").strip().lower() or "lmstudio"
        if bt not in self._backends:
            settings = self._get_backend_settings()
            self._backends[bt] = BackendFactory.create_from_string(bt, settings)
            self._backends[bt].connect()
            self._log(f"Backend: {self._backends[bt].display_name} (for {bt})")
        return self._backends[bt]

    def run(self):
        app_logger.log("CONTROLLER", f"Starting Process: {self.mode}")
        self.global_control.request_start()
        
        try:
            self.db = DatabaseManager(DEFAULT_DB_FILE)
            self.db.connect()
            self.db.create_tables()
            
            # Backends are created on demand per model (get_backend(backend_type))
            self.thermal_monitor = ThermalMonitor(self)
            self.generation_engine = GenerationEngine(self)
            self.evaluation_engine = EvaluationEngine(self)
            self.audit_engine = AuditEngine(self)
            
            if self.target_model_id and self.target_model_identifier:
                target_backend_type = self.db.get_backend_type(self.target_model_id) or "lmstudio"
                target_backend = self.get_backend(target_backend_type)
                self.deep_audit_engine = DeepAuditEngine(self.db, self.target_model_id, self.target_model_identifier, self.context_length, backend=target_backend)
            
            self.thermal_monitor.start_monitoring()

            if self.mode == "EVALUATION": 
                self.evaluation_engine.run()
            elif self.mode == "AUDIT": 
                self.audit_engine.run()
            elif self.mode == "DEEP_AUDIT": 
                if self.deep_audit_engine:
                    self.deep_audit_engine.run()
                else:
                    self._log("Error: Deep Audit started without target model.")
            else: 
                self.generation_engine.run()
                
        except Exception as e:
            self._log(f"\n[FATAL ERROR]: {e}")
            app_logger.log("FATAL", str(e), level="ERROR")
            import traceback
            traceback.print_exc()
            msg = str(e)
            if "not reachable" in msg.lower() or "unreachable" in msg.lower():
                mode_label = {"GENERATION": "Benchmark", "EVALUATION": "Evaluation", "AUDIT": "Memory Audit", "DEEP_AUDIT": "Deep Audit"}.get(self.mode, self.mode)
                msg = f"Cannot run {mode_label}: {msg}"
            self._fatal_error = msg
        finally:
            app_logger.log("CONTROLLER", "Process Finished or Stopped.")
            
            # Disconnect all cached backends
            for _bt, backend in list(self._backends.items()):
                try:
                    backend.disconnect()
                except Exception:
                    pass
            self._backends.clear()
            
            # Ensure we signal the global stop if we finished naturally
            if not self.global_control.stop_event.is_set():
                self.global_control.notify_process_finished()
            
            if self.db and self.db.connection: 
                self.db.close()
            
            event_bus.emit("controller_finished", error=getattr(self, "_fatal_error", None))

    # --- API Methods ---
    def start_generation(self):
        self.mode = "GENERATION"
        self.start()

    def start_evaluation(self):
        self.mode = "EVALUATION"
        self.start()

    def start_audit(self):
        self.mode = "AUDIT"
        self.start()
        
    def start_deep_audit(self): 
        self.mode = "DEEP_AUDIT"
        self.start()

    # --- Internal Helpers ---
    def _log(self, message: str, level: str = "INFO", **kwargs):
        # Safety Check: Don't log if we are shutting down
        if not threading.main_thread().is_alive(): return

        print(f"[CONTROLLER] {message}")
        clean_msg = message.replace("[CONTROLLER] ", "")
        app_logger.log(self.mode, clean_msg, level=level, **kwargs)
        event_bus.emit("status_update", message=message)

    def _check_control_flags(self, is_streaming=False):
        if self.global_control.stop_event.is_set():
            # Check if this is a soft-stop (requires model unload but will auto-restart)
            if hasattr(self, 'thermal_monitor') and self.thermal_monitor and self.thermal_monitor.is_soft_stopped:
                if not self._stop_logged:
                    self._log("Soft-stop detected (Power Safety). Unloading model and waiting for power to drop...")
                    self._stop_logged = True
                return "SOFT_STOP"
            # Only log once per controller instance to avoid spam
            if not self._stop_logged:
                self._log("Stop detected. Exiting...")
                self._stop_logged = True
            return "STOP"
        if self.global_control.skip_event.is_set():
            return "SKIP"
        if self.global_control.error_call_event.is_set():
            return "ERROR_CALL"
            
        if not self.global_control.run_event.is_set():
            if self.global_control.thermal_paused:
                self._log("Paused for Thermal Cooling...")
            elif self.global_control.power_paused:
                self._log("Paused for Power Safety...")
            else:
                self._log("Paused by User.")
            while not self.global_control.run_event.is_set():
                if self.global_control.stop_event.is_set():
                    # Check for soft-stop again
                    if hasattr(self, 'thermal_monitor') and self.thermal_monitor and self.thermal_monitor.is_soft_stopped:
                        return "SOFT_STOP"
                    return "STOP"
                if self.global_control.skip_event.is_set(): return "SKIP"
                if self.global_control.error_call_event.is_set(): return "ERROR_CALL"
                time.sleep(0.5) 
            
            self._log("Resumed. Running...")
            
        return "CONTINUE"

    def _sync_static_data(self):
        return True