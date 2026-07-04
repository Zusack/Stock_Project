import time
import json
import os
import flet as ft
import gc
import threading
import src.utils.memory_utils as memory_utils
from src.utils.system_utils import SystemScanner
from src.utils.stream_utils import iterate_stream_with_thermal_checks, JobTimeoutError, ErrorCallInterrupted
from src.utils.repetition_utils import detect_garbage_response, strip_thinking_blocks
from src.services.scoring_engine import ScoringEngine
from src.services.event_bus import event_bus
from src.services.tool_resolver import resolve_tools, TOOL_REGISTRY
from src.backends.types import CancellableStream, LLMHandle
from src.database.manager import DatabaseManager, DEFAULT_DB_FILE

VISION_TYPES = ("image_comprehension",)  # Image-based prompts (vision/OCR/table/chart collapsed)
TOOL_TYPES = ("tool_use", "mcp")

class GenerationEngine:
    def __init__(self, controller_ref):
        self.controller = controller_ref
        self.db = controller_ref.db
        self.scoring_engine = ScoringEngine() 
        self.consecutive_failures = 0
        # Max runtime (seconds) for a single prompt job; from user setting
        with DatabaseManager(DEFAULT_DB_FILE) as db:
            raw = db.get_setting("max_response_time", "3600")
        self.job_timeout_sec = int(raw) if (raw and raw.isdigit() and int(raw) > 0) else 3600

    def _is_critical_error(self, error_msg):
        msg = str(error_msg).lower()
        triggers = [
            "model has crashed", "exit code", "no model found",
            "connection refused", "channel closed", "unexpected unload",
            "failed to respond", "broken pipe",
            "exceeds the available context size", "model unloaded", "no model loaded", "model not found",
            "failed to parse tool call request",  # LM Studio / model tool-use incompatibility
        ]
        return any(t in msg for t in triggers)

    def run(self):
        scanner = SystemScanner()
        current_hash = scanner.system_id
        current_system_id = 1
        
        sys_row = self.db.get_system_by_hash(current_hash)
        if sys_row: current_system_id = sys_row['system_id']
        
        if not self.controller._sync_static_data(): return 
        
        all_llms = self.db.get_all_llms()
        all_prompts = self.db.get_active_prompts()
        prompt_map = {p['prompt_id']: dict(p) for p in all_prompts}
        
        completed_jobs_summary = self.db.get_all_responses_summary(system_id=current_system_id)
        
        self.controller.total_llms = len(all_llms)
        self.controller.total_prompts = len(all_prompts)
        event_bus.emit("sync_complete", all_llms=all_llms, all_prompts=all_prompts, completed_jobs=completed_jobs_summary)

        # Run benchmark jobs for all selected respondents (Run checkbox); backend per model from Model Manager
        work_queue = self.db.get_missing_responses(system_id=current_system_id)
        if not work_queue:
            self.controller._log("All work complete.")
            return

        llm_jobs = {}
        for job in work_queue:
            if job[0] not in llm_jobs:
                llm_jobs[job[0]] = []
            llm_jobs[job[0]].append(job)

        CASCADE_LOAD_FAIL_LIMIT = 3
        consecutive_load_failures = 0

        self.controller.current_llm_num = 0

        for llm_id, jobs in llm_jobs.items():
            # Check stop BEFORE each model - prevents cascading past stop request
            control = self.controller._check_control_flags()
            if control in ("STOP", "SOFT_STOP"):
                self.controller._log("Stop detected between models. Exiting.")
                break

            # Backend for this model (from Model Manager / llms.backend_type)
            current_backend_type = jobs[0][5] if len(jobs[0]) > 5 else "lmstudio"
            backend = self.controller.get_backend(current_backend_type)

            if consecutive_load_failures >= CASCADE_LOAD_FAIL_LIMIT:
                msg = (
                    f"[FATAL] {consecutive_load_failures} consecutive model load failures. "
                    f"Likely a model is already loaded in {backend.display_name}. Unload it and retry."
                )
                self.controller._log(msg)
                raise RuntimeError(backend.model_already_loaded_message) from None

            self.controller.current_llm_num += 1
            current_llm_identifier = jobs[0][2]
            self.consecutive_failures = 0
            
            llm_was_skipped = False
            llm_loop_stop = False
            
            event_bus.emit("llm_active", llm_id=llm_id)
            event_bus.emit("new_llm_loop", llm_id=llm_id, all_prompts=all_prompts)
            
            self.controller._log(f"Loading {current_llm_identifier} (Ctx: {self.controller.context_length})...")
            start_load = time.time()
            
            handle: LLMHandle | None = None
            
            # --- Initial Load ---
            try:
                load_config = {'gpu_layers': -1, 'context_length': self.controller.context_length}
                if self.controller.global_control.load_as_new_instance and backend.supports_load_alongside():
                    handle = backend.load_model_alongside(current_llm_identifier, load_config)
                else:
                    handle = backend.load_model(current_llm_identifier, load_config)
                consecutive_load_failures = 0  # Reset on successful load
                load_time = round(time.time() - start_load, 4)
                self.controller._log(f"Model loaded in {load_time}s.")
                self.db.update_load_time_sec(llm_id, current_system_id, load_time)
            except Exception as e:
                if backend.is_unreachable_error(e):
                    raise RuntimeError(backend.unreachable_message) from e
                if backend.is_load_blocked_error(e):
                    consecutive_load_failures += 1
                msg = f"[FATAL] Load failed: {e}"
                self.controller._log(msg)
                self.db.log_error(current_system_id, llm_id, "Benchmark Load", msg)
                for job in jobs:
                    self.db.add_response(job[0], job[1], {'response_text': f"Error: {e}", 'stop_reason': 'Error_Load'}, system_id=current_system_id)
                event_bus.emit("llm_complete", llm_id=llm_id, color=ft.Colors.RED_800)
                continue
            
            # --- Job Loop ---
            for i, job in enumerate(jobs):
                if self.consecutive_failures >= 3:
                    self.controller._log(f"[STOP] 3 Consecutive Crashes for {current_llm_identifier}. Skipping remaining jobs.")
                    event_bus.emit("llm_complete", llm_id=llm_id, color=ft.Colors.RED_900)
                    break

                job_llm_id, job_prompt_id, _, job_prompt_text, job_prompt_category = job[:5]
                event_bus.emit("prompt_active", prompt_id=job_prompt_id)
                    
                # 1. Pre-Flight Check
                control = self.controller._check_control_flags()
                if control == "STOP": 
                    llm_loop_stop = True; break
                if control == "SOFT_STOP":
                    if handle:
                        self.controller._log("Soft-stop: Unloading model...")
                        backend.unload_model(handle)
                        handle = None
                        gc.collect()
                    llm_loop_stop = True; break
                if control == "SKIP": 
                    llm_was_skipped = True; break

                # Wait for safe temperature before starting next benchmark prompt
                self.controller.thermal_monitor.wait_for_safe_start_temp()
                    
                # Check again after cooldown wait in case user stopped
                control = self.controller._check_control_flags()
                if control == "STOP": 
                    llm_loop_stop = True; break
                if control == "SOFT_STOP":
                    if handle:
                        self.controller._log("Soft-stop: Unloading model...")
                        backend.unload_model(handle)
                        handle = None
                        gc.collect()
                    llm_loop_stop = True; break
                if control == "SKIP": 
                    llm_was_skipped = True; break

                self.controller._log(f"-> Running Prompt: {job_prompt_category}")
                event_bus.emit("new_token", token=None)
                    
                # --- RETRY LOOP ---
                prompt_data = prompt_map.get(job_prompt_id) or {}
                benchmark_type = (prompt_data.get('benchmark_type') or 'text').strip() or 'text'
                response_stream = None
                full_response = ""
                tool_call_log = None

                for attempt in range(2):
                    response_stream = None
                    full_response = ""
                    start_time = time.time()
                    first_token_time = None
                    tokens_gen = 0
                    stop_reason = "N/A"
                    tool_call_log = None

                    try:
                        self.controller.thermal_monitor.reset_counters()
                        self.controller.thermal_monitor.collecting_vitals = True

                        if benchmark_type in VISION_TYPES:
                            # Vision: Chat + images via backend abstraction
                            attachment_paths = []
                            try:
                                raw = prompt_data.get('attachment_paths') or '[]'
                                attachment_paths = json.loads(raw) if isinstance(raw, str) else raw
                                if not isinstance(attachment_paths, list):
                                    attachment_paths = []
                            except (json.JSONDecodeError, TypeError):
                                pass
                            missing = [p for p in attachment_paths if not p or not os.path.exists(p)]
                            if missing:
                                raise ValueError(f"Image file(s) not found: {missing[:3]}")
                            try:
                                image_handles = [backend.prepare_image(p) for p in attachment_paths if p and os.path.exists(p)]
                            except Exception as img_err:
                                raise ValueError(f"Failed to prepare image(s): {img_err}") from img_err
                            if not image_handles:
                                raise ValueError("No valid images for vision prompt.")
                            messages = [{"role": "user", "content": job_prompt_text}]
                            response_stream = backend.chat_stream(handle, messages, self.controller.inference_config, images=image_handles)
                            _token_buffer = ""
                            _last_token_emit = 0.0
                            _TOKEN_EMIT_INTERVAL = 0.04
                            for chunk in iterate_stream_with_thermal_checks(
                                self.controller, response_stream, job_timeout_sec=self.job_timeout_sec
                            ):
                                if first_token_time is None:
                                    first_token_time = time.time()
                                content = chunk.content
                                if content:
                                    full_response += content
                                    tokens_gen += 1
                                    _token_buffer += content
                                    now = time.time()
                                    if now - _last_token_emit >= _TOKEN_EMIT_INTERVAL:
                                        if _token_buffer:
                                            event_bus.emit("new_token", token=_token_buffer)
                                            _token_buffer = ""
                                        _last_token_emit = now
                                    if detect_garbage_response(content, full_response):
                                        try:
                                            response_stream.cancel()
                                        except Exception:
                                            pass
                                        raise ValueError("Repetition/garbage loop detected")
                                if chunk.stop_reason:
                                    stop_reason = chunk.stop_reason
                            if _token_buffer:
                                event_bus.emit("new_token", token=_token_buffer)
                        elif benchmark_type in TOOL_TYPES:
                            # Tool use / MCP: act_with_tools() via backend abstraction.
                            # Run in a thread so we can respect Stop and job timeout.
                            tools = resolve_tools(prompt_data)
                            if not tools:
                                known = ", ".join(sorted(TOOL_REGISTRY.keys()))
                                raw = (prompt_data.get("tool_definition_json") or "")[:100]
                                raise ValueError(
                                    f"No tools resolved for tool_use/mcp prompt. "
                                    f"Known tools: {known}. "
                                    f"Re-import the prompt from Prompts tab (Refresh + Import) to sync tool_definition_json. "
                                    f"DB had: {raw!r}..."
                                )
                            tool_messages = [{"role": "user", "content": job_prompt_text}]
                            _fragment_buf = [""]
                            _last_fragment_emit = [0.0]
                            _FRAGMENT_EMIT_INTERVAL = 0.04

                            def _on_msg(m):
                                text = None
                                role = ""
                                if isinstance(m, dict):
                                    role = m.get("role", "")
                                    text = m.get("content", "")
                                else:
                                    role = getattr(m, "role", None) or getattr(m, "type", None) or ""
                                    text = getattr(m, "content", None) or getattr(m, "text", None) or ""
                                    if isinstance(text, (list, tuple)):
                                        parts = []
                                        for item in text:
                                            if hasattr(item, "text"):
                                                parts.append(str(item.text).strip())
                                            elif hasattr(item, "content"):
                                                parts.append(str(item.content).strip())
                                            elif isinstance(item, dict):
                                                parts.append((item.get("text") or item.get("content") or "").strip())
                                        text = " ".join(p for p in parts if p)
                                if text:
                                    prefix = "\n[Tool] " if role == "tool" else "\n"
                                    event_bus.emit("new_token", token=f"{prefix}{text}\n")

                            def _on_fragment(fragment, round_index=0):
                                content = getattr(fragment, "content", None)
                                if content:
                                    _fragment_buf[0] += content
                                    now = time.time()
                                    if now - _last_fragment_emit[0] >= _FRAGMENT_EMIT_INTERVAL:
                                        if _fragment_buf[0]:
                                            event_bus.emit("new_token", token=_fragment_buf[0])
                                            _fragment_buf[0] = ""
                                        _last_fragment_emit[0] = now

                            result_holder = []

                            def _run_act():
                                try:
                                    result = backend.act_with_tools(
                                        handle, tool_messages, tools,
                                        config=self.controller.inference_config,
                                        on_message=_on_msg,
                                        on_prediction_fragment=_on_fragment,
                                    )
                                    if _fragment_buf[0]:
                                        event_bus.emit("new_token", token=_fragment_buf[0])
                                    result_holder.append((result.response_text, None, result.tool_call_log))
                                except Exception as act_err:
                                    result_holder.append((None, act_err, None))

                            act_thread = threading.Thread(target=_run_act, daemon=True)
                            act_thread.start()
                            act_start = time.monotonic()
                            last_heartbeat = act_start
                            while act_thread.is_alive():
                                control = self.controller._check_control_flags(True)
                                if control == "ERROR_CALL":
                                    raise ErrorCallInterrupted("Error call by user")
                                if control in ("STOP", "SKIP"):
                                    raise InterruptedError("Stopped by user")
                                if (time.monotonic() - act_start) >= self.job_timeout_sec:
                                    raise JobTimeoutError(
                                        f"Job exceeded maximum runtime ({self.job_timeout_sec}s). "
                                        f"No completion from {backend.display_name} in time (tool/MCP call may be hung)."
                                    )
                                time.sleep(1.5)
                                if time.monotonic() - last_heartbeat >= 5.0:
                                    last_heartbeat = time.monotonic()
                                    event_bus.emit("new_token", token=" … ")
                            if not result_holder:
                                raise RuntimeError("Tool/MCP act() thread ended without result.")
                            entry = result_holder[0]
                            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                                raise RuntimeError("Tool/MCP act() returned malformed result.")
                            full_response, act_exc, tool_call_log = entry[0], entry[1] if len(entry) > 1 else None, entry[2] if len(entry) > 2 else None
                            if act_exc is not None:
                                exc_str = str(act_exc).lower()
                                if "tool" in exc_str or "vision" in exc_str or "parse" in exc_str:
                                    raise ValueError(f"Model may not support tool use: {act_exc}") from act_exc
                                raise act_exc
                            full_response = full_response or "[No final answer from model]"
                            tokens_gen = max(0, int(len(full_response.split()) * 1.3))
                            stop_reason = "stop"
                        else:
                            # Text: completion stream via backend abstraction
                            response_stream = backend.complete_stream(handle, job_prompt_text, config=self.controller.inference_config)
                            _token_buffer = ""
                            _last_token_emit = 0.0
                            _TOKEN_EMIT_INTERVAL = 0.04
                            for chunk in iterate_stream_with_thermal_checks(
                                self.controller, response_stream, job_timeout_sec=self.job_timeout_sec
                            ):
                                if first_token_time is None:
                                    first_token_time = time.time()
                                content = chunk.content
                                if content:
                                    full_response += content
                                    tokens_gen += 1
                                    _token_buffer += content
                                    now = time.time()
                                    if now - _last_token_emit >= _TOKEN_EMIT_INTERVAL:
                                        if _token_buffer:
                                            event_bus.emit("new_token", token=_token_buffer)
                                            _token_buffer = ""
                                        _last_token_emit = now
                                    if detect_garbage_response(content, full_response):
                                        try:
                                            response_stream.cancel()
                                        except Exception:
                                            pass
                                        raise ValueError("Repetition/garbage loop detected")
                                if chunk.stop_reason:
                                    stop_reason = chunk.stop_reason
                            if _token_buffer:
                                event_bus.emit("new_token", token=_token_buffer)

                        end_time = time.time()
                        self.controller.thermal_monitor.collecting_vitals = False
                            
                        # --- AGGREGATE VITALS ---
                        avg_cpu, avg_gpu, avg_cpu_temp, avg_gpu_temp = 0, 0, 0, 0
                        max_cpu_temp, max_gpu_temp = 0, 0
                        max_vram_gb, max_ram_gb = 0, 0
                            
                        if self.controller.thermal_monitor.vitals_buffer:
                            buf = self.controller.thermal_monitor.vitals_buffer
                            count = len(buf)
                            if count > 0:
                                avg_cpu = sum(d['cpu_util'] for d in buf) / count
                                avg_gpu = sum(d['gpu_util'] for d in buf) / count
                                avg_cpu_temp = sum(d['cpu_temp'] for d in buf) / count
                                avg_gpu_temp = sum(d['gpu_temp'] for d in buf) / count
                                max_cpu_temp = max(d['cpu_temp'] for d in buf)
                                max_gpu_temp = max(d['gpu_temp'] for d in buf)
                                max_vram_gb = max(d.get('vram_used', 0) for d in buf)
                                max_ram_gb = max(d.get('ram_used', 0) for d in buf)
                        else:
                            # Fallback if buffer empty (short run)
                            max_cpu_temp = memory_utils.get_cpu_temperature()
                            _, max_vram_gb, _, max_gpu_temp = memory_utils.get_gpu_vitals()
                            sys_t, sys_a = memory_utils.get_system_memory()
                            max_ram_gb = sys_t - sys_a
                            
                        thermal_count = self.controller.thermal_monitor.throttle_events
                        power_count = self.controller.thermal_monitor.power_throttle_events

                        # --- CALCULATION ---
                        if not full_response or not full_response.strip():
                            raise ValueError("Empty Response Generated")
                        else:
                            total_time = round(end_time - start_time, 4)
                            effective_start = first_token_time if first_token_time else start_time
                            gen_time = round(end_time - effective_start, 4)
                            ttft = round(effective_start - start_time, 4)
                            total_tps = round(tokens_gen / total_time, 2) if total_time > 0 else 0
                            streaming_tps = round(tokens_gen / gen_time, 2) if gen_time > 0 else 0
                                
                            scores = {}
                            prompt_data = prompt_map.get(job_prompt_id)
                            bt = (prompt_data.get('benchmark_type') or 'text').strip() if prompt_data else 'text'
                            ref_metric = (prompt_data.get('reference_metric') or '').strip().lower() if prompt_data else ''
                            if bt == 'standardized' and prompt_data and prompt_data.get('expected_response') and ref_metric in ('bleu', 'rouge', 'bert'):
                                ref_text = (prompt_data['expected_response'] or '').strip()
                                if ref_text:
                                    self.controller._log(f"   -> Calculating {ref_metric.upper()} (Standardized Benchmark)...")
                                    # Strip thinking blocks so only the final answer is scored (o1, DeepSeek R1, etc.)
                                    response_for_scoring = strip_thinking_blocks(full_response)
                                    scores = self.scoring_engine.calculate_scores(response_for_scoring, ref_text, metrics=[ref_metric])
                                
                            payload = {
                                'response_text': full_response,
                                'tokens_generated': tokens_gen,
                                'stop_reason': stop_reason,
                                'total_time_sec': total_time,
                                'time_to_first_token_sec': ttft,
                                'generation_time_sec': gen_time,
                                'streaming_tps': streaming_tps,
                                'total_tps': total_tps,
                                'context_length': self.controller.inference_config['max_tokens'],
                                'avg_cpu_usage': round(avg_cpu, 1),
                                'avg_gpu_usage': round(avg_gpu, 1),
                                'avg_cpu_temp': round(avg_cpu_temp, 1),
                                'avg_gpu_temp': round(avg_gpu_temp, 1),
                                'max_cpu_temp': round(max_cpu_temp, 1),
                                'max_gpu_temp': round(max_gpu_temp, 1),
                                'peak_vram_usage_gb': round(max_vram_gb, 2),
                                'peak_ram_usage_gb': round(max_ram_gb, 2),
                                'thermal_throttling_count': thermal_count,
                                'power_throttling_count': power_count
                            }
                            if tool_call_log is not None:
                                payload['tool_call_log'] = tool_call_log if isinstance(tool_call_log, str) else json.dumps(tool_call_log)
                            payload.update(scores)
                                
                            self.db.add_response(job_llm_id, job_prompt_id, payload, system_id=current_system_id)
                                
                            log_msg = f"   -> Saved. TPS: {streaming_tps}"
                            if thermal_count > 0: log_msg += f" [Therm x{thermal_count}]"
                            if power_count > 0: log_msg += f" [Pwr x{power_count}]"
                            self.controller._log(log_msg)
                                
                            self.consecutive_failures = 0
                            event_bus.emit("prompt_complete", prompt_id=job_prompt_id, stop_reason=stop_reason)
                                
                            # Pause if expanded output window is open, wait for user to close it
                            if self.controller.global_control.output_view_paused:
                                self.controller._log("Waiting for expanded output window to close...")
                                self.controller.global_control.request_pause(source="output_view")
                                # Wait until window is closed
                                while self.controller.global_control.output_view_paused:
                                    if self.controller._check_control_flags() in ("STOP", "SKIP"):
                                        break
                                    time.sleep(0.1)
                                # Resume after window closes (if not stopped/skipped)
                                if not self.controller.global_control.stop_event.is_set() and not self.controller.global_control.skip_event.is_set():
                                    self.controller.global_control.request_resume(source="output_view")
                                
                            break 

                    except ErrorCallInterrupted:
                        # Manual error call: save response to this point with stop_reason=Error
                        self.controller.global_control.reset_error_call()
                        response_text = full_response if full_response else "(Manual error call - no response captured)"
                        end_time = time.time()
                        total_time = round(end_time - start_time, 4)
                        effective_start = first_token_time if first_token_time else start_time
                        gen_time = round(end_time - effective_start, 4)
                        ttft = round(effective_start - start_time, 4)
                        total_tps = round(tokens_gen / total_time, 2) if total_time > 0 else 0
                        streaming_tps = round(tokens_gen / gen_time, 2) if gen_time > 0 else 0
                        payload = {
                            'response_text': response_text,
                            'tokens_generated': tokens_gen,
                            'stop_reason': 'Error',
                            'total_time_sec': total_time,
                            'time_to_first_token_sec': ttft,
                            'generation_time_sec': gen_time,
                            'streaming_tps': streaming_tps,
                            'total_tps': total_tps,
                            'context_length': self.controller.inference_config['max_tokens'],
                        }
                        if self.controller.thermal_monitor.vitals_buffer:
                            buf = self.controller.thermal_monitor.vitals_buffer
                            count = len(buf)
                            if count > 0:
                                payload['avg_cpu_usage'] = round(sum(d['cpu_util'] for d in buf) / count, 1)
                                payload['avg_gpu_usage'] = round(sum(d['gpu_util'] for d in buf) / count, 1)
                                payload['avg_cpu_temp'] = round(sum(d['cpu_temp'] for d in buf) / count, 1)
                                payload['avg_gpu_temp'] = round(sum(d['gpu_temp'] for d in buf) / count, 1)
                                payload['max_cpu_temp'] = round(max(d['cpu_temp'] for d in buf), 1)
                                payload['max_gpu_temp'] = round(max(d['gpu_temp'] for d in buf), 1)
                                payload['peak_vram_usage_gb'] = round(max(d.get('vram_used', 0) for d in buf), 2)
                                payload['peak_ram_usage_gb'] = round(max(d.get('ram_used', 0) for d in buf), 2)
                        else:
                            max_cpu_temp = memory_utils.get_cpu_temperature()
                            _, max_vram_gb, _, max_gpu_temp = memory_utils.get_gpu_vitals()
                            sys_t, sys_a = memory_utils.get_system_memory()
                            max_ram_gb = sys_t - sys_a
                            payload['max_cpu_temp'] = round(max_cpu_temp, 1)
                            payload['max_gpu_temp'] = round(max_gpu_temp, 1)
                            payload['peak_vram_usage_gb'] = round(max_vram_gb, 2)
                            payload['peak_ram_usage_gb'] = round(max_ram_gb, 2)
                        self.controller.thermal_monitor.collecting_vitals = False
                        self.db.add_response(job_llm_id, job_prompt_id, payload, system_id=current_system_id)
                        self.controller._log("   -> Error Call: Saved partial response as Error.")
                        event_bus.emit("prompt_complete", prompt_id=job_prompt_id, stop_reason='Error')
                        break

                    except InterruptedError:
                        control_sig = self.controller._check_control_flags(True)
                        if control_sig == "SOFT_STOP":
                            if handle:
                                self.controller._log("Soft-stop: Unloading model...")
                                backend.unload_model(handle)
                                handle = None
                                gc.collect()
                            llm_loop_stop = True
                            event_bus.emit("prompt_complete", prompt_id=job_prompt_id, stop_reason="SOFT_STOPPED")
                        else:
                            llm_loop_stop = (control_sig == "STOP")
                            llm_was_skipped = (control_sig == "SKIP")
                            event_bus.emit("prompt_complete", prompt_id=job_prompt_id, stop_reason="SKIPPED" if llm_was_skipped else "STOPPED")
                        break

                    except JobTimeoutError as e:
                        self.controller._log(f"   -> Job Failed: {e}")
                        self.db.add_response(
                            job_llm_id, job_prompt_id,
                            {'response_text': f"Error: {e}", 'stop_reason': 'Error_Timeout'},
                            system_id=current_system_id
                        )
                        self.consecutive_failures += 1
                        event_bus.emit("prompt_complete", prompt_id=job_prompt_id, stop_reason='Error')
                        break

                    except Exception as e:
                        # CRASH HANDLING
                        error_str = str(e)
                        is_critical = self._is_critical_error(error_str)
                        if "Gibberish" in error_str or "Repetition" in error_str: is_critical = True

                        if is_critical and attempt == 0:
                            self.controller._log(f"[CRASH DETECTED] {e}")
                            self.controller._log("   -> Attempting Auto-Reload...")
                            backend.unload_model(handle, stream=response_stream)
                            handle = None; gc.collect(); time.sleep(2)
                            try:
                                reload_config = {'gpu_layers': -1, 'context_length': self.controller.context_length}
                                if self.controller.global_control.load_as_new_instance and backend.supports_load_alongside():
                                    handle = backend.load_model_alongside(current_llm_identifier, reload_config)
                                else:
                                    handle = backend.load_model(current_llm_identifier, reload_config)
                                self.controller._log("   -> Model Reloaded. Retrying Job...")
                                continue
                            except Exception as reload_err:
                                self.controller._log(f"   -> Reload Failed: {reload_err}")
                                final_msg = f"Crash & Reload Failed: {e}"
                                self.db.add_response(job_llm_id, job_prompt_id, {'response_text': final_msg, 'stop_reason': 'Error_Crash'}, system_id=current_system_id)
                                self.consecutive_failures += 1
                                event_bus.emit("prompt_complete", prompt_id=job_prompt_id, stop_reason='Error')
                                break
                        else:
                            self.controller._log(f"   -> Job Failed: {e}")
                            err_msg = str(e).lower()
                            stop = "Error"
                            if "vision" in err_msg or "image" in err_msg:
                                stop = "Error_Vision"
                            elif "tool" in err_msg:
                                stop = "Error_Tool"
                            elif "exceeds the available context size" in err_msg or ("context size" in err_msg and "exceed" in err_msg):
                                stop = "Error_Context"
                            elif "model unloaded" in err_msg or "no model loaded" in err_msg:
                                stop = "Error_ModelUnloaded"
                            self.db.add_response(job_llm_id, job_prompt_id, {'response_text': f"Error: {e}", 'stop_reason': stop}, system_id=current_system_id)
                            self.consecutive_failures += 1
                            event_bus.emit("prompt_complete", prompt_id=job_prompt_id, stop_reason='Error')
                            break
                                
                    finally:
                        # Close stream if it exists (text/vision only; tool_use/mcp uses act(), no stream)
                        if response_stream is not None:
                            try:
                                response_stream.cancel()
                            except Exception:
                                pass
                            try:
                                response_stream.close()
                                time.sleep(0.2)
                            except Exception:
                                pass
                            
                if llm_loop_stop or llm_was_skipped: break
                
            # Pause if expanded output window is open before unloading and moving to next LLM
            if self.controller.global_control.output_view_paused:
                self.controller._log("Waiting for expanded output window to close before unloading model...")
                self.controller.global_control.request_pause(source="output_view")
                while self.controller.global_control.output_view_paused:
                    if self.controller._check_control_flags() in ("STOP", "SKIP"):
                        break
                    time.sleep(0.1)
                if not self.controller.global_control.stop_event.is_set() and not self.controller.global_control.skip_event.is_set():
                    self.controller.global_control.request_resume(source="output_view")
                
            # Safely unload the model via backend
            backend.unload_model(handle)
            handle = None
            gc.collect() 
            self.controller.thermal_monitor.cooldown_wait()
                    
            final_color = ft.Colors.GREEN_800
            if llm_was_skipped: final_color = ft.Colors.GREY_500
            elif llm_loop_stop: final_color = ft.Colors.YELLOW_700
            elif self.consecutive_failures >= 3: final_color = ft.Colors.RED_900
                
            event_bus.emit("llm_complete", llm_id=llm_id, color=final_color)
            if llm_was_skipped: self.controller.skip_llm_event.clear()
            if llm_loop_stop: break