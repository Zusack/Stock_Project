import re
import gc
import time
import json
import flet as ft
from src.services.global_control_service import GlobalControlService
from src.utils.stream_utils import iterate_stream_with_thermal_checks, ErrorCallInterrupted
from src.utils.repetition_utils import detect_garbage_response
from src.services.event_bus import event_bus
from src.backends.types import LLMHandle

DEFAULT_EVAL_CONFIG = {
    "temperature": 0.1,       
    "max_tokens": 4096
}

class EvaluationEngine:
    def __init__(self, controller_ref):
        self.controller = controller_ref
        self.db = controller_ref.db
        self.global_control = GlobalControlService()
        # No self.consecutive_failures here to prevent state leakage

    def _is_critical_error(self, error_msg):
        msg = str(error_msg).lower()
        triggers = [
            "model has crashed", "exit code", "no model found", 
            "connection refused", "channel closed", "unexpected unload", 
            "failed to respond", "broken pipe"
        ]
        return any(t in msg for t in triggers)

    def run(self):
        current_sys_id = self.global_control.active_system_id
        if not current_sys_id:
            self.controller._log("[ERROR] No active system selected.")
            return

        self.controller._log(f"Starting Evaluation for System ID: {current_sys_id}")

        # Ensure incompatible responses exist and have auto-rated-0 evaluations (skipped by evaluators)
        self.db.ensure_incompatible_responses_recorded(system_id=current_sys_id)
        self.db.ensure_incompatible_evaluations_recorded(system_id=current_sys_id)

        # Only Judge-selected models (is_evaluator) run evaluations; all respondents are still scored by those judges.
        # Excludes incompatible responses — those are auto-rated 0 and never sent to evaluators.
        eval_queue = self.db.get_missing_evaluations(system_id=current_sys_id, include_non_evaluators=False)
        if not eval_queue:
            try:
                cursor = self.db.connection.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM llm_system_metrics
                    WHERE system_id = ? AND is_evaluator = 1 AND is_available = 1
                """, (current_sys_id,))
                if cursor.fetchone()[0] == 0:
                    self.controller._log(
                        "[INFO] No Judge models selected in Model Manager. Select at least one Judge to run evaluations."
                    )
                else:
                    self.controller._log("No missing evaluations found.")
            except Exception:
                self.controller._log("No missing evaluations found.")
            return

        self.controller._log(f"Found {len(eval_queue)} pending evaluation tasks (evaluator panel).")

        # Cascade limit: stop after N consecutive load/probe failures to prevent infinite retry loops
        CASCADE_LOAD_FAIL_LIMIT = 3
        consecutive_load_failures = 0

        # Organize jobs by prompt category, then by evaluator, then by respondent
        # Category-first: complete all evaluators for one category before moving to the next
        category_jobs = {}
        for job in eval_queue:
            cat_group = job[10] if len(job) > 10 else 'Ungrouped'
            ev_id = job[1]
            respondent_id = job[2]
            if cat_group not in category_jobs:
                category_jobs[cat_group] = {}
            if ev_id not in category_jobs[cat_group]:
                category_jobs[cat_group][ev_id] = {}
            if respondent_id not in category_jobs[cat_group][ev_id]:
                category_jobs[cat_group][ev_id][respondent_id] = []
            category_jobs[cat_group][ev_id][respondent_id].append(job)

        # Sort categories: Ungrouped last, others alphabetically
        sorted_categories = sorted(category_jobs.keys(), key=lambda c: (c == "Ungrouped", c))
        category_loop_stop = False

        for category in sorted_categories:
            if category_loop_stop:
                break
            control_signal = self.controller._check_control_flags()
            if control_signal in ("STOP", "SOFT_STOP"):
                self.controller._log("Stop detected before category. Exiting.")
                break

            self.controller._log(f"Processing category: {category}")
            evaluator_jobs = category_jobs[category]
            category_missing = sum(len(jobs) for ev_id, resp_dict in evaluator_jobs.items() for resp_id, jobs in resp_dict.items())
            category_existing = self.db.get_category_evaluation_existing_count(current_sys_id, category, include_non_evaluators=False)
            category_total = category_existing + category_missing
            category_completed = category_existing
            event_bus.emit("category_started", category=category, total=category_total, completed=category_completed)

            for evaluator_llm_id, respondent_jobs_dict in evaluator_jobs.items():
                # Check stop BEFORE each evaluator - prevents cascading past stop request
                control_signal = self.controller._check_control_flags()
                if control_signal in ("STOP", "SOFT_STOP"):
                    self.controller._log("Stop detected between evaluators. Exiting.")
                    category_loop_stop = True
                    break

                # Backend for this evaluator (from Model Manager / llms.backend_type)
                backend_type = self.db.get_backend_type(evaluator_llm_id)
                backend = self.controller.get_backend(backend_type)

                # Cascade limit: stop after too many consecutive load failures
                if consecutive_load_failures >= CASCADE_LOAD_FAIL_LIMIT:
                    if self.global_control.load_as_new_instance:
                        self.controller._log(
                            f"[STOP] {consecutive_load_failures} consecutive evaluator load failures. "
                            "Could not load evaluator model(s) (e.g. insufficient memory). Unload other models or reduce context and retry."
                        )
                        category_loop_stop = True
                        break
                    msg = (
                        f"[FATAL] {consecutive_load_failures} consecutive evaluator load failures. "
                        f"Likely a model is already loaded in {backend.display_name}. Unload it and retry."
                    )
                    self.controller._log(msg)
                    raise RuntimeError(backend.model_already_loaded_message) from None

                # Get evaluator identifier from first job of first respondent
                first_respondent_id = next(iter(respondent_jobs_dict))
                current_evaluator_identifier = respondent_jobs_dict[first_respondent_id][0][3]

                event_bus.emit("llm_active", llm_id=evaluator_llm_id)
                self.controller._log(f"Preparing Evaluator: {current_evaluator_identifier}")
                
                consecutive_respondents_with_3_error_event = 0 
                
                # --- PHASE 1: PROBE & CALCULATE CONTEXT ---
                max_input_tokens = 0
                max_benchmark_context = 0
                probe_handle: LLMHandle | None = None
                required_context = 4096 
                
                try:
                    self.controller._log("   -> Phase 1: Probing token requirements...")
                    probe_config = {'gpu_layers': -1, 'context_length': 1024}
                    if self.global_control.load_as_new_instance and backend.supports_load_alongside():
                        self.controller._log("   -> Loading as new instance (alongside existing model)...")
                        probe_handle = backend.load_model_alongside(current_evaluator_identifier, probe_config)
                    else:
                        probe_handle = backend.load_model(current_evaluator_identifier, probe_config)
                    
                    for respondent_id, jobs in respondent_jobs_dict.items():
                        for job in jobs:
                            (response_id, _, _, _, response_text, _, full_rubric_md, original_prompt_text) = job[:8]
                            scoring_rubric = self._extract_rubric_section(full_rubric_md)
                            
                            sys_p, user_p = self._build_structured_eval_prompts(original_prompt_text, response_text, scoring_rubric)
                            full_payload = f"{sys_p}\n\n---\n\n{user_p}"
                            
                            token_count = len(backend.tokenize(probe_handle, full_payload))
                            if token_count > max_input_tokens: max_input_tokens = token_count
                                
                            try:
                                cur = self.db.connection.cursor()
                                cur.execute("SELECT context_length FROM responses WHERE response_id = ?", (response_id,))
                                row = cur.fetchone()
                                bench_ctx = row[0] if row and row[0] else 2048 
                                if bench_ctx > max_benchmark_context: max_benchmark_context = bench_ctx
                            except Exception: pass

                    backend.unload_model(probe_handle); probe_handle = None; gc.collect()
                    
                    required_context = max_input_tokens + max_benchmark_context + 4096 
                    self.controller._log(f"   -> Calculated Context: {required_context}")

                except Exception as e:
                    if probe_handle:
                        backend.unload_model(probe_handle)
                    if backend.is_unreachable_error(e):
                        raise RuntimeError(backend.unreachable_message) from e
                    if backend.is_load_blocked_error(e):
                        consecutive_load_failures += 1
                    self.controller._log(f"[FATAL] Probe failed: {e}")
                    self.db.log_error(current_sys_id, evaluator_llm_id, "Evaluator Probe", str(e))
                    event_bus.emit("llm_complete", llm_id=evaluator_llm_id, color=ft.Colors.RED_800)
                    continue

                # --- PHASE 2: EXECUTION ---
                handle: LLMHandle | None = None
                llm_was_skipped = False
                llm_loop_stop = False
                
                try:
                    self.controller._log(f"   -> Phase 2: Loading with Context {required_context}...")
                    load_config = {'gpu_layers': -1, 'context_length': required_context}
                    if self.global_control.load_as_new_instance and backend.supports_load_alongside():
                        handle = backend.load_model_alongside(current_evaluator_identifier, load_config)
                    else:
                        handle = backend.load_model(current_evaluator_identifier, load_config)
                    consecutive_load_failures = 0
                except Exception as e:
                    if backend.is_unreachable_error(e):
                        raise RuntimeError(backend.unreachable_message) from e
                    if backend.is_load_blocked_error(e):
                        consecutive_load_failures += 1
                    msg = f"[FATAL] Context Load Failed: {e}"
                    self.controller._log(msg)
                    self.db.log_error(current_sys_id, evaluator_llm_id, "Evaluator Load", msg)
                    for respondent_id, jobs in respondent_jobs_dict.items():
                        for job in jobs:
                            self.db.add_evaluation(job[0], evaluator_llm_id, {'rating': -1, 'rationale': f"Evaluator Hardware Failure: {msg}", 'prompt_category': job[5]})
                            category_completed += 1
                            event_bus.emit("category_progress", category=category, completed=category_completed, total=category_total)
                    event_bus.emit("llm_complete", llm_id=evaluator_llm_id, color=ft.Colors.RED_800)
                    continue

                # Iterate through respondents
                for respondent_id, jobs in respondent_jobs_dict.items():
                    respondent_identifier = jobs[0][2]  # Get from first job
                    
                    # Check if we should skip all remaining work for this evaluator
                    if consecutive_respondents_with_3_error_event >= 3:
                        msg = f"Skipped: 3 Consecutive Respondents with 3-Error Events for {current_evaluator_identifier}."
                        self.controller._log(f"[STOP] {msg}")
                        # Mark all remaining jobs as skipped
                        for job in jobs:
                            response_id = job[0]
                            prompt_category = job[5]
                            self._handle_error(current_sys_id, evaluator_llm_id, response_id, prompt_category, msg, "Evaluator Skipped")
                            category_completed += 1
                            event_bus.emit("category_progress", category=category, completed=category_completed, total=category_total)
                        continue
                    
                    self.controller._log(f"-> Processing Respondent: {respondent_identifier}")
                    consecutive_failures_for_current_respondent = 0
                    respondent_had_3_error_event = False

                    for i, job in enumerate(jobs):
                        (response_id, _, _, _, response_text, prompt_category, full_rubric_md, original_prompt_text) = job[:8]

                        # Check if we should skip remaining prompts for this respondent
                        if consecutive_failures_for_current_respondent >= 3:
                            msg = f"Skipped: 3 Consecutive Failures for {respondent_identifier} on {current_evaluator_identifier}."
                            self.controller._log(f"[SKIP RESPONDENT] {msg}")
                            self._handle_error(current_sys_id, evaluator_llm_id, response_id, prompt_category, msg, "Respondent Skipped")
                            category_completed += 1
                            event_bus.emit("category_progress", category=category, completed=category_completed, total=category_total)
                            respondent_had_3_error_event = True
                            continue

                        evaluator_llm_id = job[1]
                        respondent_llm_id = job[8] if len(job) > 8 else None
                        prompt_id = job[9] if len(job) > 9 else None
                        event_bus.emit(
                            "evaluation_step",
                            evaluator_identifier=current_evaluator_identifier,
                            respondent_identifier=respondent_identifier,
                            prompt_category=prompt_category,
                            evaluator_llm_id=evaluator_llm_id,
                            respondent_llm_id=respondent_llm_id,
                            prompt_id=prompt_id,
                        )

                        control_signal = self.controller._check_control_flags()
                        if control_signal == "STOP": 
                            llm_loop_stop = True; break
                        if control_signal == "SOFT_STOP":
                            if handle:
                                self.controller._log("Soft-stop: Unloading model...")
                                backend.unload_model(handle)
                                handle = None
                                gc.collect()
                            llm_loop_stop = True; break
                        if control_signal == "SKIP": 
                            llm_was_skipped = True; break

                        self.controller.thermal_monitor.wait_for_safe_start_temp()
                        
                        control_signal = self.controller._check_control_flags()
                        if control_signal == "STOP": 
                            llm_loop_stop = True; break
                        if control_signal == "SOFT_STOP":
                            if handle:
                                self.controller._log("Soft-stop: Unloading model...")
                                backend.unload_model(handle)
                                handle = None
                                gc.collect()
                            llm_loop_stop = True; break
                        if control_signal == "SKIP": 
                            llm_was_skipped = True; break

                        self.controller._log(f"-> Evaluating {respondent_identifier} on {prompt_category}")
                        event_bus.emit("new_token", token=None)

                        tool_call_log = None
                        try:
                            cur = self.db.connection.cursor()
                            cur.execute("SELECT tool_call_log FROM responses WHERE response_id = ?", (response_id,))
                            row = cur.fetchone()
                            tool_call_log = row['tool_call_log'] if row else None
                        except Exception:
                            pass

                        # --- RETRY LOOP ---
                        for attempt in range(2):
                            response_stream = None
                            full_response = ""
                            stop_reason = None
                                
                            try:
                                scoring_rubric = self._extract_rubric_section(full_rubric_md)
                                sys_p, user_p = self._build_structured_eval_prompts(original_prompt_text, response_text, scoring_rubric, tool_call_log=tool_call_log)
                                
                                combined_prompt = f"{sys_p}\n\n---\n\n{user_p}"
                                messages = [{"role": "user", "content": combined_prompt}]
                                
                                eval_start = time.time()
                                response_stream = backend.chat_stream(handle, messages, config=DEFAULT_EVAL_CONFIG)
                                
                                _token_buffer = ""
                                _last_token_emit = 0.0
                                _TOKEN_EMIT_INTERVAL = 0.04
                                for chunk in iterate_stream_with_thermal_checks(self.controller, response_stream):
                                    if chunk.stop_reason:
                                        stop_reason = chunk.stop_reason
                                    content = chunk.content
                                    if content:
                                        full_response += content
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
                                            raise ValueError("Evaluator Error: Gibberish/Repetition Loop Detected")

                                if _token_buffer:
                                    event_bus.emit("new_token", token=_token_buffer)
                                if not full_response or not full_response.strip():
                                    raise ValueError("Empty Response")
                                    
                                parsed = self._parse_structured_response(full_response)
                                if parsed:
                                    eval_duration = round(time.time() - eval_start, 4)
                                    self.db.add_evaluation(response_id, evaluator_llm_id, {
                                        'rating': parsed['rating'], 'rationale': parsed['rationale'], 'prompt_category': prompt_category, 'raw_response': full_response,
                                        'evaluation_time_sec': eval_duration,
                                    })
                                    self.controller._log(f"   -> Saved Score: {parsed['rating']}")
                                    category_completed += 1
                                    event_bus.emit("category_progress", category=category, completed=category_completed, total=category_total)
                                    event_bus.emit("evaluation_complete")
                                    
                                    # Pause if expanded output window is open, wait for user to close it
                                    if self.global_control.output_view_paused:
                                        self.controller._log("Waiting for expanded output window to close...")
                                        self.global_control.request_pause(source="output_view")
                                        # Wait until window is closed
                                        while self.global_control.output_view_paused:
                                            if self.controller._check_control_flags() in ("STOP", "SKIP"):
                                                break
                                            time.sleep(0.1)
                                        # Resume after window closes (if not stopped/skipped)
                                        if not self.global_control.stop_event.is_set() and not self.global_control.skip_event.is_set():
                                            self.global_control.request_resume(source="output_view")
                                    
                                    consecutive_failures_for_current_respondent = 0
                                    break 
                                else:
                                    if stop_reason == "length":
                                        msg = f"Evaluator Error: Response Truncated (Max Tokens {DEFAULT_EVAL_CONFIG['max_tokens']} reached)."
                                        raise ValueError(msg)
                                    else:
                                        # Do NOT include full_response in the exception: it would be dumped to the
                                        # system log. raw_response is passed via _handle_error and stored in DB only.
                                        raise ValueError(f"JSON Parse Error (Reason: {stop_reason})")

                            except ErrorCallInterrupted:
                                # Manual error call: save rationale to this point with rating=-1
                                self.global_control.reset_error_call()
                                rationale = full_response if full_response else "(Manual error call - no rationale captured)"
                                self.db.add_evaluation(response_id, evaluator_llm_id, {
                                    'rating': -1,
                                    'rationale': rationale,
                                    'prompt_category': prompt_category,
                                    'raw_response': full_response,
                                })
                                self.controller._log("   -> Error Call: Saved partial evaluation as Error.")
                                category_completed += 1
                                event_bus.emit("category_progress", category=category, completed=category_completed, total=category_total)
                                event_bus.emit("evaluation_complete")
                                consecutive_failures_for_current_respondent = 0
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
                                else:
                                    llm_loop_stop = (control_sig == "STOP")
                                    llm_was_skipped = (control_sig == "SKIP")
                                break 

                            except Exception as e:
                                error_str = str(e)
                                is_critical = self._is_critical_error(error_str) or "Gibberish" in error_str
                                
                                # Raw response for DB/Inspector only (never logged in full to system log)
                                raw_resp = full_response if ('full_response' in locals() and full_response) else None

                                if is_critical and attempt == 0:
                                    self.controller._log(f"[CRASH DETECTED] {e} -> Reloading...")
                                    try:
                                        backend.unload_model(handle, stream=None)
                                    except Exception:
                                        pass
                                    handle = None; gc.collect(); time.sleep(2)
                                    try:
                                        reload_config = {'gpu_layers': -1, 'context_length': required_context}
                                        if self.global_control.load_as_new_instance and backend.supports_load_alongside():
                                            handle = backend.load_model_alongside(current_evaluator_identifier, reload_config)
                                        else:
                                            handle = backend.load_model(current_evaluator_identifier, reload_config)
                                        continue 
                                    except Exception as reload_err:
                                        self._handle_error(current_sys_id, evaluator_llm_id, response_id, prompt_category, f"Crash & Reload Failed: {e}", "Evaluator Gen", raw_response=raw_resp)
                                        category_completed += 1
                                        event_bus.emit("category_progress", category=category, completed=category_completed, total=category_total)
                                        consecutive_failures_for_current_respondent += 1
                                        break
                                else:
                                    self.controller._log(f"   -> Job Failed: {e}")
                                    self._handle_error(current_sys_id, evaluator_llm_id, response_id, prompt_category, f"Evaluator Error: {e}", "Evaluator Gen", raw_response=raw_resp)
                                    category_completed += 1
                                    event_bus.emit("category_progress", category=category, completed=category_completed, total=category_total)
                                    consecutive_failures_for_current_respondent += 1
                                    break
                            finally:
                                if response_stream:
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
                    
                    # After processing all prompts for this respondent, update the 3-error-event counter
                    if respondent_had_3_error_event:
                        consecutive_respondents_with_3_error_event += 1
                        self.controller._log(f"   -> 3-Error Event occurred for {respondent_identifier}. Counter: {consecutive_respondents_with_3_error_event}/3")
                    else:
                        # Reset the counter if this respondent completed without a 3-error event
                        if consecutive_respondents_with_3_error_event > 0:
                            self.controller._log(f"   -> {respondent_identifier} completed without 3-error event. Resetting counter.")
                        consecutive_respondents_with_3_error_event = 0
                    
                    if llm_loop_stop or llm_was_skipped: break
                
                # Pause if expanded output window is open before unloading and moving to next evaluator
                if self.global_control.output_view_paused:
                    self.controller._log("Waiting for expanded output window to close before unloading model...")
                    self.global_control.request_pause(source="output_view")
                    while self.global_control.output_view_paused:
                        if self.controller._check_control_flags() in ("STOP", "SKIP"):
                            break
                        time.sleep(0.1)
                    if not self.global_control.stop_event.is_set() and not self.global_control.skip_event.is_set():
                        self.global_control.request_resume(source="output_view")
                
                # Safely unload the model via backend
                if handle:
                    backend.unload_model(handle)
                    handle = None
                gc.collect(); self.controller.thermal_monitor.cooldown_wait()
                        
                final_color = ft.Colors.GREEN_800
                if llm_was_skipped: final_color = ft.Colors.GREY_500
                elif llm_loop_stop: final_color = ft.Colors.YELLOW_700
                elif consecutive_respondents_with_3_error_event >= 3: final_color = ft.Colors.RED_900
                
                event_bus.emit("llm_complete", llm_id=evaluator_llm_id, color=final_color)
                if llm_was_skipped: self.controller.skip_llm_event.clear()
                if llm_loop_stop:
                    category_loop_stop = True
                    break

            if category_loop_stop:
                break
            event_bus.emit("category_complete", category=category)
            self.controller._log(f"Category {category} complete – all evaluators finished.")

    def _extract_rubric_section(self, full_rubric_md):
        scoring_rubric = ""
        try:
            parts = re.split(r"^##\s+Prompt:", full_rubric_md, maxsplit=1, flags=re.MULTILINE)
            if len(parts) > 1:
                next_section_match = re.search(r"\n##\s+(?:Correct|Expected)\s+Response:", parts[1], re.MULTILINE)
                if next_section_match: scoring_rubric = parts[1][next_section_match.start():].strip()
        except: pass
        return scoring_rubric if scoring_rubric else full_rubric_md

    def _build_structured_eval_prompts(self, original_prompt, response_to_evaluate, scoring_rubric, tool_call_log=None):
        system_prompt = (
            "You are an expert AI Quality Assurance Judge. Your task is to evaluate an LLM response against a rubric. "
            "You must output your evaluation in strict JSON format.\n\nJSON Schema:\n{\n  \"rationale\": \"string (detailed reasoning)\",\n  \"rating\": integer (0-10)\n}"
        )
        user_prompt = (
            f"### 1. The Prompt Sent to the Model\n{original_prompt}\n\n### 2. Scoring Rubric & Reference\n{scoring_rubric}\n\n"
            f"### 3. The Model's Response (Target)\n```text\n{response_to_evaluate}\n```\n\n"
        )
        if tool_call_log and str(tool_call_log).strip():
            user_prompt += f"The model had access to tools. Tool call log (if any):\n```\n{tool_call_log}\n```\n\n"
        user_prompt += (
            "---\n\n"
            "Evaluate the response. First, think step-by-step in the rationale field. Then, assign a final integer score (0-10) in the rating field based on the rubric.\n"
            "Output ONLY the JSON object."
        )
        return system_prompt, user_prompt

    def _parse_structured_response(self, text):
        """
        Robustly parse evaluation responses that may contain JSON in various formats.
        Handles: code blocks (```json), triple quotes ('''), plain JSON, and alternative field names.
        Also handles cases where rationale might be outside the JSON block.
        Handles special tokens and formatting that some models output (e.g., <|channel|>, <|message|>).
        """
        if not text or not text.strip():
            return None
        
        # Preprocess: clean special tokens and formatting that might interfere with JSON parsing
        clean_text = self._clean_response_text(text.strip())
        
        # Strategy 1: Try to extract JSON from code blocks (```json or ```)
        # Handle both ```json and ``` formats, and also '''json and ''' formats
        # Use balanced brace matching to get complete JSON objects
        code_block_markers = [
            (r"```(?:json)?\s*", r"\s*```"),  # Standard code blocks
            (r"'''\s*", r"\s*'''"),  # Triple single quotes
            (r'"""\s*', r'\s*"""'),  # Triple double quotes
        ]
        
        for start_pattern, end_pattern in code_block_markers:
            start_match = re.search(start_pattern, clean_text, re.IGNORECASE)
            if start_match:
                start_pos = start_match.end()
                # Find the matching end marker
                end_match = re.search(end_pattern, clean_text[start_pos:], re.IGNORECASE)
                if end_match:
                    block_content = clean_text[start_pos:start_pos + end_match.start()].strip()
                    # Extract JSON objects from the block content
                    json_candidates = self._extract_json_objects(block_content)
                    for json_str in json_candidates:
                        # Try parsing with full text context in case rationale is outside JSON
                        result = self._try_parse_json_with_context(json_str, clean_text)
                        if result:
                            return result
                    # If no complete JSON found, try parsing the whole block
                    result = self._try_parse_json_with_context(block_content, clean_text)
                    if result:
                        return result
        
        # Strategy 2: Find the first complete JSON object in the text
        # Use a more robust approach: find balanced braces
        json_candidates = self._extract_json_objects(clean_text)
        for json_str in json_candidates:
            result = self._try_parse_json_with_context(json_str, clean_text)
            if result:
                return result
        
        # Strategy 3: Fallback regex extraction for rationale and rating/score
        # This is a last resort for malformed JSON
        result = self._extract_fields_with_regex(clean_text)
        if result:
            return result
        
        return None
    
    def _clean_response_text(self, text):
        """
        Clean response text by removing special tokens and formatting that might interfere with JSON parsing.
        Handles tokens like <|channel|>, <|message|>, <|end|>, <|start|>, etc.
        Preserves word boundaries and text structure for rationale extraction.
        """
        # Remove special token patterns (e.g., <|channel|>analysis<|message|>)
        # These are often used by some models for structured output
        # Add spaces around removed tokens to preserve word boundaries
        text = re.sub(r'<\|[^|]+\|>', ' ', text)
        
        # Remove XML-like tags that might be used for formatting
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Clean up excessive whitespace but preserve structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines -> double newline
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces -> single space
        
        # Remove leading/trailing whitespace from each line (but keep newlines)
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _extract_json_objects(self, text):
        """
        Extract complete JSON objects by finding balanced braces.
        Returns a list of potential JSON strings, ordered by length (longest first).
        """
        candidates = []
        depth = 0
        start = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    # Found a complete JSON object
                    json_str = text[start:i+1]
                    candidates.append(json_str)
                    start = -1
        
        # Sort by length (descending) to try the most complete objects first
        candidates.sort(key=len, reverse=True)
        return candidates
    
    def _try_parse_json(self, json_str):
        """
        Try to parse a JSON string and validate it.
        Returns validated result or None.
        """
        try:
            data = json.loads(json_str)
            return self._validate_json(data)
        except (json.JSONDecodeError, ValueError, TypeError):
            # Try to clean common JavaScript syntax that models might output
            try:
                cleaned = self._clean_javascript_syntax(json_str)
                data = json.loads(cleaned)
                return self._validate_json(data)
            except:
                return None
    
    def _clean_javascript_syntax(self, json_str):
        """
        Clean JavaScript syntax that might appear in JSON responses.
        E.g., array.join(), template literals, etc.
        """
        # Remove .join() calls on arrays
        # Pattern: ["item1", "item2"].join("separator") -> "item1separatoritem2"
        cleaned = re.sub(r'\[(.*?)\]\.join\(["\']([^"\']*?)["\']\)', 
                        lambda m: '"' + m.group(2).join(item.strip().strip('"').strip("'") 
                                                        for item in m.group(1).split(',')) + '"', 
                        json_str)
        return cleaned
    
    def _try_parse_json_with_context(self, json_str, full_text):
        """
        Try to parse a JSON string and validate it, using full text context
        to find rationale if it's missing from the JSON.
        Returns validated result or None.
        """
        try:
            data = json.loads(json_str)
            # Find where this JSON string appears in the full text
            # Try exact match first
            json_pos = full_text.find(json_str)
            # If not found, try to find by matching the opening brace and structure
            if json_pos == -1:
                # Find the first opening brace and try to match from there
                brace_pos = full_text.find('{')
                if brace_pos != -1:
                    # Try to find a JSON object starting at this position
                    # Extract a reasonable chunk and see if it contains our JSON
                    chunk = full_text[brace_pos:brace_pos + len(json_str) + 100]
                    # Normalize whitespace for comparison
                    normalized_chunk = re.sub(r'\s+', ' ', chunk.strip())
                    normalized_json = re.sub(r'\s+', ' ', json_str.strip())
                    if normalized_json in normalized_chunk:
                        json_pos = brace_pos
            
            result = self._validate_json(data, full_text, json_pos if json_pos != -1 else None)
            return result
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Try to clean JavaScript syntax and parse again
            try:
                cleaned = self._clean_javascript_syntax(json_str)
                data = json.loads(cleaned)
                json_pos = full_text.find(json_str)
                if json_pos == -1:
                    json_pos = full_text.find('{')
                result = self._validate_json(data, full_text, json_pos if json_pos != -1 else None)
                return result
            except Exception as inner_e:
                try:
                    self.controller._log(
                        "Parse exception in evaluator JSON.",
                        level="DEBUG", exc_type=type(inner_e).__name__, msg=str(inner_e)[:200]
                    )
                except Exception:
                    pass
                return None
    
    def _validate_json(self, data, full_text=None, json_pos=None):
        """
        Validate and normalize the parsed JSON data.
        Handles alternative field names: 'score'/'rating', nested structures, etc.
        If full_text is provided and rationale is missing, attempts to extract it from the full text.
        json_pos: position in full_text where the JSON string starts (for extracting pre-JSON rationale).
        """
        if not isinstance(data, dict):
            return None
        
        # Extract rating/score - handle both field names and nested structures
        rating = None
        rationale = None
        
        # Direct fields
        if 'rating' in data:
            rating = data['rating']
        elif 'score' in data:
            rating = data['score']
        
        # Nested structures (e.g., {"response": {...}, "score": 8})
        if rating is None:
            # Check if there's a nested structure with score/rating
            for key, value in data.items():
                if isinstance(value, dict):
                    if 'rating' in value:
                        rating = value['rating']
                    elif 'score' in value:
                        rating = value['score']
        
        # Extract rationale - handle both direct and nested
        if 'rationale' in data:
            rationale = data['rationale']
            # Handle array rationales (convert to string)
            if isinstance(rationale, list):
                rationale = ' '.join(str(item) for item in rationale if item)
        else:
            # Check nested structures for rationale
            for key, value in data.items():
                if isinstance(value, dict) and 'rationale' in value:
                    rationale = value['rationale']
                    # Handle array rationales in nested structures
                    if isinstance(rationale, list):
                        rationale = ' '.join(str(item) for item in rationale if item)
                    break
        
        # If we have a rating but no rationale, try to construct one from other fields
        if rating is not None and rationale is None:
            # Check for common alternative field names in the JSON
            for alt_name in ['reasoning', 'explanation', 'analysis', 'comment', 'notes', 'evaluation']:
                if alt_name in data:
                    alt_value = data[alt_name]
                    # Handle arrays
                    if isinstance(alt_value, list):
                        rationale = ' '.join(str(item) for item in alt_value if item)
                    else:
                        rationale = str(alt_value)
                    break
                # Also check nested structures
                for key, value in data.items():
                    if isinstance(value, dict) and alt_name in value:
                        alt_value = value[alt_name]
                        # Handle arrays
                        if isinstance(alt_value, list):
                            rationale = ' '.join(str(item) for item in alt_value if item)
                        else:
                            rationale = str(alt_value)
                        break
                if rationale:
                    break
        
        # If still no rationale and we have full text, try to extract it from surrounding text
        if rating is not None and rationale is None and full_text:
            # Strategy 1: Look for explicit rationale sections after the JSON block
            # Common patterns: "**Rationale:**", "Rationale:", "## Rationale", etc.
            rationale_patterns = [
                r'(?:\*\*)?Rationale:?\s*\*\*\s*\n?(.+?)(?=\n\n|\n\*\*|$)',
                r'##\s*Rationale\s*\n(.+?)(?=\n##|$)',
                r'Rationale\s*:\s*\n(.+?)(?=\n\n|$)',
            ]
            for pattern in rationale_patterns:
                match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
                if match:
                    rationale = match.group(1).strip()
                    # Clean up markdown formatting
                    rationale = re.sub(r'\*\*([^*]+)\*\*', r'\1', rationale)  # Remove bold
                    rationale = re.sub(r'^\s*[-*]\s*', '', rationale, flags=re.MULTILINE)  # Remove list markers
                    if len(rationale) > 20:  # Only use if it's substantial
                        break
            
            # Strategy 2: If still no rationale, look for reasoning/thinking text BEFORE the JSON
            # This handles cases where models output analysis before the final JSON
            if rationale is None and json_pos is not None and json_pos > 0:
                # Extract text before the JSON
                pre_json_text = full_text[:json_pos].strip()
                # Look for substantial reasoning text (at least 50 chars, not just whitespace)
                # Remove special tokens and clean up
                pre_json_text = self._clean_response_text(pre_json_text)
                # Look for sentences that contain evaluation keywords
                evaluation_keywords = ['evaluate', 'analysis', 'reasoning', 'consider', 'according', 
                                     'rubric', 'score', 'rating', 'assess', 'judge']
                sentences = re.split(r'[.!?]\s+', pre_json_text)
                relevant_sentences = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 30:  # Substantial sentence
                        # Check if it contains evaluation-related content
                        if any(keyword in sentence.lower() for keyword in evaluation_keywords):
                            relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    # Combine relevant sentences as rationale
                    rationale = '. '.join(relevant_sentences[:3])  # Take up to 3 most relevant
                    if len(rationale) > 50:  # Ensure it's substantial
                        rationale = rationale.strip()
                    else:
                        rationale = None
                
                # If we still don't have rationale but have substantial pre-JSON text, use it
                if rationale is None and len(pre_json_text) > 100:
                    # Use the last substantial portion of pre-JSON text as rationale
                    # Take the last 200-500 characters (likely to contain the reasoning)
                    rationale = pre_json_text[-min(500, len(pre_json_text)):].strip()
                    # Clean up: remove very short lines, keep substantial content
                    lines = [line for line in rationale.split('\n') if len(line.strip()) > 20]
                    if lines:
                        rationale = ' '.join(lines[-5:])  # Last 5 substantial lines
                        if len(rationale) < 50:
                            rationale = None
                
                # Final fallback: if we have any substantial pre-JSON text and a score, use it
                # This handles cases where the model provides reasoning but it doesn't match our patterns
                if rationale is None and len(pre_json_text) > 50:
                    # Take a reasonable chunk of the pre-JSON text
                    # Prefer text that mentions scoring/evaluation
                    if any(kw in pre_json_text.lower() for kw in ['score', 'rating', 'evaluate', 'rubric']):
                        # Take up to 300 chars of the most relevant portion
                        rationale = pre_json_text[-min(300, len(pre_json_text)):].strip()
                        if len(rationale) >= 30:  # Minimum length requirement
                            # Clean up any remaining artifacts
                            rationale = re.sub(r'\s+', ' ', rationale)
                            rationale = rationale.strip()
                        else:
                            rationale = None
        
        # Validate we have both required fields
        if rating is None:
            return None
        
        # If we have a rating but no rationale, we can't create a valid evaluation
        # (rationale is required for the evaluation record)
        if rationale is None:
            return None
        
        try:
            # Convert rating to integer and clamp to 0-10
            score = int(rating)
            score = max(0, min(10, score))
            
            # Ensure rationale is a string and not empty
            rationale_str = str(rationale).strip()
            if not rationale_str:
                return None
            
            return {
                'rating': score,
                'rationale': rationale_str
            }
        except (ValueError, TypeError):
            return None
    
    def _extract_fields_with_regex(self, text):
        """
        Last resort: extract rationale and rating using regex patterns.
        Only used when JSON parsing completely fails.
        """
        try:
            # Look for "rationale" field (can be string or object)
            rationale = None
            rat_patterns = [
                r'"rationale"\s*:\s*"([^"]*)"',  # String rationale
                r'"rationale"\s*:\s*\{([^}]*)\}',  # Object rationale
            ]
            for pattern in rat_patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    rationale = match.group(1).strip()
                    break
            
            # Look for rating or score
            rating = None
            rating_patterns = [
                r'"rating"\s*:\s*(\d+)',
                r'"score"\s*:\s*(\d+)',
            ]
            for pattern in rating_patterns:
                match = re.search(pattern, text)
                if match:
                    rating = int(match.group(1))
                    break
            
            if rationale and rating is not None:
                return {
                    'rating': max(0, min(10, rating)),
                    'rationale': rationale
                }
        except Exception:
            pass
        
        return None

    def _handle_error(self, sys_id, llm_id, response_id, category, error_msg, step, raw_response=None):
        self.controller._log(error_msg)
        self.db.log_error(sys_id, llm_id, step, error_msg)
        self.db.add_evaluation(response_id, llm_id, {'rating': -1, 'rationale': error_msg, 'prompt_category': category, 'raw_response': raw_response})