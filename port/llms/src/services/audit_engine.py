import json
import time
import flet as ft
import gc
import statistics
import src.utils.memory_utils as memory_utils
from src.utils.system_utils import SystemScanner
from src.services.event_bus import event_bus

# Thresholds: flag if Range Difference > 500 MB AND Relative Range > 0.10
RANGE_DIFF_GB_THRESHOLD = 0.5   # 500 MB
RELATIVE_RANGE_THRESHOLD = 0.10


class AuditEngine:
    def __init__(self, controller_ref):
        self.controller = controller_ref
        self.db = controller_ref.db

    def _smart_sleep(self, seconds):
        """Sleeps in small increments to check for stop flags."""
        end = time.time() + seconds
        while time.time() < end:
            if self.controller._check_control_flags() == "STOP": return True
            time.sleep(0.1)
        return False

    def _compute_flag_and_medians(self, used_list):
        """
        used_list: list of 3 dicts with keys: sys_used, vram_used, total_used, sys_total, sys_pre, vram_total, vram_pre,
            and optionally load_time_sec (from timed passes 2-4).
        Returns: (is_flagged_for_range, result_dict).
        result_dict always includes: pass2_total, pass3_total, pass4_total, range_diff_gb, rel_range_pct,
            median_vram, median_sys, median_total, metrics_for_accept, plus metrics keys for save_audit_metrics.
        """
        total_rams = [u['total_used'] for u in used_list]
        mx = max(total_rams)
        mn = min(total_rams)
        mean = sum(total_rams) / 3
        range_diff_gb = mx - mn
        rel_range = (range_diff_gb / mean) if mean and mean > 0 else 0.0
        is_flagged_range = (range_diff_gb > RANGE_DIFF_GB_THRESHOLD) and (rel_range > RELATIVE_RANGE_THRESHOLD)

        median_sys = statistics.median([u['sys_used'] for u in used_list])
        median_vram = statistics.median([u['vram_used'] for u in used_list])
        median_total = statistics.median(total_rams)

        avg_sys_total = sum(u['sys_total'] for u in used_list) / 3
        avg_vram_total = sum(u['vram_total'] for u in used_list) / 3
        avg_sys_pre = sum(u['sys_pre'] for u in used_list) / 3
        avg_vram_pre = sum(u['vram_pre'] for u in used_list) / 3
        sys_ram_avail_post = avg_sys_total - median_sys
        vram_avail_post = avg_vram_total - median_vram

        # Median load time from passes 2-4 (for task time estimation). Missing keys => 0.0 for legacy used_list.
        load_times = [u.get('load_time_sec', 0) for u in used_list]
        median_load_sec = round(statistics.median(load_times), 4) if load_times else 0.0

        metrics = {
            'sys_ram_total_gb': avg_sys_total, 'vram_total_gb': avg_vram_total,
            'sys_ram_avail_pre_gb': avg_sys_pre, 'vram_avail_pre_gb': avg_vram_pre,
            'sys_ram_avail_post_gb': sys_ram_avail_post, 'vram_avail_post_gb': vram_avail_post,
            'load_time_sec': median_load_sec,
        }

        result = {
            **metrics,
            'pass2_total': total_rams[0], 'pass3_total': total_rams[1], 'pass4_total': total_rams[2],
            'range_diff_gb': range_diff_gb, 'rel_range_pct': rel_range * 100,
            'median_vram': median_vram, 'median_sys': median_sys, 'median_total': median_total,
            'metrics_for_accept': metrics,
        }
        return is_flagged_range, result

    def run(self):
        self.controller._log("Starting Memory Audit (4-Pass, reporting 2nd–4th)...")
        if not self.controller._sync_static_data():
            self.controller._log("Sync failed. Aborting audit.")
            return

        scanner = SystemScanner()
        current_sys_id = 1
        sys_row = self.db.get_system_by_hash(scanner.system_id)
        if sys_row: current_sys_id = sys_row['system_id']

        # GPU presence from System DB (for 0 VRAM + >0.1 GB RAM rule)
        gpu_present = False
        if sys_row and sys_row['gpu_info']:
            try:
                gpu_list = json.loads(sys_row['gpu_info'])
                gpu_present = isinstance(gpu_list, list) and len(gpu_list) > 0
            except (json.JSONDecodeError, TypeError):
                pass

        all_llms = self.db.get_model_summary(system_id=current_sys_id)
        audit_target = getattr(self.controller, 'audit_target_llm_ids', None)
        if audit_target is not None:
            available_llms = [r for r in all_llms if r['llm_id'] in audit_target and r['available_on_disk']]
        else:
            available_llms = [row for row in all_llms if row['available_on_disk'] and (row['system_ram_post'] is None or row['system_ram_post'] == 0)]

        total = len(available_llms)
        if total == 0:
            self.controller._log("No models require auditing on this system.")
            return

        self.controller._log(f"Found {total} models to audit.")
        flagged_list = []
        is_reaudit = audit_target is not None

        for i, row in enumerate(available_llms):
            llm_id = row['llm_id']
            identifier = row['llm_identifier']
            display_name = row['display_name'] or identifier
            try:
                backend_type = row['backend_type'] or 'lmstudio'
            except (KeyError, TypeError):
                backend_type = 'lmstudio'
            backend = self.controller.get_backend(backend_type)

            if self.controller._check_control_flags() == "STOP": break

            self.controller._log(f"Auditing {i+1}/{total}: {identifier}")
            used_list = []

            for attempt in range(1, 5):
                if self.controller._check_control_flags() == "STOP": break

                self.controller._log(f"   -> Pass {attempt}/4: Loading...")
                sys_total, sys_avail_pre = memory_utils.get_system_memory()
                vram_total, vram_avail_pre = memory_utils.get_vram_snapshot()

                load_time_sec = 0.0
                try:
                    t_start = time.time()
                    handle = backend.load_model(identifier, {"gpu_layers": -1, "context_length": self.controller.context_length})
                    load_time_sec = round(time.time() - t_start, 4)
                    try:
                        stream = backend.complete_stream(handle, " ", {"max_tokens": 1})
                        for _ in stream:
                            pass
                    except Exception:
                        pass

                    if self._smart_sleep(2): break

                    _, sys_avail_post = memory_utils.get_system_memory()
                    _, vram_avail_post = memory_utils.get_vram_snapshot()

                    backend.unload_model(handle)
                    gc.collect()

                    sys_used = max(0, sys_avail_pre - sys_avail_post)
                    vram_used = max(0, vram_avail_pre - vram_avail_post)
                    total_used = vram_used + sys_used
                    self.controller._log(f"      -> VRAM: {vram_used:.2f} GB | RAM: {sys_used:.2f} GB | Total: {total_used:.2f} GB")

                    if attempt >= 2:
                        used_list.append({
                            'sys_used': sys_used, 'vram_used': vram_used, 'total_used': total_used,
                            'sys_total': sys_total, 'sys_pre': sys_avail_pre,
                            'vram_total': vram_total, 'vram_pre': vram_avail_pre,
                            'load_time_sec': load_time_sec,
                        })

                    if attempt < 4:
                        if self._smart_sleep(1): break

                except Exception as e:
                    if backend.is_unreachable_error(e):
                        raise RuntimeError(backend.unreachable_message) from e
                    self.controller._log(f"      -> Pass {attempt} Failed: {e}")

                if self.controller._check_control_flags() == "STOP": break

                if len(used_list) < 3:
                    self.controller._log("   -> Insufficient valid passes (need 2nd, 3rd, 4th). Skipping save.")
                    self.db.log_error(current_sys_id, llm_id, "Memory Audit", "Passes 2–4 did not produce 3 valid samples.")
                    event_bus.emit("llm_complete", llm_id=llm_id, color=ft.Colors.RED_800)
                    continue

                is_flagged, data = self._compute_flag_and_medians(used_list)
                reasons = []

                if is_flagged:
                    reasons.append("High variance")
                # Rule: Total Memory < 0 GB → flag for possible re-run
                if data.get('median_total') is not None and data['median_total'] < 0.0:
                    is_flagged = True
                    reasons.append("Negative total memory")
                # Rule: GPU present but 0 VRAM and >0.1 GB System RAM → flag (possible CPU-only run, re-audit)
                if gpu_present and (data.get('median_vram') or 0) < 0.01 and (data.get('median_sys') or 0) > 0.1:
                    is_flagged = True
                    reasons.append("GPU present but 0 VRAM, >0.1 GB RAM")

                if is_flagged:
                    data['flag_reason'] = "; ".join(reasons)
                    self.controller._log(f"   -> Flagged ({data['flag_reason']}).")
                    metrics_for_accept = dict(data['metrics_for_accept'])
                    metrics_for_accept['audit_context_length'] = self.controller.context_length
                    flagged_data = {
                        'llm_id': llm_id, 'identifier': identifier, 'display_name': display_name,
                        'system_id': current_sys_id,
                        'pass2_total': data['pass2_total'], 'pass3_total': data['pass3_total'], 'pass4_total': data['pass4_total'],
                        'range_diff_gb': data['range_diff_gb'], 'rel_range_pct': data['rel_range_pct'],
                        'median_vram': data['median_vram'], 'median_sys': data['median_sys'], 'median_total': data['median_total'],
                        'metrics_for_accept': metrics_for_accept,
                        'flag_reason': data['flag_reason'],
                        'accepted': False,
                    }
                    flagged_list.append(flagged_data)
                    if is_reaudit:
                        event_bus.emit("llm_audit_result", llm_id=llm_id, saved=False, flagged_data=flagged_data)
                    event_bus.emit("llm_complete", llm_id=llm_id, color=None)
                else:
                    data['audit_context_length'] = self.controller.context_length
                    self.db.save_audit_metrics(llm_id, current_sys_id, data)
                    self.controller._log(f"   -> Saved (Median of passes 2–4).")
                    if is_reaudit:
                        event_bus.emit("llm_audit_result", llm_id=llm_id, saved=True)
                    event_bus.emit("llm_complete", llm_id=llm_id, color=None)

                self.controller.thermal_monitor.cooldown_wait()

        if flagged_list and not is_reaudit:
            event_bus.emit("audit_flagged_models", flagged_models=flagged_list, system_id=current_sys_id)

        self.controller._log("Memory Audit Complete.")
