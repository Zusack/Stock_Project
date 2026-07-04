import time
import math
import statistics
import src.utils.memory_utils as memory_utils
from src.services.event_bus import event_bus
from src.services.global_control_service import GlobalControlService

class DeepAuditEngine:
    def __init__(self, db, llm_id, model_identifier, context_limit, backend=None):
        self.db = db
        self.llm_id = llm_id
        self.identifier = model_identifier
        self.max_ctx = context_limit
        self.global_control = GlobalControlService()
        self.backend = backend
        self.results = {}
        self.had_errors = False
        self.error_details = []

    def _smart_sleep(self, seconds):
        """
        Sleeps in small increments to check for stop flags continuously.
        Returns True if a STOP signal was detected.
        """
        end = time.time() + seconds
        while time.time() < end:
            if self._check_stop(): return True
            time.sleep(0.1)
        return False

    def run(self):
        sys_id = self.global_control.active_system_id
        audit_start = time.time()

        # 1. Determine Context Steps - Start conservatively and work up
        # Start with safe, common sizes and test incrementally
        base_steps = [2048, 4096, 8192, 16384]
        
        # Add percentage-based steps only if max_ctx is reasonable
        if self.max_ctx >= 32768:
            # Add larger steps for high-context models
            percentage_steps = [int(self.max_ctx * p) for p in [0.25, 0.50, 0.75]]
            base_steps.extend(percentage_steps)
        
        steps = sorted(list(set([s for s in base_steps if s <= self.max_ctx]))) # Unique & Sorted, within limit
        total_steps = len(steps) + 3 # Contexts + (Offload + Cache + Batch)
        current_step = 0

        # Data Collectors
        ctx_mem_points = [] # List of tuples: (tokens, gb_used)
        max_vram_usage = 0
        max_ram_usage = 0
        consecutive_context_failures = 0
        max_working_context = 0
        
        event_bus.emit("status_update", message=f"Starting Deep Audit on {self.identifier}...")

        if True:  # Preserved block structure (was `with lms.Client()`)
            # --- PHASE A: CONTEXT SWEEP (Max Offload) ---
            # We measure memory usage at various context lengths to establish the Slope.
            for ctx in steps:
                if self._check_stop(): return
                current_step += 1
                event_bus.emit("audit_progress", step=current_step, total=total_steps, msg=f"Testing Context: {ctx}")
                
                mem_used = self._measure_load(ctx, gpu_offload=-1)
                
                if mem_used > 0:
                    ctx_mem_points.append((ctx, mem_used))
                    max_working_context = ctx
                    consecutive_context_failures = 0
                    # Check if this was our "Max" attempt for VRAM tracking
                    if ctx == steps[-1]:
                        max_vram_usage = mem_used
                else:
                    # Context failed - might be too large for this model
                    consecutive_context_failures += 1
                    print(f"[DeepAudit] Context {ctx} failed - model may not support this size")
                    
                    # If we've had 2 consecutive failures, stop testing larger contexts
                    if consecutive_context_failures >= 2:
                        print(f"[DeepAudit] Multiple context failures detected. Max working context: {max_working_context}")
                        print(f"[DeepAudit] Skipping remaining larger context tests.")
                        # Adjust total steps for progress bar
                        remaining_steps = len([s for s in steps if s > ctx])
                        total_steps -= remaining_steps
                        break

            # --- PHASE B: OFFLOAD CHECK (CPU Only) ---
            # We force everything to System RAM to see the difference in footprint.
            # Only do this if we have successful measurements
            if not self._check_stop() and len(ctx_mem_points) > 0:
                current_step += 1
                event_bus.emit("audit_progress", step=current_step, total=total_steps, msg="Testing CPU Offload")
                # Use a known working context size from our successful measurements
                test_ctx = min([ctx for ctx, _ in ctx_mem_points])  # Use smallest working context
                max_ram_usage = self._measure_load(test_ctx, gpu_offload=0)
                
                if max_ram_usage == 0:
                    print(f"[DeepAudit] CPU offload test failed - this is normal for some models")
                    # Use VRAM measurement as fallback
                    max_ram_usage = next((mem for ctx, mem in ctx_mem_points if ctx == test_ctx), 0)

            # --- PHASE C: CACHE QUANTIZATION CHECK ---
            # We check if enabling Q8/Q4 cache reduces memory usage at max context.
            kv_factor = 1.0
            if not self._check_stop() and len(ctx_mem_points) > 0:
                current_step += 1
                event_bus.emit("audit_progress", step=current_step, total=total_steps, msg="Testing KV Quantization")
                
                # Baseline is Phase A (Max Offload, Normal Cache)
                # We compare against the LARGEST successful context measurement
                largest_ctx = max([ctx for ctx, _ in ctx_mem_points])
                baseline_mem = next((m for c, m in reversed(ctx_mem_points) if c == largest_ctx), 0)
                
                if baseline_mem > 0:
                    # Try to force generic config keys for Quantized Cache
                    quant_config = {
                        "gpu_layers": -1, 
                        "context_length": largest_ctx,
                        "rope_frequency_base": 0, # Dummy to ensure dict isn't empty
                        # Common LM Studio / llama.cpp keys
                        "flash_attn": True,
                        "type_k": "q8_0", 
                        "type_v": "q8_0" 
                    }
                    
                    mem_quant = self._measure_load_advanced(quant_config)
                    if mem_quant > 0 and mem_quant < baseline_mem:
                        kv_factor = mem_quant / baseline_mem

            # --- PHASE D: BATCH SIZE CHECK ---
            # We check overhead of increasing batch size (scratchpad memory).
            batch_overhead = 0.0
            if not self._check_stop() and len(ctx_mem_points) >= 2:
                current_step += 1
                event_bus.emit("audit_progress", step=current_step, total=total_steps, msg="Testing Batch Size Impact")
                
                # Test at mid-context with high batch - use middle of successful measurements
                sorted_contexts = sorted([ctx for ctx, _ in ctx_mem_points])
                mid_ctx = sorted_contexts[len(sorted_contexts)//2]
                base_mem = next((m for c, m in ctx_mem_points if c == mid_ctx), 0)
                
                if base_mem > 0:
                    # Standard is usually 512. Let's push to 2048 to see delta
                    high_batch_mem = self._measure_load_advanced({"gpu_layers": -1, "context_length": mid_ctx}, batch_override=2048)
                    if high_batch_mem > base_mem:
                        # Normalize to "Per 512" unit
                        delta = high_batch_mem - base_mem
                        # (2048 - 512) / 512 = 3 units of increase
                        batch_overhead = delta / 3.0 

        # --- CALCULATION ---
        slope, intercept, r2 = self._calc_regression(ctx_mem_points)
        
        # Report measurement results
        print(f"\n[DeepAudit] Measurement Summary:")
        print(f"  - Successful measurements: {len(ctx_mem_points)}")
        if len(ctx_mem_points) > 0:
            contexts = [ctx for ctx, _ in ctx_mem_points]
            print(f"  - Tested contexts: {', '.join(str(c) for c in contexts)} tokens")
            print(f"  - Memory range: {min(mem for _, mem in ctx_mem_points):.2f} GB - {max(mem for _, mem in ctx_mem_points):.2f} GB")
            print(f"  - Linear fit R²: {r2:.4f}")
        
        # If we had errors, report them with context about what worked
        if self.had_errors:
            print(f"\n[DeepAudit] {len(self.error_details)} error(s) encountered:")
            for detail in self.error_details:
                print(f"  - {detail}")
            if max_working_context > 0:
                print(f"\n[DeepAudit] Maximum working context: {max_working_context} tokens")
                print(f"[DeepAudit] Model's actual context limit appears to be around {max_working_context} tokens")
        
        # If we have at least 2 data points, we can still create a profile
        # even with some errors (partial success)
        success = len(ctx_mem_points) >= 2
        
        profile = {
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'max_vram': max_vram_usage,
            'max_ram': max_ram_usage,
            'kv_factor': kv_factor,
            'batch_overhead': batch_overhead,
            'has_errors': self.had_errors,
            'error_count': len(self.error_details),
            'total_audit_duration_sec': round(time.time() - audit_start, 4),
        }
        
        self.db.save_audit_profile(self.llm_id, sys_id, profile)
        
        if success:
            print(f"[DeepAudit] Profile created successfully with {len(ctx_mem_points)} valid measurements")
        else:
            print(f"[DeepAudit] Insufficient data for profile creation (only {len(ctx_mem_points)} measurements)")
        
        event_bus.emit("audit_complete", success=success)

    def _measure_load(self, ctx, gpu_offload):
        return self._measure_load_advanced({"gpu_layers": gpu_offload, "context_length": ctx})

    def _measure_load_advanced(self, config, batch_override=None):
        """
        Loads model, measures VRAM/RAM delta, unloads.
        Returns the delta in GB. Uses the backend abstraction.
        """
        backend = self.backend
        if backend is None:
            return 0.0
        try:
            if self._check_stop(): return 0.0

            _, sys_avail_pre = memory_utils.get_system_memory()
            _, vram_avail_pre = memory_utils.get_vram_snapshot()

            handle = backend.load_model(self.identifier, config)

            try:
                stream = backend.complete_stream(handle, " ", {"max_tokens": 1})
                for _ in stream:
                    pass
            except Exception:
                pass

            if self._smart_sleep(2):
                backend.unload_model(handle)
                return 0.0

            _, sys_avail_post = memory_utils.get_system_memory()
            _, vram_avail_post = memory_utils.get_vram_snapshot()

            backend.unload_model(handle)

            if self._smart_sleep(2):
                return 0.0

            sys_delta = max(0, sys_avail_pre - sys_avail_post)
            vram_delta = max(0, vram_avail_pre - vram_avail_post)

            if config.get("gpu_layers") == -1:
                return vram_delta
            elif config.get("gpu_layers") == 0:
                return sys_delta
            else:
                return vram_delta + sys_delta

        except Exception as e:
            if backend.is_unreachable_error(e):
                raise RuntimeError(backend.unreachable_message) from e
            error_str = str(e)
            ctx_length = config.get("context_length", "unknown")
            gpu_layers = config.get("gpu_layers", "unknown")

            print(f"[DeepAudit] Load Failed: {error_str}")

            self.had_errors = True
            if "failed to allocate buffer for kv cache" in error_str.lower():
                detail = f"Context {ctx_length}: Model does not support this context size"
                self.error_details.append(detail)
            elif "failed to load model" in error_str.lower():
                detail = f"Context {ctx_length}: Model failed to load (GPU offload: {gpu_layers})"
                self.error_details.append(detail)
            else:
                detail = f"Context {ctx_length}: {error_str[:100]}"
                self.error_details.append(detail)

            return 0.0

    def _calc_regression(self, points):
        """
        Simple Least Squares for y = mx + b
        points: list of (x, y) tuples -> (context_tokens, memory_gb)
        """
        if len(points) < 2: return 0, 0, 0
        
        n = len(points)
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        sum_xy = sum(p[0] * p[1] for p in points)
        sum_xx = sum(p[0]**2 for p in points)
        
        # Denominator for slope
        denom = (n * sum_xx) - (sum_x**2)
        if denom == 0: return 0, 0, 0
        
        # Slope (m) -> GB per Token
        slope_gb = ((n * sum_xy) - (sum_x * sum_y)) / denom
        
        # Convert to MB per Token for storage (more readable number)
        slope_mb = slope_gb * 1024
        
        # Intercept (b) -> GB
        intercept_gb = (sum_y - (slope_gb * sum_x)) / n
        
        # R^2 Calculation
        y_mean = sum_y / n
        ss_tot = sum((p[1] - y_mean)**2 for p in points)
        
        y_pred = [(slope_gb * p[0]) + intercept_gb for p in points]
        ss_res = sum((points[i][1] - y_pred[i])**2 for i in range(n))
        
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return slope_mb, intercept_gb, r2

    def _check_stop(self):
        # Also checks if RUN event is cleared (Paused)
        if self.global_control.stop_event.is_set():
            event_bus.emit("status_update", message="Audit Stopped.")
            return True
            
        # Blocking Pause Check
        if not self.global_control.run_event.is_set():
            event_bus.emit("status_update", message="Paused...")
            while not self.global_control.run_event.is_set():
                if self.global_control.stop_event.is_set(): return True
                time.sleep(0.5)
            event_bus.emit("status_update", message="Resumed.")
            
        return False