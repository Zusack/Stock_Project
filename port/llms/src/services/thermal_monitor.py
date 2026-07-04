import threading
import time
import psutil

from src.database.manager import DatabaseManager, DEFAULT_DB_FILE
import src.utils.memory_utils as memory_utils
from src.services.global_control_service import GlobalControlService
from src.services.event_bus import event_bus

class ThermalMonitor:
    def __init__(self, controller_ref):
        self.controller = controller_ref
        self.global_control = GlobalControlService()
        
        self.thermal_enabled = True
        self.temp_cooldown = 71
        self.temp_pause = 80
        self.temp_resume = 75
        self.temp_stop = 83

        self.is_thermally_paused = False
        self.is_hard_stopped = False

        # Power safety
        self.power_enabled = False
        self.power_delay_sec = 90
        self.system_power_pause = 200
        self.system_power_stop = 250
        self.gpu_power_pause = 300
        self.gpu_power_stop = 350
        self.is_power_hard_stopped = False
        self.is_power_paused = False
        self.power_throttle_events = 0
        self.last_power_event_time = 0.0
        
        # Soft-stop state tracking
        self.is_soft_stopped = False
        self.soft_stop_start_time = 0.0
        self.soft_stop_pause_threshold = 0.0  # The pause threshold that triggered soft-stop
        self.soft_stop_scope = None  # "system" or "gpu"
        self.power_pause_periods_elapsed = 0  # Count of delay periods during pause
        self.power_dropped_below_pause = False  # Track if power dropped during pause delay
        self.soft_stop_safe_start_time = 0.0  # When power first dropped below threshold in soft-stop

        self.vitals_thread = None
        self.collecting_vitals = False
        self.vitals_buffer = []
        self.throttle_events = 0

        self._load_thermal_settings()

        event_bus.subscribe("thermal_reset", self.reset_hard_stop_state)
        event_bus.subscribe("power_reset", self.reset_power_hard_stop_state)

    def _load_thermal_settings(self):
        try:
            with DatabaseManager(DEFAULT_DB_FILE) as db:
                disabled_str = db.get_setting("thermal_disabled", "0")
                user_enabled = (disabled_str == "0")
                if user_enabled:
                    if self._test_sensors():
                        self.thermal_enabled = True
                        self.temp_cooldown = int(db.get_setting("temp_cooldown", "71"))
                        self.temp_pause = int(db.get_setting("temp_pause", "80"))
                        self.temp_resume = int(db.get_setting("temp_resume", "75"))
                        self.temp_stop = int(db.get_setting("temp_stop", "83"))
                        self.controller._log(f"Thermal Safety: ON (Stop: {self.temp_stop}C, Pause: {self.temp_pause}C)")
                    else:
                        self.thermal_enabled = False
                        self.controller._log("Thermal Safety: AUTO-DISABLED (No sensors detected)")
                else:
                    self.thermal_enabled = False
                    self.controller._log("Thermal Safety: Disabled by User Settings.")

                power_disabled_str = db.get_setting("power_disabled", "1")
                power_user_enabled = (power_disabled_str == "0")
                if power_user_enabled and self._test_power_sensors():
                    self.power_enabled = True
                    self.power_delay_sec = int(db.get_setting("power_delay_sec", "90"))
                    self.system_power_pause = int(db.get_setting("system_power_pause", "200"))
                    self.system_power_stop = int(db.get_setting("system_power_stop", "250"))
                    self.gpu_power_pause = int(db.get_setting("gpu_power_pause", "300"))
                    self.gpu_power_stop = int(db.get_setting("gpu_power_stop", "350"))
                    self.controller._log(f"Power Safety: ON (Sys: {self.system_power_stop}W stop, GPU: {self.gpu_power_stop}W stop, delay: {self.power_delay_sec}s)")
                else:
                    self.power_enabled = False
                    if power_user_enabled:
                        self.controller._log("Power Safety: AUTO-DISABLED (No power sensors).")
        except Exception as e:
            print(f"Failed to load thermal/power settings: {e}")
            self.thermal_enabled = False
            self.power_enabled = False

    def _test_sensors(self):
        cpu_ok = gpu_ok = False
        try:
            if memory_utils.get_cpu_temperature() > 0: cpu_ok = True
        except Exception: pass
        try:
            _, _, _, gpu_temp = memory_utils.get_gpu_vitals()
            if gpu_temp > 0: gpu_ok = True
        except Exception: pass
        return cpu_ok or gpu_ok

    def _test_power_sensors(self):
        _, sys_avail = memory_utils.get_system_power()
        _, gpu_avail = memory_utils.get_gpu_power()
        return sys_avail or gpu_avail

    def reset_counters(self):
        self.throttle_events = 0
        self.power_throttle_events = 0
        self.vitals_buffer = []

    def reset_hard_stop_state(self, **kwargs):
        """Called when user explicitly clicks Resume on the thermal modal."""
        self.is_hard_stopped = False
        self.is_thermally_paused = False
        self.controller._log("Thermal Hard Stop cleared by user.")

    def reset_power_hard_stop_state(self, **kwargs):
        """Called when user explicitly clicks Resume on the power modal."""
        self.is_power_hard_stopped = False
        self.is_power_paused = False
        self.is_soft_stopped = False
        self.soft_stop_start_time = 0.0
        self.soft_stop_pause_threshold = 0.0
        self.soft_stop_scope = None
        self.power_pause_periods_elapsed = 0
        self.power_dropped_below_pause = False
        self.soft_stop_safe_start_time = 0.0
        self.controller._log("Power Hard Stop cleared by user.")

    def start_monitoring(self):
        if not self.vitals_thread:
            self.vitals_thread = threading.Thread(target=self._vitals_loop, daemon=True)
            self.vitals_thread.start()

    def _monitor_thermals(self, cpu_temp, gpu_temp):
        if not self.thermal_enabled: return

        c = cpu_temp if cpu_temp else 0
        g = gpu_temp if gpu_temp else 0
        current_max = max(c, g)

        # 1. HARD STOP Condition (User Intervention Required)
        if current_max >= self.temp_stop:
            # Only trigger if we aren't ALREADY hard stopped
            if not self.is_hard_stopped:
                self.controller._log(
                    f"CRITICAL: Temp {current_max}C exceeded Hard Limit {self.temp_stop}C. Halting.",
                    level="ERROR", event="thermal_hard_stop", temp=current_max, limit=self.temp_stop
                )
                # Force Pause and full Stop so the run loop exits (stream loop will see STOP on next check)
                self.global_control.request_pause(source="thermal")
                self.global_control.request_stop()
                self.is_thermally_paused = True
                self.is_hard_stopped = True  # Lock this state
                self.throttle_events += 1
                event_bus.emit("thermal_hard_stop", t=current_max, limit=self.temp_stop)
            return

        # 2. SOFT PAUSE Condition (Auto-Resume Allowed)
        if current_max >= self.temp_pause and not self.is_thermally_paused and not self.is_hard_stopped:
            self.controller._log(
                f"WARNING: Temp {current_max}C exceeded Pause Limit {self.temp_pause}C. Pausing for cooldown.",
                level="WARN", event="thermal_pause", temp=current_max, limit=self.temp_pause
            )
            self.global_control.request_pause(source="thermal")
            self.is_thermally_paused = True
            self.throttle_events += 1
            return

        # 3. AUTO-RESUME Condition
        if self.is_thermally_paused and not self.is_hard_stopped and current_max <= self.temp_resume:
            self.controller._log(
                f"Thermal Status: Cooled to {current_max}C. Resuming.",
                level="INFO", event="thermal_resume", temp=current_max, limit=self.temp_resume
            )
            self.global_control.request_resume(source="thermal")
            self.is_thermally_paused = False

    def _monitor_power(self, sys_power: float, sys_avail: bool, gpu_power: float, gpu_avail: bool):
        if not self.power_enabled:
            return
        now = time.monotonic()
        
        # Check soft-stop escalation to hard-stop (10x delay period)
        if self.is_soft_stopped and not self.is_power_hard_stopped:
            elapsed_soft_stop = now - self.soft_stop_start_time
            if elapsed_soft_stop >= (10 * self.power_delay_sec):
                # Escalate to hard-stop
                self.controller._log(
                    f"CRITICAL: Power exceeded pause threshold {self.soft_stop_pause_threshold}W for {elapsed_soft_stop:.0f}s "
                    f"(>{10 * self.power_delay_sec}s) even after model unload. Escalating to Hard Stop.",
                    level="ERROR", event="power_soft_stop_escalated", threshold=self.soft_stop_pause_threshold,
                    elapsed=elapsed_soft_stop, scope=self.soft_stop_scope
                )
                self.global_control.request_pause(source="power")
                self.global_control.request_stop()
                self.is_power_hard_stopped = True
                self.is_power_paused = True
                event_bus.emit("power_hard_stop", 
                    watts=sys_power if self.soft_stop_scope == "system" else gpu_power,
                    limit=self.soft_stop_pause_threshold, 
                    scope=self.soft_stop_scope,
                    reason="soft_stop_escalated"
                )
                # Show user message about increasing pause threshold
                self.controller._log(
                    f"RECOMMENDATION: Consider increasing {self.soft_stop_scope.upper()} Power Pause threshold "
                    f"above {self.soft_stop_pause_threshold}W to prevent this issue.",
                    level="WARN"
                )
                return
        
        # Check soft-stop auto-restart condition
        if self.is_soft_stopped and not self.is_power_hard_stopped:
            current_power = sys_power if self.soft_stop_scope == "system" else gpu_power
            current_avail = sys_avail if self.soft_stop_scope == "system" else gpu_avail
            
            if current_avail and current_power < self.soft_stop_pause_threshold:
                # Power dropped below threshold
                if self.soft_stop_safe_start_time == 0.0:
                    # First time dropping below threshold
                    self.soft_stop_safe_start_time = now
                    self.controller._log(
                        f"Power dropped below pause threshold ({current_power:.1f}W < {self.soft_stop_pause_threshold}W). "
                        f"Monitoring for {self.power_delay_sec}s before auto-restart...",
                        level="INFO"
                    )
                elif (now - self.soft_stop_safe_start_time) >= self.power_delay_sec:
                    # Power has been below threshold for full delay period - auto-restart
                    self.controller._log(
                        f"Power remained below pause threshold for {self.power_delay_sec}s. Auto-restarting process.",
                        level="INFO", event="power_soft_stop_restart", power=round(current_power, 1), threshold=self.soft_stop_pause_threshold
                    )
                    # Clear soft-stop state
                    self.is_soft_stopped = False
                    self.soft_stop_start_time = 0.0
                    self.soft_stop_pause_threshold = 0.0
                    self.soft_stop_scope = None
                    self.soft_stop_safe_start_time = 0.0
                    self.power_pause_periods_elapsed = 0
                    self.power_dropped_below_pause = False
                    self.global_control.soft_stop_active = False  # Clear soft-stop flag
                    # Clear stop event to allow restart
                    self.global_control.stop_event.clear()
                    event_bus.emit("power_soft_stop_restart")
                    return
            else:
                # Power is still above threshold - reset safe start time
                if self.soft_stop_safe_start_time > 0.0:
                    self.soft_stop_safe_start_time = 0.0
                    self.controller._log(
                        f"Power rose above pause threshold again ({current_power:.1f}W >= {self.soft_stop_pause_threshold}W). "
                        f"Continuing soft-stop...",
                        level="INFO"
                    )
            return  # Don't process other power logic while in soft-stop
        
        # 1. HARD STOP
        if sys_avail and sys_power >= self.system_power_stop and not self.is_power_hard_stopped and not self.is_soft_stopped:
            self.controller._log(
                f"CRITICAL: System power {sys_power}W exceeded limit {self.system_power_stop}W. Halting.",
                level="ERROR", event="power_hard_stop", power=round(sys_power, 1), limit=self.system_power_stop, scope="system"
            )
            self.last_power_event_time = now
            self.global_control.request_pause(source="power")
            self.global_control.request_stop()
            self.is_power_paused = True
            self.is_power_hard_stopped = True
            self.power_throttle_events += 1
            event_bus.emit("power_hard_stop", watts=sys_power, limit=self.system_power_stop, scope="system")
            return
        if gpu_avail and gpu_power >= self.gpu_power_stop and not self.is_power_hard_stopped and not self.is_soft_stopped:
            self.controller._log(
                f"CRITICAL: GPU power {gpu_power}W exceeded limit {self.gpu_power_stop}W. Halting.",
                level="ERROR", event="power_hard_stop", power=round(gpu_power, 1), limit=self.gpu_power_stop, scope="gpu"
            )
            self.last_power_event_time = now
            self.global_control.request_pause(source="power")
            self.global_control.request_stop()
            self.is_power_paused = True
            self.is_power_hard_stopped = True
            self.power_throttle_events += 1
            event_bus.emit("power_hard_stop", watts=gpu_power, limit=self.gpu_power_stop, scope="gpu")
            return
        
        # 2. SOFT PAUSE
        if not self.is_power_paused and not self.is_power_hard_stopped and not self.is_soft_stopped:
            if sys_avail and sys_power >= self.system_power_pause:
                self.controller._log(
                    f"WARNING: System power {sys_power}W exceeded pause limit {self.system_power_pause}W. Pausing (delay {self.power_delay_sec}s).",
                    level="WARN", event="power_pause", power=round(sys_power, 1), limit=self.system_power_pause, scope="system"
                )
                self.last_power_event_time = now
                self.global_control.request_pause(source="power")
                self.is_power_paused = True
                self.power_throttle_events += 1
                self.power_pause_periods_elapsed = 0
                self.power_dropped_below_pause = False
                return
            if gpu_avail and gpu_power >= self.gpu_power_pause:
                self.controller._log(
                    f"WARNING: GPU power {gpu_power}W exceeded pause limit {self.gpu_power_pause}W. Pausing (delay {self.power_delay_sec}s).",
                    level="WARN", event="power_pause", power=round(gpu_power, 1), limit=self.gpu_power_pause, scope="gpu"
                )
                self.last_power_event_time = now
                self.global_control.request_pause(source="power")
                self.is_power_paused = True
                self.power_throttle_events += 1
                self.power_pause_periods_elapsed = 0
                self.power_dropped_below_pause = False
                return
        
        # Monitor power during pause to check if it drops below threshold
        if self.is_power_paused and not self.is_power_hard_stopped and not self.is_soft_stopped:
            # Check if power dropped below pause threshold
            if sys_avail and sys_power < self.system_power_pause:
                self.power_dropped_below_pause = True
            elif gpu_avail and gpu_power < self.gpu_power_pause:
                self.power_dropped_below_pause = True
            
            # Check if delay period elapsed
            if (now - self.last_power_event_time) >= self.power_delay_sec:
                self.power_pause_periods_elapsed += 1
                
                if self.power_dropped_below_pause:
                    # Power dropped - resume normally
                    self.controller._log(
                        f"Power Status: Delay {self.power_delay_sec}s elapsed and power dropped below threshold. Resuming.",
                        level="INFO", event="power_resume", delay=self.power_delay_sec
                    )
                    self.global_control.request_resume(source="power")
                    self.is_power_paused = False
                    self.power_pause_periods_elapsed = 0
                    self.power_dropped_below_pause = False
                else:
                    # Power did NOT drop - check if we've waited 3 periods
                    if self.power_pause_periods_elapsed >= 3:
                        # Trigger soft-stop
                        pause_threshold = self.system_power_pause if sys_avail and sys_power >= self.system_power_pause else self.gpu_power_pause
                        scope = "system" if sys_avail and sys_power >= self.system_power_pause else "gpu"
                        current_power = sys_power if scope == "system" else gpu_power
                        
                        self.controller._log(
                            f"CRITICAL: Power {current_power:.1f}W did not drop below pause threshold {pause_threshold}W "
                            f"after {3 * self.power_delay_sec}s. Triggering soft-stop (unloading model).",
                            level="ERROR", event="power_soft_stop", power=round(current_power, 1), 
                            threshold=pause_threshold, scope=scope
                        )
                        self.is_soft_stopped = True
                        self.soft_stop_start_time = now
                        self.soft_stop_pause_threshold = pause_threshold
                        self.soft_stop_scope = scope
                        self.soft_stop_safe_start_time = 0.0
                        self.global_control.soft_stop_active = True  # Mark soft-stop in global control
                        self.global_control.request_stop()  # Stop the process
                        self.is_power_paused = False  # Clear pause state
                        self.power_pause_periods_elapsed = 0
                        self.power_dropped_below_pause = False
                        event_bus.emit("power_soft_stop", watts=current_power, threshold=pause_threshold, scope=scope)
                    else:
                        # Wait another period
                        self.last_power_event_time = now
                        self.power_dropped_below_pause = False  # Reset for next period
                        self.controller._log(
                            f"Power still above pause threshold after {self.power_pause_periods_elapsed * self.power_delay_sec}s. "
                            f"Waiting period {self.power_pause_periods_elapsed + 1}/3...",
                            level="WARN"
                        )

    def wait_for_power_delay(self):
        """Wait until power_delay_sec has passed since last power event. No-op if none or not enabled."""
        if not self.power_enabled or self.last_power_event_time <= 0:
            return
        if self.global_control.stop_event.is_set():
            return
        now = time.monotonic()
        elapsed = now - self.last_power_event_time
        if elapsed >= self.power_delay_sec:
            return
        remain = self.power_delay_sec - elapsed
        self.controller._log(f"Power delay: waiting {max(1, int(remain))}s before next step...")
        while not self.global_control.stop_event.is_set():
            now = time.monotonic()
            if (now - self.last_power_event_time) >= self.power_delay_sec:
                break
            if self.global_control.user_paused or self.global_control.thermal_paused or self.global_control.power_paused:
                time.sleep(0.5)
                continue
            time.sleep(1)

    def wait_for_safe_start_temp(self):
        """
        Wait for temperature to drop below Running Temp Start threshold and for power delay to elapse.
        Blocks if power hard stop is active until user clears it.
        """
        if self.global_control.stop_event.is_set():
            return
        # Block until power hard stop is cleared by user
        while self.is_power_hard_stopped and not self.global_control.stop_event.is_set():
            time.sleep(0.5)

        if self.thermal_enabled:
            cpu_t = memory_utils.get_cpu_temperature()
            _, _, _, gpu_t = memory_utils.get_gpu_vitals()
            c = cpu_t if cpu_t else 0
            g = gpu_t if gpu_t else 0
            current_max = max(c, g)
            if current_max > self.temp_cooldown:
                self.controller._log(f"Temperature {current_max}C above safe start threshold. Waiting for cooldown to {self.temp_cooldown}C...")
                while not self.global_control.stop_event.is_set():
                    cpu_t = memory_utils.get_cpu_temperature()
                    _, _, _, gpu_t = memory_utils.get_gpu_vitals()
                    c = cpu_t if cpu_t else 0
                    g = gpu_t if gpu_t else 0
                    current_max = max(c, g)
                    if current_max <= self.temp_cooldown:
                        self.controller._log(f"Temperature cooled to {current_max}C. Resuming operations.")
                        break
                    if self.global_control.user_paused or self.global_control.thermal_paused or self.global_control.power_paused:
                        while (self.global_control.user_paused or self.global_control.thermal_paused or self.global_control.power_paused) and not self.global_control.stop_event.is_set():
                            time.sleep(0.5)
                    time.sleep(2)

        self.wait_for_power_delay()

    def cooldown_wait(self):
        """
        Standard cooldown wait after model unload (thermal and power delay).
        """
        if self.global_control.stop_event.is_set():
            return
        if self.thermal_enabled:
            self.controller._log(f"Cooling down model... Target < {self.temp_cooldown}C")
            time.sleep(2)
            while not self.global_control.stop_event.is_set():
                cpu_t = memory_utils.get_cpu_temperature()
                _, _, _, gpu_t = memory_utils.get_gpu_vitals()
                c = cpu_t if cpu_t else 0
                g = gpu_t if gpu_t else 0
                current_max = max(c, g)
                if current_max <= self.temp_cooldown:
                    break
                if self.global_control.user_paused or self.global_control.thermal_paused or self.global_control.power_paused:
                    while (self.global_control.user_paused or self.global_control.thermal_paused or self.global_control.power_paused) and not self.global_control.stop_event.is_set():
                        time.sleep(0.5)
                time.sleep(2)
        self.wait_for_power_delay()

    def _vitals_loop(self):
        try:
            psutil.cpu_percent(interval=None)
            while not self.global_control.stop_event.is_set():
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_temp = memory_utils.get_cpu_temperature()
                sys_total, sys_avail = memory_utils.get_system_memory()
                vram_total, vram_used, gpu_util, gpu_temp = memory_utils.get_gpu_vitals()
                sys_power, sys_power_ok = memory_utils.get_system_power()
                gpu_power, gpu_power_ok = memory_utils.get_gpu_power()
                battery = memory_utils.get_battery_status()

                self._monitor_thermals(cpu_temp, gpu_temp)
                self._monitor_power(sys_power, sys_power_ok, gpu_power, gpu_power_ok)

                if self.collecting_vitals:
                    sys_used = sys_total - sys_avail
                    self.vitals_buffer.append({
                        'cpu_util': cpu_percent, 'gpu_util': gpu_util,
                        'cpu_temp': cpu_temp, 'gpu_temp': gpu_temp,
                        'ram_used': sys_used, 'vram_used': vram_used,
                        'sys_power_w': sys_power, 'gpu_power_w': gpu_power,
                        'battery': battery
                    })

                event_bus.emit("vitals_update",
                    cpu=cpu_percent, cpu_temp=cpu_temp,
                    sys_total=sys_total, sys_avail=sys_avail,
                    vram_total=vram_total, vram_used=vram_used,
                    gpu_util=gpu_util, gpu_temp=gpu_temp,
                    sys_power_w=sys_power, gpu_power_w=gpu_power,
                    battery=battery
                )

                for _ in range(2):
                    if self.global_control.stop_event.is_set():
                        break
                    time.sleep(0.5)
        except Exception:
            pass
        self.vitals_thread = None