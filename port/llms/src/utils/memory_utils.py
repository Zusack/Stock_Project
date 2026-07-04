import psutil
import subprocess
import json
import sys
import platform
import re
import os
import shutil
import glob
import time
from typing import Tuple, Dict, Any

# Detect OS and Hardware Platform once on import
OS_TYPE = platform.system()
IS_MACOS = OS_TYPE == "Darwin"
IS_LINUX = OS_TYPE == "Linux"
IS_WINDOWS = OS_TYPE == "Windows"

# Cache for RAPL energy sampling (system power)
_rapl_cache: Tuple[float, float] = (0.0, 0.0)  # (energy_uj, timestamp)

def get_system_memory() -> Tuple[float, float]:
    """
    Gets total and available system RAM.
    Returns: (total_gb, available_gb)
    """
    try:
        mem = psutil.virtual_memory()
        total_gb = round(mem.total / (1024**3), 2)
        available_gb = round(mem.available / (1024**3), 2)
        return total_gb, available_gb
    except Exception as e:
        print(f"Warning: Could not get system memory via psutil: {e}")
        return (0.0, 0.0)

def get_cpu_temperature() -> float:
    """
    Gets the current CPU temperature (Cross-platform best effort).
    """
    try:
        if IS_LINUX:
            temps = psutil.sensors_temperatures()
            if not temps: return 0.0
            # Common sensor names
            for name in ['coretemp', 'k10temp', 'zenpower', 'acpitz']:
                if name in temps:
                    return max(entry.current for entry in temps[name])
            first_key = next(iter(temps))
            return max(entry.current for entry in temps[first_key])
        
        elif IS_MACOS:
            return 0.0
            
        elif IS_WINDOWS:
            # Windows usually requires WMI or OpenHardwareMonitor (complex)
            # returning 0.0 is standard safely fallback
            return 0.0

    except Exception:
        pass
    return 0.0

def get_vram_snapshot() -> Tuple[float, float]:
    """
    Gets total and AVAILABLE VRAM (Static Snapshot).
    Returns: (total_gb, available_gb)
    """
    total, _, used, _, _ = _get_gpu_stats_strategy()
    avail = total - used
    return round(total, 2), round(avail, 2)

def get_gpu_vitals() -> Tuple[float, float, int, int]:
    """
    Gets live GPU vitals.
    Returns: (total_gb, used_gb, util_%, temp_C)
    """
    total, _, used, util, temp = _get_gpu_stats_strategy()
    return total, used, util, temp


def get_system_power() -> Tuple[float, bool]:
    """
    Gets system/CPU package power in watts. Best-effort; needs Linux RAPL.
    Returns: (watts, available). (0.0, False) when not available.
    """
    global _rapl_cache
    try:
        if IS_LINUX:
            # Intel RAPL: .../intel-rapl:0/energy_uj or .../intel-rapl:0/0/energy_uj
            # AMD: amd_energy/energy1_input (different units; some in µJ)
            paths = []
            for p in glob.glob("/sys/class/powercap/intel-rapl:0/energy_uj"):
                paths.append(p)
            for p in glob.glob("/sys/class/powercap/intel-rapl:0/intel-rapl:0/energy_uj"):
                paths.append(p)
            for p in glob.glob("/sys/class/powercap/amd_energy/energy1_input"):
                paths.append(p)
            if not paths:
                return (0.0, False)
            path = paths[0]
            with open(path, "r") as f:
                raw = f.read().strip()
            cur = float(raw)
            now = time.monotonic()
            prev, t0 = _rapl_cache
            _rapl_cache = (cur, now)
            if t0 > 0 and (now - t0) >= 0.3:
                delta_uj = cur - prev
                if delta_uj < 0:
                    delta_uj = abs(delta_uj)  # overflow wrap
                power_w = (delta_uj / 1e6) / (now - t0)
                return (round(power_w, 2), True)
            return (0.0, True)
        # Windows, macOS: no standard userspace API without root/admin
        return (0.0, False)
    except Exception:
        return (0.0, False)


def get_gpu_power() -> Tuple[float, bool]:
    """
    Gets GPU power draw in watts. Uses nvidia-smi or rocm-smi when available.
    Returns: (watts, available). (0.0, False) when not available.
    """
    try:
        smi_path = _find_nvidia_smi_path()
        if smi_path:
            creationflags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
            cmd = [smi_path, "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
            r = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8", creationflags=creationflags)
            total = 0.0
            for line in r.stdout.strip().split("\n"):
                line = line.strip()
                if not line or line == "[N/A]":
                    continue
                total += float(line.replace(",", "."))
            return (round(total, 2), True)
        if IS_LINUX:
            try:
                rocm = subprocess.run(["rocm-smi", "--showpower", "--json"], capture_output=True, text=True, check=True, timeout=2)
                data = json.loads(rocm.stdout)
                total = 0.0
                for k, v in data.items():
                    if isinstance(v, dict) and "Average Graphics Power" in str(v):
                        for kk, vv in v.items():
                            if "Power" in kk and isinstance(vv, (int, float)):
                                total += float(vv)
                            elif isinstance(vv, str) and "W" in vv:
                                try:
                                    total += float(re.search(r"[\d.]+", vv).group())
                                except Exception:
                                    pass
                if total > 0:
                    return (round(total, 2), True)
            except Exception:
                pass
        return (0.0, False)
    except Exception:
        return (0.0, False)


def get_system_power_limit_w() -> Tuple[float, bool]:
    """
    Best-effort system/CPU package max power limit in watts (e.g. RAPL constraint).
    Returns (watts, available). (0.0, False) when not detectable.
    """
    try:
        if IS_LINUX:
            # Intel RAPL: constraint_0_power_limit_uw (long-term, often = TDP)
            for base in ["/sys/class/powercap/intel-rapl:0", "/sys/class/powercap/intel-rapl:0/intel-rapl:0"]:
                for i in range(2):
                    p = os.path.join(base, f"constraint_{i}_power_limit_uw")
                    if os.path.isfile(p):
                        with open(p, "r") as f:
                            uw = float(f.read().strip())
                        return (round(uw / 1e6, 1), True)
            # AMD RAPL-style (if exists)
            for p in glob.glob("/sys/class/powercap/amd_energy/*/power_limit_uw"):
                try:
                    with open(p, "r") as f:
                        uw = float(f.read().strip())
                    if uw > 0:
                        return (round(uw / 1e6, 1), True)
                except Exception:
                    pass
        return (0.0, False)
    except Exception:
        return (0.0, False)


def get_gpu_power_limit_w() -> Tuple[float, bool]:
    """
    GPU power limit in watts (e.g. nvidia-smi power.limit). Sum if multi-GPU.
    Returns (watts, available). (0.0, False) when not available.
    """
    try:
        smi_path = _find_nvidia_smi_path()
        if smi_path:
            creationflags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
            cmd = [smi_path, "--query-gpu=power.limit", "--format=csv,noheader,nounits"]
            r = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8", creationflags=creationflags)
            total = 0.0
            for line in r.stdout.strip().split("\n"):
                line = line.strip()
                if not line or line == "[N/A]":
                    continue
                total += float(line.replace(",", "."))
            if total > 0:
                return (round(total, 1), True)
        return (0.0, False)
    except Exception:
        return (0.0, False)


def get_gpu_product_name() -> Tuple[str, bool]:
    """
    First GPU product name (e.g. "NVIDIA GeForce RTX 4090"). For TDP lookup when power.limit is N/A.
    Returns (name, available).
    """
    try:
        smi_path = _find_nvidia_smi_path()
        if smi_path:
            creationflags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
            cmd = [smi_path, "--query-gpu=name", "--format=csv,noheader"]
            r = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8", creationflags=creationflags)
            line = (r.stdout.strip().split("\n") or [""])[0].strip()
            if line and line != "[N/A]":
                return (line, True)
        return ("", False)
    except Exception:
        return ("", False)


# TDP (W) for common Nvidia GPUs when power.limit is not reported. Pause=70%, Stop=80% of TDP.
# Ordered by specificity (longer/full names first) for match; first match wins.
_GPU_TDP_W = [
    ("RTX 5090", 575), ("RTX 5080", 320), ("RTX 5070", 220), ("RTX 5060", 130),
    ("RTX 4090", 450), ("RTX 4080", 320), ("RTX 4070", 200), ("RTX 4060", 115), ("RTX 4050", 115),
    ("RTX 3090 Ti", 450), ("RTX 3090", 350), ("RTX 3080 Ti", 350), ("RTX 3080", 320),
    ("RTX 3070 Ti", 290), ("RTX 3070", 220), ("RTX 3060 Ti", 200), ("RTX 3060", 170), ("RTX 3050", 130),
    ("RTX 2080 Ti", 250), ("RTX 2080", 215), ("RTX 2070", 175), ("RTX 2060", 160),
]


def get_gpu_tdp_fallback_w(name: str) -> float:
    """
    Returns TDP in watts for known Nvidia models, or 0 if unknown.
    """
    if not name:
        return 0.0
    u = name.upper()
    for sub, tdp in _GPU_TDP_W:
        if sub.upper() in u:
            return float(tdp)
    return 0.0


def get_battery_status() -> Dict[str, Any]:
    """
    Battery status for laptops. On desktop often present=False.
    Returns: {present, on_battery, percent, charging}.
    """
    out = {"present": False, "on_battery": False, "percent": None, "charging": False}
    try:
        bat = psutil.sensors_battery()
        if bat is not None:
            out["present"] = True
            out["percent"] = int(bat.percent) if bat.percent is not None else None
            out["on_battery"] = bat.power_plugged is False if bat.power_plugged is not None else False
            # Infer charging: plugged and percent < 100, or we need status
            if bat.power_plugged and out["percent"] is not None and out["percent"] < 100:
                out["charging"] = True
        if IS_LINUX and not out["present"]:
            for p in glob.glob("/sys/class/power_supply/BAT*/capacity"):
                try:
                    with open(p, "r") as f:
                        out["percent"] = int(f.read().strip())
                    out["present"] = True
                    st = p.replace("/capacity", "/status")
                    if os.path.exists(st):
                        with open(st, "r") as f:
                            s = f.read().strip().lower()
                        out["on_battery"] = "discharging" in s
                        out["charging"] = "charging" in s
                    break
                except Exception:
                    pass
        return out
    except Exception:
        return out


# --- HARDWARE STRATEGIES ---

def _get_gpu_stats_strategy() -> Tuple[float, float, float, int, int]:
    """
    Dispatches to the correct vendor tool.
    Returns: (total_gb, free_gb, used_gb, util_%, temp_C)
    """
    # 1. Try NVIDIA (nvidia-smi) - Most common for LLMs
    # On Windows, this might need a specific path check
    try:
        return _query_nvidia_smi()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # 2. Try AMD (rocm-smi) - Linux
    if IS_LINUX:
        try:
            return _query_rocm_smi()
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    # 3. Try Apple Silicon (powermetrics) - macOS
    if IS_MACOS:
        try:
            return _query_apple_silicon()
        except Exception:
            pass

    return (0.0, 0.0, 0.0, 0, 0)

def _find_nvidia_smi_path():
    """Helper to find nvidia-smi on Windows if not in PATH."""
    # 1. Check global path
    if shutil.which("nvidia-smi"):
        return "nvidia-smi"
        
    if IS_WINDOWS:
        # 2. Check standard install locations
        common_paths = [
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            r"C:\Windows\System32\nvidia-smi.exe"
        ]
        for p in common_paths:
            if os.path.exists(p):
                return p
    return None

def _query_nvidia_smi() -> Tuple[float, float, float, int, int]:
    """
    Aggregates stats for ALL detected Nvidia GPUs.
    """
    smi_path = _find_nvidia_smi_path()
    if not smi_path:
        raise FileNotFoundError("nvidia-smi not found")

    cmd = [
        smi_path, 
        '--query-gpu=memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu', 
        '--format=csv,noheader,nounits'
    ]
    
    # Creation flags to hide console window on Windows
    creationflags = 0
    if IS_WINDOWS:
        creationflags = subprocess.CREATE_NO_WINDOW

    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        check=True, 
        encoding='utf-8',
        creationflags=creationflags
    )
    
    lines = result.stdout.strip().split('\n')
    
    total_gb = 0.0
    free_gb = 0.0
    used_gb = 0.0
    max_util = 0
    max_temp = 0
    
    for line in lines:
        if not line.strip(): continue
        parts = line.split(',')
        if len(parts) < 5: continue
        
        try:
            # Accumulate Memory
            total_gb += float(parts[0].strip())
            free_gb += float(parts[1].strip())
            used_gb += float(parts[2].strip())
            
            # Take Max for Util/Temp (Safety first)
            util = int(parts[3].strip())
            temp = int(parts[4].strip())
            
            if util > max_util: max_util = util
            if temp > max_temp: max_temp = temp
        except ValueError:
            continue
    
    # Convert MB to GB
    return round(total_gb / 1024, 2), round(free_gb / 1024, 2), round(used_gb / 1024, 2), max_util, max_temp

def _query_rocm_smi() -> Tuple[float, float, float, int, int]:
    """
    Aggregates stats for ALL detected AMD GPUs via rocm-smi JSON.
    """
    cmd = ['rocm-smi', '--showuse', '--showmeminfo', '--showtemp', '--json']
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    total_gb = 0.0
    used_gb = 0.0
    max_util = 0
    max_temp = 0
    
    for card_key, card_data in data.items():
        # Parse VRAM (Usually bytes)
        vram_total_bytes = int(card_data.get("VRAM Total Memory (B)", 0))
        vram_used_bytes = int(card_data.get("VRAM Total Used Memory (B)", 0))
        
        total_gb += vram_total_bytes
        used_gb += vram_used_bytes
        
        # Parse Utilization
        util_str = card_data.get("GPU use (%)", "0").replace("%", "")
        util = int(float(util_str))
        if util > max_util: max_util = util
        
        # Parse Temp
        temp_str = card_data.get("Temperature (Sensor #1) (C)", "0.0").replace("c", "").replace("C", "")
        temp = int(float(temp_str))
        if temp > max_temp: max_temp = temp
    
    # Convert Bytes to GB
    total_gb = round(total_gb / (1024**3), 2)
    used_gb = round(used_gb / (1024**3), 2)
    free_gb = total_gb - used_gb
    
    return total_gb, free_gb, used_gb, max_util, max_temp

def _query_apple_silicon() -> Tuple[float, float, float, int, int]:
    """
    Uses 'powermetrics' to get GPU utilization and shared memory.
    Note: M-series chips usually have Unified Memory, so 'multi-gpu' isn't standard here.
    """
    sys_mem = psutil.virtual_memory()
    total_gb = round(sys_mem.total / (1024**3), 2)
    used_gb = round(sys_mem.used / (1024**3), 2) 
    free_gb = round(sys_mem.available / (1024**3), 2)
    
    util = 0
    temp = 0
    
    try:
        cmd = ["sudo", "powermetrics", "--samplers", "gpu_power", "-i", "200", "-n", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1)
        output = result.stdout
        
        match = re.search(r"GPU active residency:\s+([\d\.]+)%", output)
        if match:
            util = int(float(match.group(1)))
    except Exception:
        pass

    return total_gb, free_gb, used_gb, util, temp

if __name__ == "__main__":
    # Self-test
    t_sys, a_sys = get_system_memory()
    print(f"System RAM: {a_sys}/{t_sys} GB")
    
    t, f, u, util, temp = get_gpu_vitals()
    print(f"GPU Vitals: {util}% Util, {temp}°C, {u}/{t} GB VRAM")