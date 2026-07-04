# src/utils/system_utils.py
import platform
import psutil
import subprocess
import json
import hashlib
import os
import sys

# No external src.* imports required for this specific file based on previous context,
# but we ensure it is ready for the new structure.

class SystemScanner:
    def __init__(self):
        self.os_type = platform.system()
        self.info = {
            "os": self._get_os_info(),
            "cpu": self._get_cpu_info(),
            "ram": self._get_ram_info(),
            "gpus": self._get_gpu_info()
        }
        self.system_id = self._generate_hardware_hash()

    @staticmethod
    def open_file_explorer(path: str):
        """
        Opens the file explorer and selects the file if possible.
        """
        folder_path = os.path.dirname(path)
        
        try:
            if platform.system() == "Windows":
                subprocess.run(['explorer', '/select,', path])
            elif platform.system() == "Darwin":
                subprocess.run(['open', '-R', path])
            else:
                # Linux (xdg-open opens the folder, selecting is distro-specific)
                subprocess.run(['xdg-open', folder_path])
        except Exception as e:
            print(f"Error opening file explorer: {e}")

    def _get_os_info(self):
        try:
            if self.os_type == "Linux":
                try:
                    with open("/etc/os-release") as f:
                        lines = f.readlines()
                        name = next((line.split("=")[1].strip().strip('"') for line in lines if line.startswith("PRETTY_NAME")), "Linux")
                        return {"name": name, "family": "Linux", "release": platform.release()}
                except:
                    pass
            return {
                "name": f"{self.os_type} {platform.release()}",
                "family": self.os_type,
                "version": platform.version()
            }
        except Exception:
            return {"name": "Unknown OS", "family": "Unknown"}

    def _get_cpu_info(self):
        info = {
            "model": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "frequency_mhz": "N/A"
        }
        try:
            if self.os_type == "Windows":
                cmd = "wmic cpu get name"
                res = subprocess.check_output(cmd, shell=True).decode().strip().split('\n')
                if len(res) > 1: info['model'] = res[1].strip()
            elif self.os_type == "Darwin":
                cmd = "sysctl -n machdep.cpu.brand_string"
                res = subprocess.check_output(cmd, shell=True).decode().strip()
                info['model'] = res
            elif self.os_type == "Linux":
                cmd = "cat /proc/cpuinfo"
                res = subprocess.check_output(cmd, shell=True).decode()
                for line in res.split('\n'):
                    if "model name" in line:
                        info['model'] = line.split(":")[1].strip()
                        break
            freq = psutil.cpu_freq()
            if freq:
                info['frequency_mhz'] = round(freq.max, 2) if freq.max > 0 else round(freq.current, 2)
        except Exception as e:
            print(f"Warning: CPU scan partial failure: {e}")
        return info

    def _get_ram_info(self):
        try:
            mem = psutil.virtual_memory()
            total_gb = round(mem.total / (1024**3), 2)
            ram_type, speed_mhz = self._detect_ram_type_and_speed()
            speed_display = self._format_ram_speed_display(ram_type, speed_mhz)
            result = {
                "capacity_gb": total_gb,
                "percent_used": mem.percent,
                "speed": speed_display
            }
            if ram_type:
                result["type"] = ram_type
            if speed_mhz is not None:
                result["speed_mhz"] = speed_mhz
            return result
        except Exception:
            return {"capacity_gb": 0, "percent_used": 0, "speed": "Unknown"}

    def _detect_ram_type_and_speed(self):
        """Returns (type_str, speed_mhz) or (None, None) if detection fails."""
        try:
            if self.os_type == "Windows":
                return self._detect_ram_windows()
            elif self.os_type == "Darwin":
                return self._detect_ram_macos()
            elif self.os_type == "Linux":
                return self._detect_ram_linux()
        except Exception:
            pass
        return (None, None)

    def _detect_ram_windows(self):
        """Windows: WMI Win32_PhysicalMemory via PowerShell."""
        try:
            ps_script = (
                "Get-WmiObject Win32_PhysicalMemory | "
                "Select-Object -First 1 Speed, SMBIOSMemoryType | "
                "ForEach-Object { $_.Speed.ToString() + '|' + $_.SMBIOSMemoryType.ToString() }"
            )
            res = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps_script],
                encoding="utf-8",
                timeout=10,
            ).strip()
            if not res or "|" not in res:
                return (None, None)
            speed_str, type_code_str = res.split("|", 1)
            speed_mhz = int(speed_str) if speed_str.isdigit() else None
            if speed_mhz == 0:
                speed_mhz = None
            type_code = int(type_code_str) if type_code_str.isdigit() else None
            ram_type = self._smbios_type_to_name(type_code) if type_code else None
            return (ram_type, speed_mhz)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            return (None, None)

    def _smbios_type_to_name(self, code):
        """Map SMBIOS memory type code to DDR name."""
        mapping = {
            20: "DDR",
            21: "DDR2",
            22: "DDR2 FB-DIMM",
            24: "DDR3",
            26: "DDR4",
            27: "DDR5",
        }
        return mapping.get(code, f"DDR{code}" if code and code > 27 else None)

    def _detect_ram_macos(self):
        """macOS: system_profiler SPMemoryDataType -json."""
        try:
            res = subprocess.check_output(
                ["system_profiler", "-json", "SPMemoryDataType"],
                encoding="utf-8",
                timeout=15,
            )
            data = json.loads(res)
            items = data.get("SPMemoryDataType", [])
            if not isinstance(items, list):
                items = [items] if items else []
            ram_type = None
            speed_mhz = None
            for item in items:
                if not isinstance(item, dict):
                    continue
                # Type: try various keys (spdimm_type, dimm_type, _name, etc.)
                for key in ("spdimm_type", "dimm_type", "_name", "type"):
                    type_str = item.get(key)
                    if isinstance(type_str, str) and ("DDR" in type_str or "LPDDR" in type_str):
                        ram_type = type_str.strip()
                        break
                # Speed: try various keys (spdimm_speed, dimm_speed, speed, etc.)
                for key in ("spdimm_speed", "dimm_speed", "speed"):
                    speed_val = item.get(key)
                    if isinstance(speed_val, str) and "MHz" in speed_val:
                        try:
                            speed_mhz = int("".join(c for c in speed_val if c.isdigit()))
                            break
                        except ValueError:
                            pass
                    elif isinstance(speed_val, (int, float)) and speed_val > 0:
                        speed_mhz = int(speed_val)
                        break
                if ram_type or speed_mhz:
                    break
            return (ram_type, speed_mhz)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            return (None, None)

    def _detect_ram_linux(self):
        """Linux: dmidecode -t memory (may require root)."""
        try:
            res = subprocess.check_output(
                ["dmidecode", "-t", "memory"],
                encoding="utf-8",
                timeout=10,
                stderr=subprocess.DEVNULL,
            )
            ram_type = None
            speed_mhz = None
            for line in res.splitlines():
                line = line.strip()
                if line.startswith("Type:"):
                    val = line.split(":", 1)[1].strip()
                    if val and val != "Unknown":
                        ram_type = val
                elif line.startswith("Speed:"):
                    # e.g. "Speed: 3600 MHz" or "Speed: 3600 MT/s"
                    try:
                        digits = "".join(c for c in line if c.isdigit())
                        if digits:
                            speed_mhz = int(digits)
                    except ValueError:
                        pass
            return (ram_type, speed_mhz)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            return (None, None)

    def _format_ram_speed_display(self, ram_type, speed_mhz):
        """Build display string e.g. 'DDR4 3600 MHz' or 'Unknown'."""
        if speed_mhz is not None and speed_mhz > 0:
            type_part = f"{ram_type} " if ram_type else ""
            return f"{type_part}{speed_mhz} MHz"
        if ram_type:
            return ram_type
        return "Unknown"

    def _get_gpu_info(self):
        gpus = []
        # 1. Try NVIDIA-SMI
        try:
            cmd = ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits']
            res = subprocess.check_output(cmd, encoding='utf-8')
            for line in res.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 2:
                    gpus.append({
                        "make": "Nvidia",
                        "model": parts[0].strip(),
                        "vram_gb": round(int(parts[1].strip()) / 1024, 2),
                        "driver": parts[2].strip() if len(parts) > 2 else "N/A"
                    })
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # 2. Try AMD ROCm-SMI
        if not gpus and self.os_type == "Linux":
            try:
                cmd = ['rocm-smi', '--showid', '--showproductname', '--showmeminfo', 'vram', '--json']
                res = subprocess.check_output(cmd, encoding='utf-8')
                data = json.loads(res)
                for card_key, card_val in data.items():
                    model = card_val.get("Card Series", card_val.get("Market Name", "Unknown AMD GPU"))
                    vram_bytes = int(card_val.get("VRAM Total Memory (B)", 0))
                    gpus.append({
                        "make": "AMD",
                        "model": model,
                        "vram_gb": round(vram_bytes / (1024**3), 2),
                        "driver": "ROCm"
                    })
            except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
                pass

        # 3. Try Apple Silicon
        if not gpus and self.os_type == "Darwin":
            try:
                cmd = ["system_profiler", "SPDisplaysDataType", "-json"]
                res = subprocess.check_output(cmd, encoding='utf-8')
                data = json.loads(res)
                items = data.get('SPDisplaysDataType', [])
                for item in items:
                    model = item.get('sppci_model', 'Unknown Apple GPU')
                    vram_gb = 0.0
                    if 'spdisplays_vram' in item:
                        vram_str = item['spdisplays_vram']
                        if "GB" in vram_str:
                            vram_gb = float(vram_str.replace("GB", "").strip())
                    
                    if "Apple" in model and vram_gb == 0:
                        mem = psutil.virtual_memory()
                        vram_gb = round(mem.total / (1024**3), 2)

                    gpus.append({
                        "make": "Apple",
                        "model": model,
                        "vram_gb": vram_gb,
                        "driver": "CoreGraphics"
                    })
            except Exception:
                pass

        return gpus

    def _generate_hardware_hash(self):
        ram_rounded = int(self.info['ram']['capacity_gb'])
        cpu_model = self.info['cpu']['model']
        os_name = self.info['os']['name']
        sorted_gpus = sorted(self.info['gpus'], key=lambda x: x['model'])
        gpu_str = f"Count:{len(sorted_gpus)}|" 
        for g in sorted_gpus:
            gpu_str += f"{g['make']}-{g['model']}-{g['vram_gb']}GB|"
        fingerprint_str = f"{os_name}|{cpu_model}|{ram_rounded}|{gpu_str}"
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

    def get_fuzzy_signature(self):
        """
        Generates a stable identifier based on physical counts rather than strings.
        Used to detect the same machine after driver updates.
        Format: "Cores:X|RAM:Y|VRAM:Z"
        """
        cores = self.info['cpu']['physical_cores']
        
        # Round RAM to nearest 4GB to handle minor OS reporting differences
        ram_raw = self.info['ram']['capacity_gb']
        ram_bucket = round(ram_raw / 4) * 4
        
        # Total VRAM across all cards
        vram_total = sum(g['vram_gb'] for g in self.info['gpus'])
        vram_bucket = round(vram_total)
        
        return f"Cores:{cores}|RAM:{ram_bucket}|VRAM:{vram_bucket}"