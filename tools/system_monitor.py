"""
system_monitor.py — System status and monitoring
Reports CPU, memory, disk usage, and system temperature.
"""

import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


class SystemMonitorTool:
    NAME = "system_monitor"
    DESCRIPTION = "System status: status, cpu_usage, memory_usage, disk_usage, temperature"
    PARAMETERS = {
        "action": {
            "type": "string",
            "enum": ["status", "cpu_usage", "memory_usage", "disk_usage", "temperature"],
        },
    }

    def __init__(self, config: dict):
        self.cfg = config
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            logger.warning("psutil not installed — system monitoring disabled")
            self.psutil = None

    def run(self, action="status", **_) -> str:
        if not self.psutil:
            return "System monitoring not available (install psutil)."
        
        if action == "status":
            return self._full_status()
        elif action == "cpu_usage":
            return self._cpu_usage()
        elif action == "memory_usage":
            return self._memory_usage()
        elif action == "disk_usage":
            return self._disk_usage()
        elif action == "temperature":
            return self._temperature()
        return "Unknown action."

    def _full_status(self) -> str:
        try:
            cpu = self.psutil.cpu_percent(interval=0.1)
            mem = self.psutil.virtual_memory()
            disk = self.psutil.disk_usage("/")
            
            result = f"System status: CPU at {cpu} percent, "
            result += f"memory at {mem.percent} percent, "
            result += f"disk at {disk.percent} percent."
            
            return result
        except Exception as e:
            logger.error("Failed to get system status: %s", e)
            return f"Failed to get system status: {e}"

    def _cpu_usage(self) -> str:
        try:
            cpu = self.psutil.cpu_percent(interval=0.2)
            return f"CPU usage is {cpu} percent."
        except Exception as e:
            logger.error("Failed to get CPU usage: %s", e)
            return f"Failed to get CPU usage: {e}"

    def _memory_usage(self) -> str:
        try:
            mem = self.psutil.virtual_memory()
            used_gb = mem.used / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            return f"Memory usage: {used_gb:.1f} gigabytes of {total_gb:.1f} gigabytes used. That's {mem.percent} percent."
        except Exception as e:
            logger.error("Failed to get memory usage: %s", e)
            return f"Failed to get memory usage: {e}"

    def _disk_usage(self) -> str:
        try:
            disk = self.psutil.disk_usage("/")
            free_gb = disk.free / (1024 ** 3)
            total_gb = disk.total / (1024 ** 3)
            return f"Disk usage: {free_gb:.1f} gigabytes free of {total_gb:.1f} gigabytes total. That's {100 - disk.percent} percent free."
        except Exception as e:
            logger.error("Failed to get disk usage: %s", e)
            return f"Failed to get disk usage: {e}"

    def _temperature(self) -> str:
        """Try multiple methods to get system temperature."""
        try:
            # Method 1: psutil (Linux only)
            if hasattr(self.psutil, "sensors_temperatures"):
                temps = self.psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            temp = entries[0].current
                            return f"System temperature is {temp:.1f} degrees Celsius."
            
            # Method 2: /sys/class/thermal (Linux)
            thermal_path = "/sys/class/thermal/thermal_zone0/temp"
            if os.path.exists(thermal_path):
                with open(thermal_path, "r") as f:
                    temp_millidegrees = int(f.read().strip())
                    temp = temp_millidegrees / 1000
                    return f"System temperature is {temp:.1f} degrees Celsius."
            
            # Method 3: vcgencmd (Raspberry Pi)
            try:
                result = subprocess.run(
                    ["vcgencmd", "measure_temp"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    # Parse "temp=XX.X'C"
                    output = result.stdout.strip()
                    if "temp=" in output:
                        temp_str = output.split("=")[1].split("'")[0]
                        return f"Temperature is {temp_str} degrees Celsius."
            except Exception:
                pass
            
            return "Temperature sensor not available on this system."
        except Exception as e:
            logger.error("Failed to get temperature: %s", e)
            return "Could not read system temperature."
