"""
watchdog.py — Hardware health monitor
Runs in a background thread and checks:
  1. USB microphone is still streaming (via last-read timestamp)
  2. Ollama is still reachable
Calls speak_callback with a warning if something dies.
"""

import logging
import threading
import time
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)


class Watchdog:
    def __init__(
        self,
        config: dict,
        get_mic_time: Callable[[], float],
        speak: Callable[[str], None],
    ):
        self.cfg = config["watchdog"]
        self.ollama_host = config["ollama"]["host"]
        self._get_mic_time = get_mic_time
        self._speak = speak
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mic_warned = False
        self._ollama_warned = False
        self._temp_warned = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if not self.cfg.get("enabled", True):
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="watchdog")
        self._thread.start()
        logger.info("Watchdog started.")

    def stop(self):
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self):
        interval = self.cfg.get("check_interval_sec", 5)
        mic_timeout = self.cfg.get("mic_timeout_sec", 10)
        mic_check_enabled = self.cfg.get("mic_check_enabled", True)
        temp_limit = self.cfg.get("temp_limit_celsius", 80.0)

        while not self._stop_event.is_set():
            self._stop_event.wait(interval)
            if self._stop_event.is_set():
                break

            # ── Mic check ────────────────────────────────────────────
            if mic_check_enabled:
                last = self._get_mic_time()
                if last > 0:  # only check after first read
                    age = time.monotonic() - last
                    if age > mic_timeout and not self._mic_warned:
                        logger.warning("Watchdog: mic stream stalled (%.1f s since last read)", age)
                        self._speak("Warning: I can't hear the microphone. Please check the USB connection.")
                        self._mic_warned = True
                    elif age <= mic_timeout:
                        self._mic_warned = False

            # ── Ollama check ─────────────────────────────────────────
            try:
                r = requests.get(f"{self.ollama_host}/api/tags", timeout=2)
                if r.status_code == 200 and self._ollama_warned:
                    logger.info("Watchdog: Ollama is back online.")
                    self._speak("My brain is back online.")
                    self._ollama_warned = False
            except requests.exceptions.RequestException:
                if not self._ollama_warned:
                    logger.warning("Watchdog: Ollama unreachable.")
                    self._speak("Warning: I've lost connection to my brain. I'll let you know when it's back.")
                    self._ollama_warned = True

            # ── Thermal check ────────────────────────────────────────
            temp = self._get_cpu_temp()
            if temp and temp >= temp_limit and not self._temp_warned:
                logger.warning("Watchdog: CPU temperature critical: %.1f °C", temp)
                self._speak(f"Warning: My processor is getting very hot at {temp:.0f} degrees. I might slow down soon.")
                self._temp_warned = True
            elif temp and temp < temp_limit - 5:
                self._temp_warned = False

    def _get_cpu_temp(self) -> Optional[float]:
        """Try multiple methods to get system temperature (consistent with SystemMonitorTool)."""
        try:
            # Method 1: /sys/class/thermal (standard Linux)
            thermal_path = "/sys/class/thermal/thermal_zone0/temp"
            if os.path.exists(thermal_path):
                with open(thermal_path, "r") as f:
                    return float(f.read().strip()) / 1000.0
            
            # Method 2: vcgencmd (Raspberry Pi)
            import subprocess
            try:
                result = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=1)
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if "temp=" in output:
                        return float(output.split("=")[1].split("'")[0])
            except Exception:
                pass
            
            # Method 3: psutil
            import psutil
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            return float(entries[0].current)
        except Exception:
            pass
        return None
