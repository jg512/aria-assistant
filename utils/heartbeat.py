"""
heartbeat.py — Scheduled tasks and morning briefing
Runs a background thread that fires at configured times.
Components are pluggable — only runs what's available and configured.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class Heartbeat:
    def __init__(
        self,
        config: dict,
        speak: Callable[[str], None],
        tools: dict,
    ):
        self.cfg = config.get("heartbeat", {})
        self._speak = speak
        self._tools = tools
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._briefing_done_date: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if not self.cfg.get("enabled", True):
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="heartbeat")
        self._thread.start()
        logger.info("Heartbeat started. Morning briefing at %s.",
                    self.cfg.get("morning_briefing_time", "08:00"))

    def stop(self):
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(30)   # check every 30 seconds
            if self._stop_event.is_set():
                break
            self._check_morning_briefing()

    def _check_morning_briefing(self):
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        briefing_time = self.cfg.get("morning_briefing_time", "08:00")

        try:
            target_hour, target_min = map(int, briefing_time.split(":"))
            target_dt = now.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
        except ValueError:
            return

        # Fire if:
        # 1. It's currently past the target briefing time
        # 2. We haven't successfully delivered it today
        if now >= target_dt and self._briefing_done_date != today:
            # Check if it's too late (don't briefing if we just turned on at 11 PM)
            # Max delay: 4 hours after scheduled time
            if now < target_dt + timedelta(hours=4):
                self._briefing_done_date = today
                self._deliver_briefing(now)
            else:
                # It's too late in the day, just mark it done so it doesn't fire
                self._briefing_done_date = today
                logger.info("Missed morning briefing window (it is now %s).", now.strftime("%H:%M"))

    def _deliver_briefing(self, now: datetime):
        logger.info("Delivering morning briefing.")
        parts = []
        components = self.cfg.get("briefing_components", ["time", "weather", "calendar", "todo"])

        if "time" in components:
            hour = now.strftime("%I").lstrip("0") or "12"
            period = now.strftime("%p")
            day = now.strftime("%A, %B %-d")
            parts.append(f"Good morning! It's {hour} {period} on {day}.")

        if "weather" in components and "weather" in self._tools:
            try:
                weather_summary = self._tools["weather"].run(action="current")
                parts.append(weather_summary)
            except Exception as e:
                logger.debug("Briefing: weather error: %s", e)

        if "calendar" in components and "calendar" in self._tools:
            try:
                cal_summary = self._tools["calendar"].run(action="today")
                parts.append(cal_summary)
            except Exception as e:
                logger.debug("Briefing: calendar error: %s", e)

        if "todo" in components and "todo" in self._tools:
            try:
                todo_summary = self._tools["todo"].run(action="summary")
                parts.append(todo_summary)
            except Exception as e:
                logger.debug("Briefing: todo error: %s", e)

        if parts:
            briefing_text = " ".join(parts)
            self._speak(briefing_text)
