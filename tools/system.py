"""
tools/system.py — System utilities: time, date, timer, persistent alarms, shutdown
Alarms survive reboots via JSON storage.

Enhancements:
- Recurring alarms: daily, weekdays, weekends, or specific days ("every monday 07:00")
- Custom alarm sounds per alarm
- Real snooze: refire after N minutes instead of just clearing
"""

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Day-name → weekday number (Monday=0 … Sunday=6)
_DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


class SystemTool:
    NAME = "system"
    DESCRIPTION = (
        "System actions: get_time, get_date, set_timer [seconds], "
        "set_alarm [time_str HH:MM, recurrence?, sound?], "
        "list_alarms, cancel_alarm [time_str], "
        "dismiss_alarm, snooze_alarm [snooze_minutes], "
        "set_briefing_time, shutdown, reboot."
    )
    PARAMETERS = {
        "action": {
            "type": "string",
            "enum": [
                "get_time", "get_date", "set_timer", "set_alarm",
                "list_alarms", "cancel_alarm", "dismiss_alarm", "snooze_alarm",
                "set_briefing_time", "shutdown", "reboot",
            ],
        },
        "seconds":         {"type": "integer"},
        "time_str":        {"type": "string",  "description": "HH:MM (24h) format."},
        "recurrence":      {
            "type": "string",
            "description": (
                "Recurrence rule. One of: 'daily', 'weekdays', 'weekends', "
                "'every monday' … 'every sunday', or omit for one-shot."
            ),
        },
        "sound":           {
            "type": "string",
            "description": "Path to a custom alarm sound file (wav/mp3). Uses system default if omitted.",
        },
        "snooze_minutes":  {"type": "integer", "description": "Minutes to snooze (1–60)."},
    }

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: dict,
        speak_callback: Optional[Callable[[str], None]] = None,
        confirm_callback: Optional[Callable[[str], bool]] = None,
    ):
        self.cfg = config
        self._speak  = speak_callback  or (lambda t: None)
        self._confirm = confirm_callback
        self._shutdown_confirmation = config["assistant"].get("shutdown_confirmation", True)

        alarms_cfg       = config.get("alarms", {})
        self._alarms_path = Path(alarms_cfg.get("file_path", "/tmp/aria-alarms.json"))
        self._default_sound: Optional[str] = alarms_cfg.get("default_sound")   # may be None
        self._alarms: list[dict] = []
        self._lock = threading.Lock()   # SHARED LOCK for all alarm data access
        self._load_alarms()

        self._timer_thread: Optional[threading.Thread] = None

        # Active-alarm state (protected by separate alarm_lock for sounding state)
        self._alarm_lock    = threading.Lock()
        self._active_alarm: Optional[dict] = None   # the full alarm dict that fired
        self._active_sound_proc: Optional[subprocess.Popen] = None
        self._snooze_thread: Optional[threading.Thread] = None

        self._alarm_thread = threading.Thread(
            target=self._alarm_loop, daemon=True, name="alarm-loop"
        )
        self._alarm_thread.start()

    def reload_config(self, config: dict):
        self.cfg = config
        self._shutdown_confirmation = config["assistant"].get("shutdown_confirmation", True)
        alarms_cfg = config.get("alarms", {})
        self._default_sound = alarms_cfg.get("default_sound")
        logger.info("SystemTool: config reloaded.")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> str:
        with self._lock:
            n = len(self._alarms)
            if n == 0:
                return "no alarms set"
            times = ", ".join(self._describe_alarm(a) for a in self._alarms)
        return f"{n} alarm(s): {times}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        action: str,
        seconds: int = 0,
        time_str: str = "",
        recurrence: str = "",
        sound: str = "",
        snooze_minutes: int = 10,
        **_,
    ) -> str:
        action = action.lower().strip()
        if action == "get_time":
            return self._get_time()
        elif action == "get_date":
            return self._get_date()
        elif action == "set_timer":
            return self._set_timer(seconds)
        elif action == "set_alarm":
            return self._set_alarm(time_str, recurrence, sound)
        elif action == "list_alarms":
            return self._list_alarms()
        elif action == "cancel_alarm":
            return self._cancel_alarm(time_str)
        elif action == "dismiss_alarm":
            return self._dismiss_alarm()
        elif action == "snooze_alarm":
            return self._snooze_alarm(snooze_minutes)
        elif action == "set_briefing_time":
            return self._set_briefing_time(time_str)
        elif action == "shutdown":
            return self._shutdown()
        elif action == "reboot":
            return self._reboot()
        return f"Unknown system action: {action}"

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _get_time(self) -> str:
        now = datetime.now()
        hour   = now.strftime("%I").lstrip("0") or "12"
        minute = now.strftime("%M")
        period = now.strftime("%p")
        if minute == "00":
            return f"It's {hour} {period}."
        return f"It's {hour} {minute} {period}."

    def _get_date(self) -> str:
        return datetime.now().strftime("Today is %A, %B %-d.")

    # ---- Timer -------------------------------------------------------

    def _set_timer(self, seconds: int) -> str:
        if seconds <= 0:
            return "Please give me a valid number of seconds."
        if self._timer_thread and self._timer_thread.is_alive():
            return "A timer is already running."

        def _run():
            time.sleep(seconds)
            self._speak(f"Timer done! {self._seconds_to_words(seconds)} are up.")

        self._timer_thread = threading.Thread(target=_run, daemon=True)
        self._timer_thread.start()
        return f"Timer set for {self._seconds_to_words(seconds)}."

    # ---- Alarms ------------------------------------------------------

    def _set_alarm(self, time_str: str, recurrence: str = "", sound: str = "") -> str:
        """
        Create an alarm entry.

        Alarm dict schema:
          {
            "time":       "HH:MM",
            "recurrence": "daily" | "weekdays" | "weekends"
                          | "monday" … "sunday"   (specific weekday)
                          | ""                    (one-shot)
            "sound":      "/path/to/sound.wav"    (or "" for default)
          }
        """
        try:
            datetime.strptime(time_str.strip(), "%H:%M")
        except ValueError:
            return "Please give the time as HH:MM, for example 07:30."

        rec = self._parse_recurrence(recurrence)

        with self._lock:
            # Avoid exact duplicates (same time + same recurrence)
            for a in self._alarms:
                if a["time"] == time_str and a.get("recurrence", "") == rec:
                    return f"There's already an alarm set for {time_str} ({rec or 'one-shot'})."
            alarm = {
                "time":       time_str.strip(),
                "recurrence": rec,
                "sound":      sound.strip() if sound else "",
            }
            self._alarms.append(alarm)
            self._save_alarms()

        target = datetime.strptime(time_str.strip(), "%H:%M")
        hour   = target.strftime("%I").lstrip("0") or "12"
        minute = target.strftime("%M")
        period = target.strftime("%p")
        rec_msg = f" ({rec})" if rec else ""
        return f"Alarm set for {hour}:{minute} {period}{rec_msg}."

    def _parse_recurrence(self, text: str) -> str:
        """Normalise recurrence input → canonical string stored in JSON."""
        t = text.lower().strip()
        if not t:
            return ""
        if t in ("daily", "every day"):
            return "daily"
        if t in ("weekdays", "weekday", "every weekday"):
            return "weekdays"
        if t in ("weekends", "weekend", "every weekend"):
            return "weekends"
        for day_name in _DAY_NAMES:
            if day_name in t:
                return day_name   # e.g. "monday"
        return ""   # unrecognised → treat as one-shot

    def _alarm_should_fire_today(self, alarm: dict, now: datetime) -> bool:
        """Return True if this alarm is scheduled to fire on today's weekday."""
        rec = alarm.get("recurrence", "")
        wd  = now.weekday()   # 0=Monday … 6=Sunday
        if not rec:
            return True   # one-shot: always eligible (fired_today prevents repeat)
        if rec == "daily":
            return True
        if rec == "weekdays":
            return wd < 5
        if rec == "weekends":
            return wd >= 5
        if rec in _DAY_NAMES:
            return _DAY_NAMES[rec] == wd
        return False

    def _list_alarms(self) -> str:
        if not self._alarms:
            return "You have no alarms set."
        descriptions = [self._describe_alarm(a) for a in self._alarms]
        if len(descriptions) == 1:
            return f"You have one alarm: {descriptions[0]}."
        return "Your alarms: " + ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}."

    def _describe_alarm(self, alarm: dict) -> str:
        rec = alarm.get("recurrence", "")
        t   = alarm["time"]
        snd = alarm.get("sound", "")
        desc = f"{t}"
        if rec:
            desc += f" ({rec})"
        if snd:
            desc += f" [sound: {Path(snd).name}]"
        return desc

    def _cancel_alarm(self, time_str: str) -> str:
        before = len(self._alarms)
        self._alarms = [
            a for a in self._alarms if a["time"] != time_str.strip()
        ]
        if len(self._alarms) < before:
            self._save_alarms()
            return f"Alarm for {time_str} cancelled."
        return f"I don't have an alarm set for {time_str}."

    def _snooze_alarm(self, snooze_minutes: int = 10) -> str:
        """
        Snooze the active alarm by setting 'snoozed_until' and clearing its 
        'active' state. The background loop will refire it later.
        """
        snooze_minutes = max(1, min(60, snooze_minutes))

        with self._alarm_lock:
            if self._active_alarm is None:
                return "No alarm is currently sounding."
            
            # Snap the alarm info
            snoozed_alarm_time = self._active_alarm["time"]
            snoozed_alarm_rec = self._active_alarm.get("recurrence", "")
            self._active_alarm = None

        # Update the alarm in the main list so it persists
        with self._lock:
            refire_time = (datetime.now() + timedelta(minutes=snooze_minutes)).isoformat()
            for a in self._alarms:
                if a["time"] == snoozed_alarm_time and a.get("recurrence", "") == snoozed_alarm_rec:
                    a["snoozed_until"] = refire_time
                    break
            self._save_alarms()

        return f"Snoozed for {snooze_minutes} minute{'s' if snooze_minutes != 1 else ''}."

    def _dismiss_alarm(self) -> str:
        """Dismiss the active alarm and cleanup one-shot alarms."""
        with self._alarm_lock:
            if self._active_alarm is None:
                return "No alarm is currently sounding."
            
            # Stop the sound process
            if self._active_sound_proc and self._active_sound_proc.poll() is None:
                self._active_sound_proc.terminate()
                self._active_sound_proc = None

            dismissed_alarm = self._active_alarm
            self._active_alarm = None

        # If it's a one-shot alarm, remove it now that it's finally dismissed
        if not dismissed_alarm.get("recurrence"):
            with self._lock:
                self._alarms = [
                    a for a in self._alarms 
                    if not (a["time"] == dismissed_alarm["time"] and not a.get("recurrence"))
                ]
                self._save_alarms()
        
        return "Alarm dismissed."

    # ---- Briefing time -----------------------------------------------

    def _set_briefing_time(self, time_str: str) -> str:
        if not time_str or ":" not in time_str:
            return "Invalid time format. Use HH:MM."
        try:
            parts  = time_str.split(":")
            hour   = int(parts[0])
            minute = int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return "Time must be 00:00 to 23:59."

            config_path = Path("./config.json")
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                config["heartbeat"]["morning_briefing_time"] = time_str
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                logger.info("Briefing time set to %s", time_str)
                hour_12  = hour % 12 or 12
                ampm     = "AM" if hour < 12 else "PM"
                return f"Briefing time set to {hour_12}:{minute:02d} {ampm}."
            return "Config file not found."
        except Exception as e:
            logger.error("Failed to set briefing time: %s", e)
            return f"Failed to set briefing time: {e}"

    # ---- Power -------------------------------------------------------

    def _shutdown(self) -> str:
        if self._shutdown_confirmation and self._confirm:
            if not self._confirm("Are you sure you want to shut down?"):
                return "Shutdown cancelled."
        subprocess.Popen(["sudo", "shutdown", "-h", "now"])
        return "Shutting down. Goodbye!"

    def _reboot(self) -> str:
        if self._shutdown_confirmation and self._confirm:
            if not self._confirm("Are you sure you want to reboot?"):
                return "Reboot cancelled."
        subprocess.Popen(["sudo", "reboot"])
        return "Rebooting now. Be right back!"

    # ------------------------------------------------------------------
    # Alarm loop (background thread)
    # ------------------------------------------------------------------

    def _alarm_loop(self):
        """Check alarms every 30 seconds and fire when time matches (regular or snooze)."""
        fired_today: set[str] = set()   # keys: "HH:MM|recurrence"
        last_date = datetime.now().date()

        while True:
            time.sleep(30)
            now = datetime.now()

            # Reset fired set at midnight
            if now.date() != last_date:
                fired_today.clear()
                last_date = now.date()

            # Copy list to avoid concurrent modification errors
            for alarm in list(self._alarms):
                fire_key = f"{alarm['time']}|{alarm.get('recurrence', '')}"
                
                # Check if it's currently active (don't refire an already-sounding alarm)
                with self._alarm_lock:
                    if self._active_alarm is alarm:
                        continue

                should_fire = False
                
                # 1. Check persistent snooze
                snooze_until_str = alarm.get("snoozed_until")
                if snooze_until_str:
                    snooze_dt = datetime.fromisoformat(snooze_until_str)
                    if now >= snooze_dt:
                        should_fire = True
                        # Clear the snooze field once triggered
                        with self._lock:
                            alarm.pop("snoozed_until", None)
                            self._save_alarms()

                # 2. Check regular schedule (if not already fired today)
                if not should_fire and fire_key not in fired_today:
                    if self._alarm_should_fire_today(alarm, now):
                        try:
                            t = datetime.strptime(alarm["time"], "%H:%M")
                            if now.hour == t.hour and now.minute == t.minute:
                                should_fire = True
                                fired_today.add(fire_key)
                        except ValueError:
                            pass

                if should_fire:
                    with self._alarm_lock:
                        self._active_alarm = alarm
                    
                    self._play_alarm_sound(alarm)
                    
                    # Convert time to spoken format
                    t_orig = datetime.strptime(alarm["time"], "%H:%M")
                    hour   = t_orig.strftime("%I").lstrip("0") or "12"
                    period = t_orig.strftime("%p")
                    
                    prefix = "Snooze over! " if snooze_until_str else ""
                    self._speak(
                        f"{prefix}Alarm! It's {hour} {t_orig.strftime('%M')} {period}."
                    )
                    # Note: One-shot alarms are now removed in _dismiss_alarm,
                    # not here, so they survive crashes/restarts during snooze.

    # ------------------------------------------------------------------
    # Sound playback
    # ------------------------------------------------------------------

    def _play_alarm_sound(self, alarm: dict):
        """
        Play the alarm sound.
        Priority: alarm-specific sound → config default → system beep.
        Uses aplay for .wav, mpg123 for .mp3, paplay as fallback.
        """
        # Stop any existing alarm sound first
        with self._alarm_lock:
            if self._active_sound_proc and self._active_sound_proc.poll() is None:
                self._active_sound_proc.terminate()
                self._active_sound_proc = None

        sound_path = alarm.get("sound") or self._default_sound
        proc = None
        if sound_path:
            path = Path(sound_path)
            if path.exists():
                ext = path.suffix.lower()
                if ext == ".mp3":
                    player = ["mpg123", "-q", str(path)]
                elif ext in (".wav", ".wave"):
                    player = ["aplay", "-q", str(path)]
                else:
                    player = ["paplay", str(path)]
                try:
                    proc = subprocess.Popen(player)
                except FileNotFoundError:
                    logger.warning("Sound player not found for %s", sound_path)
            else:
                logger.warning("Alarm sound file not found: %s", sound_path)

        # Fallback: system bell / beep
        if proc is None:
            try:
                proc = subprocess.Popen(["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"])
            except Exception:
                pass   # silent if no sound system
        
        with self._alarm_lock:
            self._active_sound_proc = proc

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_alarms(self):
        if self._alarms_path.exists():
            try:
                with open(self._alarms_path) as f:
                    raw = json.load(f)
                # Support old format (list of {"time": ...}) and new format
                if isinstance(raw, list):
                    self._alarms = [
                        a if "recurrence" in a else {**a, "recurrence": "", "sound": ""}
                        for a in raw
                    ]
                else:
                    self._alarms = raw.get("alarms", [])
                logger.info("Loaded %d alarm(s).", len(self._alarms))
            except Exception:
                self._alarms = []

    def _save_alarms(self):
        self._alarms_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._alarms_path, "w") as f:
            json.dump({"alarms": self._alarms}, f, indent=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seconds_to_words(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds} second{'s' if seconds != 1 else ''}"
        minutes = seconds // 60
        secs    = seconds % 60
        label   = f"{minutes} minute{'s' if minutes != 1 else ''}"
        if secs:
            label += f" and {secs} second{'s' if secs != 1 else ''}"
        return label