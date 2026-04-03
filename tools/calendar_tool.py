"""
tools/calendar_tool.py — Read and create events in a local .ics file
No external calendar API needed — works with any exported Google/Apple calendar.
Supports voice commands: "add meeting tomorrow at 2pm", "remind me to call mom friday"

Enhancements:
- Background reminder thread: proactively speaks when an event is about to start
- Configurable reminder lead-times (e.g. 10 minutes before)
- Reminder deduplication so the same event isn't announced twice
"""

import logging
import re
import threading
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)


class CalendarTool:
    NAME = "calendar"
    DESCRIPTION = (
        "Read and create calendar events. "
        "Actions: today, tomorrow, week, next [query], add [summary], add_reminder [summary]."
    )
    PARAMETERS = {
        "action": {
            "type": "string",
            "enum": ["today", "tomorrow", "week", "next", "add", "add_reminder"],
        },
        "query":   {"type": "string", "description": "Event name keyword (used with action=next)."},
        "summary": {"type": "string", "description": "Event title (used with action=add or add_reminder)."},
        "when":    {
            "type": "string",
            "description": "When to create event: 'tomorrow 2pm', 'friday at 6', 'next monday 10am', etc.",
        },
    }

    def __init__(
        self,
        config: dict,
        speak_callback: Optional[Callable[[str], None]] = None,
    ):
        self.cfg        = config["calendar"]
        raw_path = Path(self.cfg["ics_path"])
        
        # Resolve relative paths against the project root
        if not raw_path.is_absolute():
            project_root = Path(__file__).parent.parent
            self.ics_path = (project_root / raw_path).resolve()
        else:
            self.ics_path = raw_path

        self.lookahead  = self.cfg.get("lookahead_days", 7)
        self._speak     = speak_callback or (lambda t: None)
        self._reminded: set[tuple] = set()

        # Reminder config ---------------------------------------------------
        self.reload_config(config)

        if self._remind_enabled:
            self._reminder_thread = threading.Thread(
                target=self._reminder_loop, daemon=True, name="calendar-reminders"
            )
            self._reminder_thread.start()
            logger.info(
                "Calendar reminder thread started (lead times: %s min).",
                self._lead_minutes,
            )

    def reload_config(self, config: dict):
        self.cfg = config["calendar"]
        raw_path = Path(self.cfg["ics_path"])
        if not raw_path.is_absolute():
            project_root = Path(__file__).parent.parent
            self.ics_path = (project_root / raw_path).resolve()
        else:
            self.ics_path = raw_path
        
        self.lookahead = self.cfg.get("lookahead_days", 7)
        reminder_cfg = self.cfg.get("reminders", {})
        self._remind_enabled = reminder_cfg.get("enabled", True)
        self._lead_minutes = reminder_cfg.get("lead_minutes", [10, 1])
        logger.info("CalendarTool: config reloaded.")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> str:
        return "ics loaded" if self.ics_path.exists() else "no ics file"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, action: str = "today", query: str = "", summary: str = "", when: str = "", **_) -> str:
        if action == "add":
            return self._add_event(summary, when, is_reminder=False)
        elif action == "add_reminder":
            return self._add_event(summary, when, is_reminder=True)

        events = self._parse_ics()
        today  = date.today()

        if action == "today":
            return self._events_on(events, today)
        elif action == "tomorrow":
            return self._events_on(events, today + timedelta(days=1))
        elif action == "week":
            return self._events_range(events, today, today + timedelta(days=self.lookahead))
        elif action == "next":
            return self._next_event(events, query)
        return "Unknown calendar action."

    # ------------------------------------------------------------------
    # ICS parsing
    # ------------------------------------------------------------------

    def _parse_ics(self) -> list[dict]:
        if not self.ics_path.exists():
            logger.warning("Calendar: no .ics file at %s", self.ics_path)
            return []
        try:
            text = self.ics_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        # Handle line folding: join lines starting with a space/tab to the previous line
        lines = []
        for line in text.splitlines():
            if (line.startswith(" ") or line.startswith("\t")) and lines:
                lines[-1] += line[1:]
            else:
                lines.append(line)

        events: list[dict] = []
        current: dict      = {}
        in_event           = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN:VEVENT":
                in_event = True
                current  = {}
            elif line == "END:VEVENT" and in_event:
                in_event = False
                ev = self._finalise(current)
                if ev:
                    events.append(ev)
            elif in_event and ":" in line:
                key, _, value = line.partition(":")
                key = key.split(";")[0]
                current[key] = value

        return sorted(events, key=lambda e: (e["start"], e.get("start_dt") or datetime.min))

    def _finalise(self, raw: dict) -> Optional[dict]:
        summary  = raw.get("SUMMARY", "Untitled event")
        dtstart  = raw.get("DTSTART", "")
        dtend    = raw.get("DTEND", dtstart)
        location = raw.get("LOCATION", "")

        start    = self._parse_dt(dtstart)
        end      = self._parse_dt(dtend)
        start_dt = self._parse_datetime(dtstart)   # full datetime for reminder matching
        if start is None:
            return None

        return {
            "summary":  summary,
            "start":    start,
            "end":      end,
            "location": location,
            "start_dt": start_dt,   # may be None for all-day events
        }

    @staticmethod
    def _parse_dt(value: str) -> Optional[date]:
        value = value.strip()
        try:
            if len(value) == 8:
                return datetime.strptime(value, "%Y%m%d").date()
            elif "T" in value:
                return datetime.strptime(value.rstrip("Z")[:15], "%Y%m%dT%H%M%S").date()
        except ValueError:
            pass
        return None

    @staticmethod
    def _parse_datetime(value: str) -> Optional[datetime]:
        """Return a full datetime for timed events, None for all-day."""
        value = value.strip()
        if "T" not in value:
            return None
        try:
            return datetime.strptime(value.rstrip("Z")[:15], "%Y%m%dT%H%M%S")
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _events_on(self, events: list[dict], day: date) -> str:
        found = [e for e in events if e["start"] == day]
        label = "Today" if day == date.today() else day.strftime("%A")
        if not found:
            return f"{label} you have no events."
        summaries = [e["summary"] for e in found]
        return f"{label} you have: {', '.join(summaries)}."

    def _events_range(self, events: list[dict], start: date, end: date) -> str:
        found = [e for e in events if start <= e["start"] <= end]
        if not found:
            return f"You have no events in the next {self.lookahead} days."
        parts = []
        for e in found[:5]:
            day_name = "Today" if e["start"] == date.today() else e["start"].strftime("%A")
            parts.append(f"{day_name}: {e['summary']}")
        return ". ".join(parts) + "."

    def _next_event(self, events: list[dict], query: str) -> str:
        today    = date.today()
        upcoming = [e for e in events if e["start"] >= today]
        if query:
            upcoming = [e for e in upcoming if query.lower() in e["summary"].lower()]
        if not upcoming:
            return f"No upcoming events matching '{query}'." if query else "No upcoming events."
        e   = upcoming[0]
        day = "Today" if e["start"] == today else e["start"].strftime("%A, %B %-d")
        return f"Your next event is {e['summary']} on {day}."

    # ------------------------------------------------------------------
    # Event creation
    # ------------------------------------------------------------------

    def _add_event(self, summary: str, when: str, is_reminder: bool = False) -> str:
        if not summary:
            return "Please provide an event title."
        if not when:
            return "When should I schedule this? (e.g., tomorrow at 2pm, friday, next monday 10am)"

        event_dt = self._parse_when(when)
        if event_dt is None:
            return "I couldn't understand when you meant. Try: 'tomorrow', 'friday at 3pm', 'next monday 10am'."

        # 👯 Duplicate prevention
        existing = self._parse_ics()
        for ev in existing:
            if ev["summary"].lower() == summary.lower():
                # Compare as date or datetime
                if ev["start_dt"]:
                    if ev["start_dt"] == event_dt:
                        return f"You already have '{summary}' scheduled for that time."
                elif ev["start"] == event_dt.date():
                    return f"You already have '{summary}' scheduled for that day."

        try:
            self._append_to_ics(summary, event_dt, is_reminder)
            day_name = event_dt.strftime("%A, %B %-d")
            return f"Added: {summary} on {day_name}."
        except Exception as e:
            logger.error("Failed to add event: %s", e)
            return f"Failed to add event: {e}"

    def _parse_when(self, when_str: str) -> Optional[datetime]:
        text  = when_str.lower().strip()
        today = date.today()

        time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)?', text)
        hour, minute, ampm = 0, 0, None
        if time_match:
            hour   = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            ampm   = time_match.group(3)
            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0

        target_date = None
        if "tomorrow" in text:
            target_date = today + timedelta(days=1)
        elif "today" in text:
            target_date = today
        elif re.search(r'in\s+(\d+)\s+days?', text):
            days = int(re.search(r'(\d+)', text).group(1))
            target_date = today + timedelta(days=days)
        else:
            weekdays = {
                "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6,
            }
            for day_name, day_num in weekdays.items():
                if day_name in text:
                    days_ahead = (day_num - today.weekday()) % 7
                    if days_ahead == 0 and "next" in text:
                        days_ahead = 7
                    target_date = today + timedelta(days=days_ahead)
                    break

        if target_date is None:
            return None
        return datetime.combine(target_date, datetime.min.time().replace(hour=hour, minute=minute))

    def _append_to_ics(self, summary: str, dt: datetime, is_reminder: bool = False):
        if not self.ics_path.exists():
            self._create_empty_ics()

        content = self.ics_path.read_text(encoding="utf-8")
        uid          = str(uuid.uuid4())
        dtstamp      = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        summary_safe = summary.replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;")

        if dt.hour == 0 and dt.minute == 0:
            dtstart_line = f"DTSTART;VALUE=DATE:{dt.strftime('%Y%m%d')}"
            dtend_line   = f"DTEND;VALUE=DATE:{(dt + timedelta(days=1)).strftime('%Y%m%d')}"
        else:
            dtstart_line = f"DTSTART:{dt.strftime('%Y%m%dT%H%M%S')}"
            dtend_line   = f"DTEND:{(dt + timedelta(hours=1)).strftime('%Y%m%dT%H%M%S')}"

        vevent = (
            f"BEGIN:VEVENT\n"
            f"UID:{uid}\n"
            f"DTSTAMP:{dtstamp}\n"
            f"{dtstart_line}\n"
            f"{dtend_line}\n"
            f"SUMMARY:{summary_safe}\n"
            f"DESCRIPTION:{'Reminder: ' + summary if is_reminder else summary}\n"
            f"END:VEVENT\n"
        )

        if "END:VCALENDAR" in content:
            content = content.replace("END:VCALENDAR", vevent + "END:VCALENDAR")
        else:
            content += vevent

        self.ics_path.write_text(content, encoding="utf-8")
        logger.info("Added event to calendar: %s", summary)

    def _create_empty_ics(self):
        template = (
            "BEGIN:VCALENDAR\n"
            "VERSION:2.0\n"
            "PRODID:-//Aria Assistant//EN\n"
            "CALSCALE:GREGORIAN\n"
            "METHOD:PUBLISH\n"
            "END:VCALENDAR\n"
        )
        self.ics_path.parent.mkdir(parents=True, exist_ok=True)
        self.ics_path.write_text(template, encoding="utf-8")

    # ------------------------------------------------------------------
    # Proactive reminder background thread
    # ------------------------------------------------------------------

    def _reminder_loop(self):
        """
        Runs every 60 seconds.
        For each upcoming timed event, checks whether the current time falls
        within any of the configured lead-minute windows and speaks a reminder
        if it hasn't already done so for that (event, lead_time) combination.
        """
        while True:
            try:
                self._check_reminders()
            except Exception as e:
                logger.warning("Reminder loop error: %s", e)
            time.sleep(60)

    def _check_reminders(self):
        events = self._parse_ics()
        now    = datetime.now()
        today  = now.date()

        for event in events:
            start_dt: Optional[datetime] = event.get("start_dt")
            if start_dt is None:
                continue   # skip all-day events
            if start_dt.date() < today:
                continue   # past event

            minutes_until = (start_dt - now).total_seconds() / 60

            for lead in self._lead_minutes:
                if not (0 <= minutes_until - lead < 1):
                    # Not within the 1-minute firing window for this lead time
                    continue

                remind_key = (event["summary"], start_dt.isoformat(), lead)
                if remind_key in self._reminded:
                    continue

                self._reminded.add(remind_key)
                self._announce_reminder(event, lead)

        # Prune old reminded keys to avoid unbounded growth
        cutoff = (now - timedelta(days=1)).isoformat()
        self._reminded = {
            k for k in self._reminded
            if len(k) >= 2 and k[1] >= cutoff
        }

    def _announce_reminder(self, event: dict, lead_minutes: int):
        summary  = event["summary"]
        start_dt = event["start_dt"]
        time_str = start_dt.strftime("%I:%M %p").lstrip("0")

        if lead_minutes == 1:
            msg = f"Heads up! {summary} starts in 1 minute, at {time_str}."
        elif lead_minutes < 60:
            msg = f"Reminder: {summary} starts in {lead_minutes} minutes, at {time_str}."
        else:
            hours = lead_minutes // 60
            msg   = f"Reminder: {summary} starts in {hours} hour{'s' if hours != 1 else ''}, at {time_str}."

        logger.info("Calendar reminder: %s", msg)
        self._speak(msg)