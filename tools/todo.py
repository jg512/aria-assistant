"""
tools/todo.py — Shopping and to-do list manager
Persistent JSON storage. Supports multiple named lists.

Enhancements:
- Priority levels: high, medium, low (default medium)
- Due dates: stored as ISO date string, surfaced in list/summary
- Overdue detection in summary
"""

import json
import logging
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Priority ordering (lower number = more urgent)
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_PRIORITY_ALIASES = {
    "h": "high",   "hi": "high",
    "m": "medium", "med": "medium", "normal": "medium",
    "l": "low",    "lo": "low",
}


class TodoTool:
    NAME = "todo"
    DESCRIPTION = (
        "Manage to-do and shopping lists. "
        "Actions: add [item, list?, priority?, due?], "
        "remove [item, list?], list [list?], "
        "clear [list?], summary."
    )
    PARAMETERS = {
        "action": {
            "type": "string",
            "enum": ["add", "remove", "list", "clear", "summary"],
        },
        "item": {
            "type": "string",
            "description": "The item to add or remove.",
        },
        "list": {
            "type": "string",
            "description": "List name (default: 'todo'). Use 'shopping' for the shopping list.",
        },
        "priority": {
            "type": "string",
            "description": "Priority level: high, medium (default), or low.",
        },
        "due": {
            "type": "string",
            "description": "Due date in YYYY-MM-DD format, or natural language: 'tomorrow', 'friday', 'next week'.",
        },
    }

    def __init__(self, config: dict):
        raw_path = Path(config["todo"]["file_path"])

        # Resolve relative paths against the project root
        if not raw_path.is_absolute():
            project_root = Path(__file__).parent.parent
            self.path = (project_root / raw_path).resolve()
        else:
            self.path = raw_path

        self._lock = threading.Lock()
        self._data: dict[str, list[dict]] = {}
        self._load()

    def reload_config(self, config: dict):
        # Todo mainly cares about path which doesn't change usually, 
        # but we reload data just in case path changed in config
        raw_path = Path(config["todo"]["file_path"])
        if not raw_path.is_absolute():
            project_root = Path(__file__).parent.parent
            self.path = (project_root / raw_path).resolve()
        else:
            self.path = raw_path
        self._load()
        logger.info("TodoTool: config reloaded.")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> str:
        with self._lock:
            total = sum(len(v) for v in self._data.values())
        return f"{total} item(s) across {len(self._data)} list(s)"

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def run(
        self,
        action: str,
        item: str = "",
        list: str = "todo",
        priority: str = "medium",
        due: str = "",
        **_,
    ) -> str:
        list_name = (list or "todo").lower().strip()
        action    = action.lower().strip()

        if action == "add":
            return self._add(item, list_name, priority, due)
        elif action == "remove":
            return self._remove(item, list_name)
        elif action == "list":
            return self._list(list_name)
        elif action == "clear":
            return self._clear(list_name)
        elif action == "summary":
            return self._summary()
        return f"Unknown todo action: {action}"

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _add(self, item: str, list_name: str, priority: str, due: str) -> str:
        if not item:
            return "What should I add?"

        pri  = self._normalise_priority(priority)
        due_date = self._parse_due(due) if due else ""

        with self._lock:
            self._data.setdefault(list_name, [])
            entry = {
                "text":     item,
                "added":    datetime.now().isoformat(),
                "done":     False,
                "priority": pri,
                "due":      due_date,
            }
            self._data[list_name].append(entry)
            # Keep list sorted: high → medium → low, then by due date
            self._data[list_name].sort(key=self._sort_key)
            self._save()

        parts = [f"Added '{item}' to your {list_name} list"]
        if pri != "medium":
            parts.append(f"priority: {pri}")
        if due_date:
            parts.append(f"due: {due_date}")
        return ". ".join(parts) + "."

    def _remove(self, item: str, list_name: str) -> str:
        if not item:
            return "What should I remove?"
        with self._lock:
            items  = self._data.get(list_name, [])
            q      = item.lower()
            before = len(items)
            self._data[list_name] = [i for i in items if q not in i["text"].lower()]
            removed = before - len(self._data[list_name])
            self._save()
        if removed:
            return f"Removed '{item}' from your {list_name} list."
        return f"I couldn't find '{item}' in your {list_name} list."

    def _list(self, list_name: str) -> str:
        with self._lock:
            items = list(self._data.get(list_name, []))
        if not items:
            return f"Your {list_name} list is empty."

        today = date.today().isoformat()
        parts = []
        for entry in items:
            text = entry["text"]
            pri  = entry.get("priority", "medium")
            due  = entry.get("due", "")
            tags = []
            if pri == "high":
                tags.append("HIGH")
            elif pri == "low":
                tags.append("low")
            if due:
                if due < today:
                    tags.append(f"OVERDUE {due}")
                elif due == today:
                    tags.append("due today")
                else:
                    tags.append(f"due {due}")
            label = f"{text}"
            if tags:
                label += f" [{', '.join(tags)}]"
            parts.append(label)

        count = len(parts)
        noun  = "item" if count == 1 else "items"
        return f"Your {list_name} list ({count} {noun}): " + "; ".join(parts) + "."

    def _clear(self, list_name: str) -> str:
        with self._lock:
            self._data[list_name] = []
            self._save()
        return f"Cleared your {list_name} list."

    def _summary(self) -> str:
        with self._lock:
            snapshot = {k: list(v) for k, v in self._data.items() if v}
        if not snapshot:
            return "All your lists are empty."

        today = date.today().isoformat()
        parts = []
        overdue_total = 0
        high_total    = 0

        for name, items in snapshot.items():
            count   = len(items)
            overdue = sum(1 for i in items if i.get("due") and i["due"] < today)
            high    = sum(1 for i in items if i.get("priority") == "high")
            overdue_total += overdue
            high_total    += high
            label = f"{count} item{'s' if count != 1 else ''} on the {name} list"
            if overdue:
                label += f" ({overdue} overdue)"
            parts.append(label)

        result = "You have " + ", ".join(parts) + "."
        if high_total:
            result += f" {high_total} high-priority item{'s' if high_total != 1 else ''} need attention."
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_priority(raw: str) -> str:
        p = raw.lower().strip()
        return _PRIORITY_ALIASES.get(p, p if p in _PRIORITY_ORDER else "medium")

    @staticmethod
    def _parse_due(text: str) -> str:
        """
        Accept YYYY-MM-DD or natural language and return an ISO date string.
        Returns "" on parse failure.
        """
        t = text.lower().strip()
        today = date.today()

        # Try ISO format first
        try:
            return date.fromisoformat(t).isoformat()
        except ValueError:
            pass

        # Natural language shortcuts
        if t in ("today",):
            return today.isoformat()
        if t in ("tomorrow",):
            return (today + __import__("datetime").timedelta(days=1)).isoformat()
        if t in ("next week",):
            return (today + __import__("datetime").timedelta(weeks=1)).isoformat()

        # Weekday names
        _DAYS = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }
        for day_name, day_num in _DAYS.items():
            if day_name in t:
                ahead = (day_num - today.weekday()) % 7
                if ahead == 0:
                    ahead = 7   # "friday" when already friday → next friday
                return (today + __import__("datetime").timedelta(days=ahead)).isoformat()

        logger.warning("Could not parse due date: %s", text)
        return ""

    @staticmethod
    def _sort_key(entry: dict):
        pri_val = _PRIORITY_ORDER.get(entry.get("priority", "medium"), 1)
        due     = entry.get("due", "9999-12-31") or "9999-12-31"
        return (pri_val, due, entry.get("text", ""))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    raw = json.load(f)
                # Backwards-compat: inject missing fields
                for items in raw.values():
                    for item in items:
                        item.setdefault("priority", "medium")
                        item.setdefault("due", "")
                self._data = raw
            except Exception as e:
                logger.warning("Todo: load error: %s", e)
                self._data = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)