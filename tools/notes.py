"""
notes.py — Quick notes storage and retrieval
Stores notes in JSON with timestamps for later retrieval.

Enhancements:
- Tags/categories: notes can carry one or more tags
- Tag-based filtering in list_notes and search_notes
- list_tags action to discover all tags in use
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class NotesTool:
    NAME = "notes"
    DESCRIPTION = (
        "Quick note taking. "
        "Actions: add_note [text, tags?], list_notes [tag?], "
        "search_notes [query, tag?], list_tags, clear_notes."
    )
    PARAMETERS = {
        "action": {
            "type": "string",
            "enum": ["add_note", "list_notes", "clear_notes", "search_notes", "list_tags"],
        },
        "text":  {"type": "string", "description": "Note content."},
        "query": {"type": "string", "description": "Keyword to search for."},
        "tags":  {
            "type": "string",
            "description": (
                "Comma-separated tags/categories for the note, "
                "e.g. 'work, urgent' or 'shopping'."
            ),
        },
        "tag":   {
            "type": "string",
            "description": "Filter results to notes with this single tag.",
        },
    }
    def __init__(self, config: dict):
        self.cfg = config
        raw_path = Path(config["notes"]["file_path"])

        # Resolve relative paths against the project root
        if not raw_path.is_absolute():
            project_root = Path(__file__).parent.parent
            self.file_path = (project_root / raw_path).resolve()
        else:
            self.file_path = raw_path

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    def reload_config(self, config: dict):
        self.cfg = config
        raw_path = Path(config["notes"]["file_path"])
        if not raw_path.is_absolute():
            project_root = Path(__file__).parent.parent
            self.file_path = (project_root / raw_path).resolve()
        else:
            self.file_path = raw_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()
        logger.info("NotesTool: config reloaded.")
    # ------------------------------------------------------------------

    def _ensure_file(self):
        if not self.file_path.exists():
            with open(self.file_path, "w") as f:
                json.dump({"notes": []}, f)

    # ------------------------------------------------------------------

    def run(self, action="list_notes", text="", query="", tags="", tag="", **_) -> str:
        if action == "add_note":
            return self._add_note(text, tags)
        elif action == "list_notes":
            return self._list_notes(tag)
        elif action == "clear_notes":
            return self._clear_notes()
        elif action == "search_notes":
            return self._search_notes(query, tag)
        elif action == "list_tags":
            return self._list_tags()
        return "Unknown action."

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _add_note(self, text: str, tags: str = "") -> str:
        if not text or not text.strip():
            return "No note text provided."
        parsed_tags = self._parse_tags(tags)
        try:
            data = self._load()
            note = {
                "timestamp": datetime.now().isoformat(),
                "text":      text.strip(),
                "tags":      parsed_tags,
            }
            data["notes"].append(note)
            self._save(data)
            logger.info("Note added: %s", text[:50])
            tag_msg = f" Tagged: {', '.join(parsed_tags)}." if parsed_tags else ""
            return f"Note saved: {text[:60]}...{tag_msg}" if len(text) > 60 else f"Note saved: {text}.{tag_msg}"
        except Exception as e:
            logger.error("Failed to add note: %s", e)
            return f"Failed to save note: {e}"

    def _list_notes(self, tag: str = "") -> str:
        try:
            data  = self._load()
            notes = data.get("notes", [])
            if tag:
                notes = self._filter_by_tag(notes, tag)
            if not notes:
                base = f"You have no notes" + (f" tagged '{tag}'" if tag else "") + "."
                return base

            count  = len(notes)
            noun   = "note" if count == 1 else "notes"
            header = f"You have {count} {noun}" + (f" tagged '{tag}'" if tag else "") + ". "
            # Show last 5, most recent first
            recent = list(reversed(notes))[:5]
            parts  = []
            for i, note in enumerate(recent, 1):
                preview = note["text"][:50]
                note_tags = note.get("tags", [])
                tag_str   = f" [{', '.join(note_tags)}]" if note_tags else ""
                parts.append(f"Note {i}: {preview}{tag_str}")
            return header + ". ".join(parts) + "."
        except Exception as e:
            logger.error("Failed to list notes: %s", e)
            return f"Failed to read notes: {e}"

    def _clear_notes(self) -> str:
        try:
            self._save({"notes": []})
            logger.info("Notes cleared")
            return "Notes cleared."
        except Exception as e:
            logger.error("Failed to clear notes: %s", e)
            return f"Failed to clear notes: {e}"

    def _search_notes(self, query: str, tag: str = "") -> str:
        if not query:
            return "No search query provided."
        try:
            data  = self._load()
            notes = data.get("notes", [])
            if tag:
                notes = self._filter_by_tag(notes, tag)
            matches = [n for n in notes if query.lower() in n["text"].lower()]
            if not matches:
                qualifier = f" tagged '{tag}'" if tag else ""
                return f"No notes{qualifier} found containing '{query}'."

            count = len(matches)
            noun  = "note" if count == 1 else "notes"
            result = f"Found {count} {noun}. "
            for i, note in enumerate(matches[:5], 1):
                preview   = note["text"][:50]
                note_tags = note.get("tags", [])
                tag_str   = f" [{', '.join(note_tags)}]" if note_tags else ""
                result   += f"Note {i}: {preview}{tag_str}. "
            return result.rstrip()
        except Exception as e:
            logger.error("Failed to search notes: %s", e)
            return f"Failed to search notes: {e}"

    def _list_tags(self) -> str:
        try:
            data  = self._load()
            notes = data.get("notes", [])
            # Count tag usage
            counts: dict[str, int] = {}
            for note in notes:
                for t in note.get("tags", []):
                    counts[t] = counts.get(t, 0) + 1
            if not counts:
                return "Your notes have no tags yet."
            tag_list = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
            parts    = [f"{tag} ({n})" for tag, n in tag_list]
            return "Tags in use: " + ", ".join(parts) + "."
        except Exception as e:
            logger.error("Failed to list tags: %s", e)
            return f"Failed to list tags: {e}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tags(raw: str) -> list[str]:
        if not raw or not raw.strip():
            return []
        return [t.strip().lower() for t in raw.split(",") if t.strip()]

    @staticmethod
    def _filter_by_tag(notes: list[dict], tag: str) -> list[dict]:
        t = tag.lower().strip()
        return [n for n in notes if t in [x.lower() for x in n.get("tags", [])]]

    def _load(self) -> dict:
        with open(self.file_path) as f:
            return json.load(f)

    def _save(self, data: dict):
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)