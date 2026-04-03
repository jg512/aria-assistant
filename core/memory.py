"""
memory.py — Persistent long-term memory for Aria
Stores facts as key-value pairs in a JSON file.
The agent injects remembered facts into every system prompt.
"""

import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Memory:
    def __init__(self, config: dict):
        self.cfg = config["memory"]
        self.path = Path(self.cfg["file_path"])
        self.max_facts = self.cfg.get("max_facts", 50)
        self._lock = threading.Lock()
        self._facts: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember(self, key: str, value: str) -> str:
        """Store a fact. Returns confirmation string."""
        with self._lock:
            self._facts[key.lower().strip()] = {
                "value": value,
                "timestamp": datetime.now().isoformat(),
            }
            # Prune oldest if over limit
            if len(self._facts) > self.max_facts:
                oldest = sorted(self._facts, key=lambda k: self._facts[k]["timestamp"])[0]
                del self._facts[oldest]
            self._save()
        logger.info("Memory: stored '%s' = '%s'", key, value)
        return f"I'll remember that {key} is {value}."

    def recall(self, key: str) -> Optional[str]:
        """Look up a fact by key. Returns None if not found."""
        with self._lock:
            entry = self._facts.get(key.lower().strip())
        return entry["value"] if entry else None

    def forget(self, key: str) -> str:
        """Delete a stored fact."""
        with self._lock:
            if key.lower().strip() in self._facts:
                del self._facts[key.lower().strip()]
                self._save()
                return f"I've forgotten {key}."
        return f"I don't have anything stored for {key}."

    def all_facts(self) -> dict[str, str]:
        """Return all facts as {key: value}."""
        with self._lock:
            return {k: v["value"] for k, v in self._facts.items()}

    def as_prompt_context(self) -> str:
        """
        Return a string suitable for injection into the system prompt.
        Empty string if no facts stored.
        """
        facts = self.all_facts()
        if not facts:
            return ""
        lines = "\n".join(f"  - {k}: {v}" for k, v in facts.items())
        return f"\nThings you remember about the user:\n{lines}\n"

    def get_relevant_context(self, text: str, max_items: int = 5) -> str:
        """
        Search for facts related to the input text using simple keyword matching.
        Returns a formatted string for the prompt.
        """
        with self._lock:
            if not self._facts:
                return ""
            
            # 1. Tokenize query
            words = set(re.findall(r'\w+', text.lower()))
            if not words:
                # If no text (e.g. initial greeting), show few most recent
                relevant = sorted(self._facts.keys(), key=lambda k: self._facts[k]["timestamp"], reverse=True)[:3]
            else:
                # 2. Score facts by keyword overlap
                scores = []
                for key, data in self._facts.items():
                    key_words = set(re.findall(r'\w+', key))
                    val_words = set(re.findall(r'\w+', data["value"].lower()))
                    overlap = len(words.intersection(key_words | val_words))
                    if overlap > 0:
                        scores.append((overlap, key))
                
                # 3. Sort by score, then by recency
                scores.sort(key=lambda x: (x[0], self._facts[x[1]]["timestamp"]), reverse=True)
                relevant = [s[1] for s in scores[:max_items]]

            if not relevant:
                return ""

            lines = "\n".join(f"  - {k}: {self._facts[k]['value']}" for k in relevant)
            return f"\nRelevant things you remember about the user:\n{lines}\n"

    def count(self) -> int:
        with self._lock:
            return len(self._facts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self._facts = json.load(f)
                logger.info("Memory: loaded %d facts from %s", len(self._facts), self.path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Memory: could not load %s — starting fresh. (%s)", self.path, e)
                self._facts = {}
        else:
            self._facts = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._facts, f, indent=2)
