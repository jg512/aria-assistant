"""
agent.py — AI brain
Sends conversation history to TinyLlama via Ollama, parses tool-call JSON,
and injects long-term memory into the system prompt.
"""

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Optional

import httpx
import requests

if TYPE_CHECKING:
    from .memory import Memory

logger = logging.getLogger(__name__)

TOOL_SCHEMA_TEMPLATE = """
You have these tools available for user requests.

Available tools:
{tools}

INSTRUCTIONS FOR TOOLS:
1. To call a tool, use this exact format: [ACTION: tool_name action=... param=...]
2. Alternatively, just state what you are doing naturally.
3. If a TOOL_RESULT is provided, you MUST use that information to form your response. Do not ignore it.
4. If the tool failed or found nothing, tell the user exactly that.
5. Only use tools when explicitly requested. For chat, reply naturally.
"""

# ---------------------------------------------------------------------------
# Helpers shared across pattern-matching functions
# ---------------------------------------------------------------------------

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_KEYWORDS  = ["tomorrow", "today"] + _WEEKDAYS + ["next"]
_TIME_KEYWORDS = ["am", "pm", "at", "o'clock", ":"]
_PRIORITY_WORDS = {"high priority", "urgent", "important", "low priority", "low", "medium priority"}


def _extract_time_str(text: str) -> Optional[str]:
    """Return a normalised HH:MM string from a text fragment, or None."""
    # HH:MM am/pm
    m = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)?', text)
    if m:
        hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
        if ampm:
            if ampm.lower() == "pm" and hour != 12:
                hour += 12
            elif ampm.lower() == "am" and hour == 12:
                hour = 0
        return f"{hour:02d}:{minute:02d}"
    # H am/pm  (no colon)
    m = re.search(r'(\d{1,2})\s*(am|pm)', text)
    if m:
        hour, ampm = int(m.group(1)), m.group(2).lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:00"
    return None


def _extract_recurrence(text: str) -> str:
    """Return a canonical recurrence string from user text, or ''."""
    if re.search(r'\bevery\s+day\b|\bdaily\b', text):
        return "daily"
    if re.search(r'\bevery\s+weekday\b|\bweekdays\b', text):
        return "weekdays"
    if re.search(r'\bevery\s+weekend\b|\bweekends\b', text):
        return "weekends"
    for day in _WEEKDAYS:
        if re.search(rf'\bevery\s+{day}\b', text):
            return day
    return ""


def _split_summary_when(text: str, prefix: str) -> tuple[str, str]:
    """
    Strip *prefix* from *text*, then split the remainder into
    (summary, when) by finding the first day/time keyword.
    Returns (summary, when_str).
    """
    remainder = text.replace(prefix, "", 1).strip()
    when = "tomorrow"
    for kw in _DAY_KEYWORDS + _TIME_KEYWORDS:
        if kw in remainder:
            parts   = remainder.split(kw, 1)
            summary = parts[0].strip()
            when    = kw + (" " + parts[1].strip() if len(parts) > 1 else "")
            return summary, when.strip()
    return remainder, when


class Agent:
    def __init__(self, config: dict, tools: dict, memory: Optional["Memory"] = None):
        self.cfg        = config
        self.ollama_cfg = config["ollama"]
        self.tools      = tools
        self.memory     = memory
        self._history: list[dict] = []
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        # STEP 1: Rule-based pattern matching on USER input (fast path)
        tool_call = self._detect_tool_pattern(user_text)

        if tool_call:
            logger.info("Matched user rule pattern: %s", tool_call)
            tool_result = self._execute_tool(tool_call)
            
            # Add context for the LLM to provide the final spoken response
            self._history.append({"role": "assistant", "content": f"[Executing: {tool_call['tool']}]"})
            self._history.append({"role": "user",      "content": f"TOOL_RESULT: {tool_result}"})

            system_prompt = self._build_system_prompt(user_text)
            final_reply   = await self._call_ollama(self._history, system_prompt)
        else:
            # STEP 2: LLM for general conversation / deciding to call tools
            logger.info("No user rule matched; querying LLM")
            system_prompt = self._build_system_prompt(user_text)
            final_reply   = await self._call_ollama(self._history, system_prompt)

            # STEP 3: Check LLM output for tool triggers (both [ACTION: ...] and natural language)
            tool_call = self._parse_tool_call(final_reply)
            
            # If no bracketed action found, try natural language detection on the LLM's own words
            if not tool_call:
                tool_call = self._detect_tool_pattern(final_reply)

            if tool_call:
                logger.info("LLM triggered tool call: %s", tool_call)
                tool_result = self._execute_tool(tool_call)
                
                # Update history with the tool result and get a final polished response
                self._history.append({"role": "assistant", "content": final_reply})
                self._history.append({"role": "user",      "content": f"TOOL_RESULT: {tool_result}"})
                final_reply = await self._call_ollama(self._history, system_prompt)

        self._history.append({"role": "assistant", "content": final_reply})
        return final_reply

    def reset(self):
        self._history.clear()
        logger.info("Conversation history cleared.")

    def reload_config(self, config: dict):
        self._config    = config
        self.ollama_cfg = config["ollama"]
        logger.info("Agent: config reloaded.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_system_prompt(self, user_text: str = "") -> str:
        tool_desc    = "\n".join(f"- {name}: {inst.DESCRIPTION}" for name, inst in self.tools.items())
        base         = self._config["assistant"]["personality"]
        tool_section = TOOL_SCHEMA_TEMPLATE.format(tools=tool_desc)
        mem_section  = self.memory.get_relevant_context(user_text) if self.memory else ""
        return base + tool_section + mem_section

    async def _call_ollama(self, messages: list[dict], system_prompt: str) -> str:
        payload = {
            "model":   self.ollama_cfg["model"],
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "stream":  False,
            "options": {
                "temperature": self._config["assistant"]["temperature"],
                "num_predict": self._config["assistant"]["max_response_tokens"],
            },
        }
        url = f"{self.ollama_cfg['host']}/api/chat"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=self.ollama_cfg["timeout"])
                resp.raise_for_status()
                return resp.json()["message"]["content"].strip()
        except httpx.ConnectError:
            return "Sorry, I can't reach my brain right now."
        except httpx.TimeoutException:
            return "That took too long. Please try again."
        except Exception as e:
            logger.error("Ollama error: %s", e)
            return "Something went wrong. Please try again."

    def _parse_tool_call(self, text: str) -> Optional[dict]:
        """
        Parses tool calls from assistant text. 
        Supports:
        1. [ACTION: tool action=... param=...] (SLM friendly)
        2. {"tool": "...", ...} (JSON fallback)
        """
        # 1. Try Bracketed Action (Regex)
        # Format: [ACTION: tool_name action=value param1=val1 param2=val2]
        bracket_match = re.search(r'\[ACTION:\s*(\w+)\s+(.*?)\]', text, re.IGNORECASE)
        if bracket_match:
            tool_name = bracket_match.group(1).lower()
            params_str = bracket_match.group(2)
            
            call = {"tool": tool_name}
            # Extract key=value pairs (handles quotes and spaces reasonably)
            kv_pairs = re.findall(r'(\w+)=([\w:/.-]+|\"[^\"]+\")', params_str)
            for key, val in kv_pairs:
                val = val.strip('"')
                # Try to convert to int if possible
                if val.isdigit():
                    val = int(val)
                call[key] = val
            return call

        # 2. Try JSON Fallback
        json_match = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+?"[^{}]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None

    # ------------------------------------------------------------------
    # Rule-based pattern matching
    # ------------------------------------------------------------------

    def _detect_tool_pattern(self, user_text: str) -> Optional[dict]:
        """
        Checked FIRST before calling LLM.
        Order matters: specific patterns before generic ones.
        """
        text = user_text.lower().strip()

        # ── SYSTEM: alarm dismiss / snooze (highest priority) ─────────────
        if any(p in text for p in [
            "stop the alarm", "stop alarm", "dismiss alarm", "dismiss the alarm",
            "turn off alarm", "silence the alarm", "quiet the alarm",
        ]):
            return {"tool": "system", "action": "dismiss_alarm"}

        if any(p in text for p in ["snooze", "five more minutes", "ten more minutes", "sleep for"]):
            m = re.search(r'(\d+)\s*(minute|min)', text)
            return {"tool": "system", "action": "snooze_alarm",
                    "snooze_minutes": int(m.group(1)) if m else 10}

        # ── SYSTEM: briefing time ──────────────────────────────────────────
        if any(p in text for p in ["set briefing", "briefing at", "briefing time"]):
            ts = _extract_time_str(text)
            if ts:
                return {"tool": "system", "action": "set_briefing_time", "time_str": ts}

        # ── SYSTEM: alarms ────────────────────────────────────────────────
        if any(p in text for p in ["cancel alarm", "remove alarm", "delete alarm"]):
            ts = _extract_time_str(text)
            if ts:
                return {"tool": "system", "action": "cancel_alarm", "time_str": ts}

        if any(p in text for p in ["set alarm", "alarm for", "alarm at", "wake me"]):
            ts = _extract_time_str(text)
            if ts:
                rec   = _extract_recurrence(text)
                sound = self._extract_sound_path(text)
                call  = {"tool": "system", "action": "set_alarm", "time_str": ts}
                if rec:
                    call["recurrence"] = rec
                if sound:
                    call["sound"] = sound
                return call

        # Recurring alarm shorthand: "every monday at 7am"
        if re.search(r'\bevery\s+(' + '|'.join(_WEEKDAYS) + r'|day|weekday|weekend)\b', text):
            ts  = _extract_time_str(text)
            rec = _extract_recurrence(text)
            if ts and rec:
                call = {"tool": "system", "action": "set_alarm",
                        "time_str": ts, "recurrence": rec}
                sound = self._extract_sound_path(text)
                if sound:
                    call["sound"] = sound
                return call

        if any(p in text for p in ["show alarms", "list alarms", "my alarms", "what alarms", "list all alarms"]):
            return {"tool": "system", "action": "list_alarms"}

        # ── SYSTEM: time / date / timer ───────────────────────────────────
        if any(p in text for p in ["what time is it", "what's the time", "tell me the time", "current time"]):
            return {"tool": "system", "action": "get_time"}

        if any(p in text for p in ["what date", "what day", "today's date", "what's today", "what's the date"]):
            return {"tool": "system", "action": "get_date"}

        if any(p in text for p in ["set timer", "timer for", "set a timer", "remind me in"]):
            # Match formats like "5 minutes", "1 hour", "30 seconds", "10 min", "2 hrs"
            m = re.search(r'(\d+)\s*(sec|second|min|minute|hour|hr)', text)
            if m:
                value   = int(m.group(1))
                unit    = m.group(2).lower()
                seconds = value
                if unit.startswith("hour") or unit.startswith("hr"):
                    seconds = value * 3600
                elif unit.startswith("min"):
                    seconds = value * 60
                return {"tool": "system", "action": "set_timer", "seconds": seconds}
            
            # Fallback for "timer for 5" (assume minutes)
            m = re.search(r'(?:timer|remind me) (?:for |in )?(\d+)$', text)
            if m:
                return {"tool": "system", "action": "set_timer", "seconds": int(m.group(1)) * 60}
            
            return {"tool": "system", "action": "set_timer", "seconds": 300}

        # ── CALENDAR: events & reminders ──────────────────────────────────
        if any(p in text for p in [
            "add event", "add meeting", "add appointment",
            "schedule event", "schedule meeting",
        ]):
            prefix  = next(p for p in ["add event", "add meeting", "add appointment",
                                        "schedule event", "schedule meeting"] if p in text)
            summary, when = _split_summary_when(text, prefix)
            return {"tool": "calendar", "action": "add", "summary": summary, "when": when}

        if text.startswith("schedule "):
            has_time = any(kw in text for kw in _DAY_KEYWORDS + _TIME_KEYWORDS)
            if has_time:
                summary, when = _split_summary_when(text, "schedule ")
                return {"tool": "calendar", "action": "add", "summary": summary, "when": when}

        if any(p in text for p in ["remind me to", "remind me on", "don't forget to"]):
            prefix  = next(p for p in ["remind me to", "remind me on", "don't forget to"] if p in text)
            summary, when = _split_summary_when(text, prefix)
            return {"tool": "calendar", "action": "add_reminder", "summary": summary, "when": when}

        if any(p in text for p in [
            "calendar", "upcoming events", "what's on my calendar",
            "appointments", "show calendar", "my calendar",
        ]):
            return {"tool": "calendar", "action": "today"}

        # "add lunch/dinner/breakfast with ... at ..."  → calendar
        _MEAL_WORDS = ["lunch", "dinner", "breakfast", "brunch", "meeting", "appointment", "event"]
        if text.startswith("add "):
            has_meal = any(kw in text for kw in _MEAL_WORDS)
            has_time = any(kw in text for kw in _DAY_KEYWORDS + _TIME_KEYWORDS)
            if has_meal and has_time:
                summary, when = _split_summary_when(text, "add ")
                return {"tool": "calendar", "action": "add", "summary": summary, "when": when}

        # ── MUSIC: local & radio control (PRIORITY) ───────────────────────
        if text in ["pause", "resume", "stop", "next", "skip"] or \
                text.startswith(("pause ", "resume ", "stop ", "next ", "skip ")):
            if text.startswith("pause"):
                return {"tool": "music", "action": "pause"}
            if text.startswith(("resume", "continue")):
                return {"tool": "music", "action": "resume"}
            if text.startswith("stop"):
                return {"tool": "music", "action": "stop"}
            if text.startswith(("next", "skip")):
                return {"tool": "music", "action": "next"}

        if any(p in text for p in ["stop music", "turn off music", "stop the music", "stop playing", "stop radio"]):
            return {"tool": "music", "action": "stop"}

        # ── RADIO / STREAMS ───────────────────────────────────────────────
        if any(p in text for p in ["play radio", "tune in", "listen to radio", "stream radio"]):
            # "play radio BBC", "tune in to jazz radio"
            station = re.sub(
                r'\b(play radio|tune in to|tune in|listen to radio|stream radio)\b', '', text
            ).strip()
            return {"tool": "music", "action": "radio", "query": station or ""}

        # "play [station name]" when the word appears in config stations
        # (Only if it's a 'play' intent or a lone station name)
        _station_names = [s.lower() for s in self._get_station_names()]
        is_play_intent = any(p in text for p in ["play", "listen", "start", "tune"])
        for station_lc in _station_names:
            if station_lc in text:
                if is_play_intent or text == station_lc:
                    return {"tool": "music", "action": "radio", "query": station_lc}

        # "play [YouTube/URL]"
        if re.search(r'https?://', text):
            url = re.search(r'https?://\S+', user_text)
            if url:
                return {"tool": "music", "action": "radio", "query": url.group()}

        # ── MUSIC: local playback ─────────────────────────────────────────
        if any(p in text for p in ["play music", "play some", "shuffle", "start music", "play song"]):
            # Extract any meaningful query word
            stop_words = {"play", "music", "some", "please", "can", "you", "start", "a", "song", "and"}
            query = next((w for w in text.split() if w not in stop_words), "shuffle")
            return {"tool": "music", "action": "play", "query": query}

        if text.startswith("play ") and not any(x in text for x in ["video", "movie", "film", "show"]):
            query = text[5:].strip()
            return {"tool": "music", "action": "play", "query": query or "shuffle"}

        if any(p in text for p in ["pause music", "pause the music", "pause song", "pause that"]):
            return {"tool": "music", "action": "pause"}
        if any(p in text for p in ["resume", "continue playing", "unpause", "keep playing"]):
            return {"tool": "music", "action": "resume"}
        if any(p in text for p in ["next song", "next track", "skip song", "skip track"]):
            return {"tool": "music", "action": "next"}
        
        if any(p in text for p in ["start spotify", "turn on spotify", "enable spotify"]):
            return {"tool": "music", "action": "spotify", "query": "start"}
        if any(p in text for p in ["stop spotify", "turn off spotify", "disable spotify"]):
            return {"tool": "music", "action": "spotify", "query": "stop"}
        
        # SoundCloud patterns
        if "on soundcloud" in text:
            query = text.replace("on soundcloud", "").replace("play", "").strip()
            return {"tool": "music", "action": "soundcloud", "query": query}
        if text.startswith("soundcloud "):
            return {"tool": "music", "action": "soundcloud", "query": text[11:].strip()}
        if text.startswith("play ") and "soundcloud" in text:
            query = text.replace("play", "").replace("soundcloud", "").strip()
            return {"tool": "music", "action": "soundcloud", "query": query}

        if any(p in text for p in ["what's playing", "what is playing", "now playing", "currently playing", "what song"]):
            return {"tool": "music", "action": "what_is_playing"}

        # ── WEATHER ───────────────────────────────────────────────────────
        if any(p in text for p in ["weather", "forecast", "rain", "sunny", "cloudy", "how is the weather", "what's the weather"]):
            action = "forecast" if "forecast" in text else "current"
            return {"tool": "weather", "action": action}

        # ── NEWS ──────────────────────────────────────────────────────────
        if any(p in text for p in ["news", "headlines", "what's new", "latest news", "tell me the news"]):
            return {"tool": "news", "action": "headlines"}

        # ── MUSIC: generic catch-all ──────────────────────────────────────
        if text.startswith("play ") and not any(x in text for x in ["video", "movie", "film", "show", "todo", "alarm"]):
            query = text[5:].strip()
            if query:
                return {"tool": "music", "action": "play", "query": query}

        # ── VOLUME ────────────────────────────────────────────────────────
        if "volume" in text or text in ["mute", "unmute"] or any(x in text for x in ["louder", "quieter"]):
            if any(p in text for p in ["mute", "silent", "silence"]):
                return {"tool": "volume", "action": "mute"}
            if any(p in text for p in ["unmute", "unmuted", "max", "maximum", "full volume"]):
                return {"tool": "volume", "action": "unmute"}
            
            # Detect intensity for step size
            step = 10
            if any(p in text for p in ["little bit", "bit", "slightly", "small amount"]):
                step = 5
            elif any(p in text for p in ["a lot", "much", "way"]):
                step = 25

            if "up" in text or "louder" in text:
                return {"tool": "volume", "action": "up", "step": step}
            if "down" in text or "quieter" in text:
                return {"tool": "volume", "action": "down", "step": step}
            
            m = re.search(r'\b(\d{1,3})\b', text)
            if m:
                return {"tool": "volume", "action": "set", "level": int(m.group(1))}

        # ── NOTES ────────────────────────────────────────────────────────
        if any(p in text for p in ["add note", "note that", "write down", "remember to note", "make a note"]):
            prefix = next(p for p in ["add note", "note that", "write down",
                                       "remember to note", "make a note"] if p in text)
            note_text = text.replace(prefix, "", 1).strip()
            tags      = self._extract_note_tags(note_text)
            if tags:
                note_text = re.sub(r'\b(tagged?|tag|category|under)\s+[\w,\s]+$', '', note_text).strip()
            call = {"tool": "notes", "action": "add_note", "text": note_text}
            if tags:
                call["tags"] = tags
            return call

        if any(p in text for p in ["read my notes", "show my notes", "what notes", "list notes", "my notes"]):
            # "show my notes tagged work" → filter by tag
            tag = self._extract_filter_tag(text)
            call: dict = {"tool": "notes", "action": "list_notes"}
            if tag:
                call["tag"] = tag
            return call

        if any(p in text for p in ["search notes", "find note", "find in notes"]):
            prefix = next(p for p in ["search notes", "find note", "find in notes"] if p in text)
            query  = text.replace(prefix, "", 1).strip()
            tag    = self._extract_filter_tag(query)
            call   = {"tool": "notes", "action": "search_notes", "query": query}
            if tag:
                call["tag"] = tag
            return call

        if any(p in text for p in ["what tags", "show tags", "list tags", "note tags"]):
            return {"tool": "notes", "action": "list_tags"}

        if any(p in text for p in ["clear notes", "delete all notes"]):
            return {"tool": "notes", "action": "clear_notes"}

        # ── SYSTEM MONITOR ────────────────────────────────────────────────
        if any(p in text for p in ["system status", "how's the system", "system info"]):
            return {"tool": "system_monitor", "action": "status"}
        if any(p in text for p in ["cpu usage", "how much cpu", "processor usage"]):
            return {"tool": "system_monitor", "action": "cpu_usage"}
        if any(p in text for p in ["memory usage", "how much memory", "ram usage"]):
            return {"tool": "system_monitor", "action": "memory_usage"}
        if any(p in text for p in ["disk usage", "storage", "free space", "disk space"]):
            return {"tool": "system_monitor", "action": "disk_usage"}
        if any(p in text for p in ["temperature", "how hot", "what's the temp", "how warm"]):
            return {"tool": "system_monitor", "action": "temperature"}

        # ── TO-DO (checked last — most generic "add" pattern) ─────────────
        if any(p in text for p in ["add to my todo", "add to my to-do", "add task", "new task", "add item"]):
            item_text = re.sub(r'\b(add to my to-?do|add task|new task|add item)\b', '', text).strip()
            call = {"tool": "todo", "action": "add", "item": item_text or text}
            pri  = self._extract_priority(text)
            due  = self._extract_due_date(text)
            if pri:
                call["priority"] = pri
            if due:
                call["due"] = due
            return call

        if text.startswith("add "):
            item_text = text[4:].strip()
            # Strip priority / due modifiers from item text
            clean = re.sub(r'\b(high|medium|low)\s+priority\b', '', item_text).strip()
            clean = re.sub(r'\bdue\s+\S+', '', clean).strip()
            call  = {"tool": "todo", "action": "add", "item": clean or item_text}
            pri   = self._extract_priority(text)
            due   = self._extract_due_date(text)
            if pri:
                call["priority"] = pri
            if due:
                call["due"] = due
            return call

        if any(p in text for p in ["show todo", "list todo", "my tasks", "list my tasks", "what's on my list"]):
            return {"tool": "todo", "action": "list"}

        # ── SYSTEM: shutdown / reboot ─────────────────────────────────────
        if any(p in text for p in ["shutdown", "power off", "reboot", "restart"]):
            action = "reboot" if any(p in text for p in ["reboot", "restart"]) else "shutdown"
            return {"tool": "system", "action": action}

        return None

    # ------------------------------------------------------------------
    # Pattern-matching helpers
    # ------------------------------------------------------------------

    def _get_station_names(self) -> list[str]:
        """Return station names from config, safe even if key is absent."""
        return list(self._config.get("music", {}).get("radio_stations", {}).keys())

    @staticmethod
    def _extract_sound_path(text: str) -> str:
        """Extract a file path from 'with sound /path/to/file.wav' style text."""
        m = re.search(r'\bsound\s+([^\s]+\.(wav|mp3|ogg|oga))', text)
        return m.group(1) if m else ""

    @staticmethod
    def _extract_priority(text: str) -> str:
        """Return 'high', 'medium', or 'low' if mentioned, else ''."""
        if re.search(r'\b(high\s+priority|urgent|important)\b', text):
            return "high"
        if re.search(r'\blow\s+priority\b', text):
            return "low"
        if re.search(r'\bmedium\s+priority\b', text):
            return "medium"
        return ""

    @staticmethod
    def _extract_due_date(text: str) -> str:
        """Extract a due date expression from text, returning it as a string."""
        # "due friday", "due tomorrow", "due 2025-06-01"
        m = re.search(
            r'\bdue\s+('
            r'\d{4}-\d{2}-\d{2}'
            r'|tomorrow|today|next\s+week'
            r'|monday|tuesday|wednesday|thursday|friday|saturday|sunday'
            r')',
            text,
        )
        return m.group(1) if m else ""

    @staticmethod
    def _extract_note_tags(text: str) -> str:
        """
        Extract tags from patterns like:
          "buy milk tagged shopping"
          "meeting notes tag work, urgent"
          "dentist appointment category health"
        Returns a comma-separated tag string.
        """
        m = re.search(r'\b(tagged?|tag|category|under)\s+([\w,\s]+)$', text)
        if m:
            raw  = m.group(2)
            tags = [t.strip() for t in re.split(r'[,\s]+', raw) if t.strip()]
            return ", ".join(tags)
        return ""

    @staticmethod
    def _extract_filter_tag(text: str) -> str:
        """
        Extract a tag filter from patterns like:
          "show my notes tagged work"
          "list notes under shopping"
        """
        m = re.search(r'\b(tagged?|tag|under|category)\s+(\w+)', text)
        return m.group(2) if m else ""

    # ------------------------------------------------------------------

    def _execute_tool(self, call: dict) -> str:
        name = call.get("tool", "")
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: no tool named '{name}'."
        try:
            kwargs = {k: v for k, v in call.items() if k != "tool"}
            return tool.run(**kwargs)
        except Exception as e:
            logger.error("Tool '%s' raised: %s", name, e)
            return f"Error running {name}: {e}"

    def _trim_history(self):
        limit = self._config["system"]["conversation_history_limit"]
        if len(self._history) > limit * 2:
            self._history = self._history[-(limit * 2):]