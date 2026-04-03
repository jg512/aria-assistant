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

import requests

if TYPE_CHECKING:
    from .memory import Memory

logger = logging.getLogger(__name__)

TOOL_SCHEMA_TEMPLATE = """
You have these tools available for user requests.

Available tools:
{tools}

GUIDANCE: Only call tools for explicit requests. For general questions or chat, reply naturally in plain English.
"""


class Agent:
    def __init__(self, config: dict, tools: dict, memory: Optional["Memory"] = None):
        self.cfg = config
        self.ollama_cfg = config["ollama"]
        self.tools = tools
        self.memory = memory
        self._history: list[dict] = []
        self._config = config  # kept for live reload

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        # ─────────────────────────────────────────────────────────
        # STEP 1: Try rule-based pattern matching FIRST
        # ─────────────────────────────────────────────────────────
        tool_call = self._detect_tool_pattern(user_text)
        
        if tool_call:
            logger.info("Matched rule-based pattern: %s", tool_call)
            tool_result = self._execute_tool(tool_call)
            logger.info("Tool result: %s", tool_result)
            
            # Record tool execution in history
            self._history.append({"role": "assistant", "content": f"[Executing: {tool_call['tool']}]"})
            self._history.append({"role": "user", "content": f"TOOL_RESULT: {tool_result}"})
            
            # Ask LLM to formulate a spoken reply
            system_prompt = self._build_system_prompt()
            final_reply = self._call_ollama(self._history, system_prompt)
        else:
            # ─────────────────────────────────────────────────────────
            # STEP 2: Fall back to LLM for general conversation
            # ─────────────────────────────────────────────────────────
            logger.info("No rule matched; querying LLM")
            system_prompt = self._build_system_prompt()
            final_reply = self._call_ollama(self._history, system_prompt)
            
            # Try to parse JSON tool calls from LLM (just in case)
            tool_call = self._parse_tool_call(final_reply)
            if tool_call:
                logger.info("LLM generated tool call: %s", tool_call)
                tool_result = self._execute_tool(tool_call)
                logger.info("Tool result: %s", tool_result)
                
                # Get a final spoken response from LLM
                self._history.append({"role": "assistant", "content": final_reply})
                self._history.append({"role": "user", "content": f"TOOL_RESULT: {tool_result}"})
                final_reply = self._call_ollama(self._history, system_prompt)

        self._history.append({"role": "assistant", "content": final_reply})
        return final_reply

    def reset(self):
        self._history.clear()
        logger.info("Conversation history cleared.")

    def reload_config(self, config: dict):
        """Hot-reload: update config without restarting."""
        self._config = config
        self.ollama_cfg = config["ollama"]
        logger.info("Agent: config reloaded.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        tool_desc = "\n".join(
            f"- {name}: {inst.DESCRIPTION}" for name, inst in self.tools.items()
        )
        base = self._config["assistant"]["personality"]
        tool_section = TOOL_SCHEMA_TEMPLATE.format(tools=tool_desc)
        mem_section = self.memory.as_prompt_context() if self.memory else ""
        return base + tool_section + mem_section

    def _call_ollama(self, messages: list[dict], system_prompt: str) -> str:
        payload = {
            "model": self.ollama_cfg["model"],
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "stream": False,
            "options": {
                "temperature": self._config["assistant"]["temperature"],
                "num_predict": self._config["assistant"]["max_response_tokens"],
            },
        }
        url = f"{self.ollama_cfg['host']}/api/chat"
        try:
            resp = requests.post(url, json=payload, timeout=self.ollama_cfg["timeout"])
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            return "Sorry, I can't reach my brain right now."
        except requests.exceptions.Timeout:
            return "That took too long. Please try again."
        except Exception as e:
            logger.error("Ollama error: %s", e)
            return "Something went wrong. Please try again."

    @staticmethod
    def _parse_tool_call(text: str) -> Optional[dict]:
        match = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+?"[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _detect_tool_pattern(self, user_text: str) -> Optional[dict]:
        """
        Rule-based pattern matching for tool invocation.
        Checked FIRST before calling LLM for better performance with small models.
        """
        text = user_text.lower().strip()
        
        # ─── MUSIC TOOL ───────────────────────────────────────────
        if any(p in text for p in ["play music", "play some", "shuffle", "start music", "play song"]):
            query = "shuffle"
            words = text.split()
            for i, word in enumerate(words):
                if word not in ["play", "music", "some", "please", "can", "you", "start", "a", "song", "and"]:
                    query = word
                    break
            return {"tool": "music", "action": "play", "query": query}
        
        if text.startswith("play ") and not any(x in text for x in ["video", "movie", "film", "show"]):
            query = text[5:].strip()
            if query:
                return {"tool": "music", "action": "play", "query": query}
            return {"tool": "music", "action": "play", "query": "shuffle"}
        
        # Music control: be lenient with context (these are common standalone commands)
        if text in ["pause", "resume", "stop", "next", "skip"] or text.startswith(("pause ", "resume ", "stop ", "next ", "skip ")):
            if text.startswith("pause") or text == "pause":
                return {"tool": "music", "action": "pause"}
            if text.startswith("resume") or text == "resume" or text == "continue":
                return {"tool": "music", "action": "resume"}
            if text.startswith("stop") or text == "stop":
                return {"tool": "music", "action": "stop"}
            if text.startswith("next") or text == "next":
                return {"tool": "music", "action": "next"}
            if text.startswith("skip") or text == "skip":
                return {"tool": "music", "action": "next"}
        
        # More explicit music control patterns
        if any(p in text for p in ["pause music", "pause the music", "pause song", "pause that", "pause now"]):
            return {"tool": "music", "action": "pause"}
        if any(p in text for p in ["resume", "continue playing", "unpause", "restart music", "keep playing"]):
            return {"tool": "music", "action": "resume"}
        if any(p in text for p in ["stop music", "turn off music", "stop the music", "stop playing", "silence"]):
            return {"tool": "music", "action": "stop"}
        if any(p in text for p in ["next song", "next track", "skip song", "skip track"]):
            return {"tool": "music", "action": "next"}
        
        if any(p in text for p in ["what's playing", "what is playing", "now playing", "currently playing", "what song", "what track"]):
            return {"tool": "music", "action": "what_is_playing"}
        
        # ─── WEATHER TOOL ─────────────────────────────────────────
        if any(p in text for p in ["weather", "temperature", "forecast", "rain", "sunny", "cloudy", "how is the weather", "what's the weather"]):
            return {"tool": "weather", "action": "current"}
        
        # ─── TO-DO TOOL ───────────────────────────────────────────
        if any(p in text for p in ["add to my todo", "add to my to-do", "add item", "new task", "remind me to", "don't forget", "add to todo"]):
            return {"tool": "todo", "action": "add", "item": text}
        if text.startswith("add "):
            # Generic add pattern: "add [task]"
            return {"tool": "todo", "action": "add", "item": text[4:].strip() if text[4:].strip() else text}
        
        if any(p in text for p in ["show my todo", "show my to-do", "show todo", "list todo", "what's on my list", "my tasks", "list my tasks"]):
            return {"tool": "todo", "action": "list"}
        
        # ─── CALENDAR TOOL ────────────────────────────────────────
        if any(p in text for p in ["calendar", "events", "upcoming events", "what's on my calendar", "appointments", "show calendar"]):
            return {"tool": "calendar", "action": "list"}
        
        # Event creation patterns
        if any(p in text for p in ["add event", "add meeting", "add to calendar", "schedule", "remind me to"]):
            # Extract event title and when
            summary = text
            when = ""
            
            # Remove common prefixes
            for prefix in ["add event", "add meeting", "add to calendar", "schedule", "remind me to"]:
                if prefix in text:
                    summary = text.replace(prefix, "").strip()
                    break
            
            # Split "title when" (e.g., "call mom tomorrow" → "call mom", "tomorrow")
            # Look for day names or time keywords
            day_keywords = ["tomorrow", "today", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "next", "tomorrow", "in"]
            time_keywords = ["am", "pm", "at", "o'clock"]
            for kw in day_keywords + time_keywords:
                if kw in summary:
                    parts = summary.split(kw, 1)
                    summary = parts[0].strip()
                    when = kw + " " + parts[1].strip() if len(parts) > 1 else kw
                    break
            
            if not when:
                when = "tomorrow"  # default
            
            return {"tool": "calendar", "action": "add_reminder", "summary": summary, "when": when}
        
        
        # ─── NEWS TOOL ────────────────────────────────────────────
        if any(p in text for p in ["news", "headlines", "what's new", "latest news", "read me the news", "tell me the news"]):
            return {"tool": "news", "action": "headlines"}
        
        # ─── VOLUME TOOL ──────────────────────────────────────────
        if "volume" in text or text in ["mute", "unmute"]:
            if any(p in text for p in ["mute", "silent", "silence"]):
                return {"tool": "volume", "action": "set", "level": 0}
            if any(p in text for p in ["unmute", "unmuted", "max", "maximum", "full", "loud"]):
                return {"tool": "volume", "action": "set", "level": 100}
            if "up" in text or text == "louder":
                return {"tool": "volume", "action": "up"}
            if "down" in text or text == "quieter":
                return {"tool": "volume", "action": "down"}
            # Try to extract a number: "volume 50" or "set volume to 50"
            match = re.search(r'\b(\d{1,3})\b', text)
            if match:
                level = int(match.group(1))
                return {"tool": "volume", "action": "set", "level": level}
        
        # ─── SYSTEM TOOL ──────────────────────────────────────────
        if any(p in text for p in ["shutdown", "power off", "turn off", "reboot", "restart"]):
            return {"tool": "system", "action": "shutdown"}
        if any(p in text for p in ["what time", "what's the time", "tell me the time"]):
            return {"tool": "system", "action": "get_time"}
        if any(p in text for p in ["what date", "what day", "today's date", "what's today"]):
            return {"tool": "system", "action": "get_date"}
        if any(p in text for p in ["set timer", "timer", "set a timer"]):
            match = re.search(r'(\d+)\s*(second|minute|hour)', text)
            if match:
                value = int(match.group(1))
                unit = match.group(2).lower()
                seconds = value * (3600 if unit == "hour" else 60 if unit == "minute" else 1)
                return {"tool": "system", "action": "set_timer", "seconds": seconds}
            return {"tool": "system", "action": "set_timer", "seconds": 300}  # default 5 min
        if any(p in text for p in ["set alarm", "alarm at"]):
            match = re.search(r'(\d{1,2}):?(\d{2})\s*(am|pm)?', text)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                ampm = match.group(3)
                if ampm == "pm" and hour != 12:
                    hour += 12
                elif ampm == "am" and hour == 12:
                    hour = 0
                time_str = f"{hour:02d}:{minute:02d}"
                return {"tool": "system", "action": "set_alarm", "time_str": time_str}
        if any(p in text for p in ["show alarms", "list alarms", "my alarms"]):
            return {"tool": "system", "action": "list_alarms"}
        if any(p in text for p in ["cancel alarm", "remove alarm"]):
            match = re.search(r'(\d{1,2}):?(\d{2})', text)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                time_str = f"{hour:02d}:{minute:02d}"
                return {"tool": "system", "action": "cancel_alarm", "time_str": time_str}
        
        # Alarm dismissal / snooze (for active alarms)
        if any(p in text for p in ["stop the alarm", "dismiss alarm", "stop alarm", "quiet", "silence alarm", "turn off alarm"]):
            return {"tool": "system", "action": "dismiss_alarm"}
        if any(p in text for p in ["snooze", "snooze alarm", "snooze for", "five more minutes"]):
            match = re.search(r'(\d+)\s*(minute|min)', text)
            snooze_min = int(match.group(1)) if match else 10
            return {"tool": "system", "action": "snooze_alarm", "snooze_minutes": snooze_min}
        
        return None

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
