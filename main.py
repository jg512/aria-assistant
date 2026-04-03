#!/usr/bin/env python3
"""
main.py — Aria: Privacy-first local AI companion for Raspberry Pi 4
─────────────────────────────────────────────────────────────────────
Full pipeline:
  [OpenWakeWord] → [faster-whisper STT] → [TinyLlama via Ollama]
                → [tool execution] → [Piper TTS]

Features:
  • Wake-word detection (OpenWakeWord, multi-model)
  • STT with no_speech_prob filtering
  • LLM tool-calling via JSON protocol
  • Volume ducking during TTS
  • TTS interrupt on "stop" / "cancel"
  • Conversation reset after configurable silence
  • Persistent memory, alarms, to-do
  • Morning briefing heartbeat
  • Hardware watchdog (mic + Ollama health)
  • Web dashboard at :5000
  • SIGHUP hot-reloads config without restart

Usage:
  python3 main.py [--config config.json] [--no-wake-word]
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

@contextmanager
def suppress_stderr():
    """Temporarily redirect stderr to devnull (to hide noisy ALSA/JACK warnings)."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def validate_config(config: dict):
    """Check for essential config keys to prevent crashes later."""
    required = {
        "assistant": ["name", "personality"],
        "ollama": ["host", "model"],
        "stt": ["engine"],
        "tts": ["engine", "piper_binary", "model_path"],
        "system": ["conversation_history_limit"]
    }
    for section, keys in required.items():
        if section not in config:
            raise KeyError(f"Missing required config section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Missing required config key: '{section}.{key}'")
from typing import Callable, Optional

# Core engines
from core.agent import Agent
from engines.stt_engine import STTEngine
from engines.tts_engine import TTSEngine
from engines.wake_word import WakeWordDetector
from core.memory import Memory
from utils.heartbeat import Heartbeat
from utils.watchdog import Watchdog
from utils.dashboard import Dashboard

# Tools
from tools.music import MusicTool
from tools.system import SystemTool
from tools.weather import WeatherTool
from tools.home_assistant import HomeAssistantTool
from tools.todo import TodoTool
from tools.news import NewsTool
from tools.calendar_tool import CalendarTool
from tools.volume import VolumeTool
from tools.notes import NotesTool
from tools.system_monitor import SystemMonitorTool
from tools.web_search import WebSearchTool

# ── Ollama startup check ──────────────────────────────────────────────────────

def wait_for_ollama(config: dict, speak: Optional[Callable] = None) -> bool:
    import requests
    host = config["ollama"]["host"]
    model = config["ollama"]["model"]
    retries = config["ollama"].get("startup_retries", 10)
    delay = config["ollama"].get("startup_retry_delay_sec", 3)
    logger = logging.getLogger("startup")

    for attempt in range(1, retries + 1):
        try:
            # 1. Check if server is up
            r = requests.get(f"{host}/api/tags", timeout=2)
            if r.status_code == 200:
                # 2. Check if specific model is pulled
                models = [m["name"] for m in r.json().get("models", [])]
                # Ollama often adds ':latest' automatically
                if model in models or f"{model}:latest" in models:
                    logger.info("Ollama is ready. Model '%s' is available.", model)
                    return True
                else:
                    logger.warning("Ollama server is up, but model '%s' is not pulled. Run 'ollama pull %s'", model, model)
                    if speak:
                        speak(f"Warning: model {model} is not installed in your Ollama library.")
                    return False
        except Exception:
            pass
        logger.warning("Waiting for Ollama … attempt %d/%d", attempt, retries)
        time.sleep(delay)

    logger.error("Ollama did not start after %d attempts.", retries)
    if speak:
        speak("Warning: my brain isn't responding. Some features will be unavailable.")
    return False


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(config: dict):
    from logging.handlers import RotatingFileHandler
    level = getattr(logging, config["system"]["log_level"].upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    log_file = config["system"].get("log_file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # Max 5MB per file, keep 3 backups
        handlers.append(RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "httpx", "werkzeug", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Main application ──────────────────────────────────────────────────────────

class Aria:
    def __init__(self, config: dict, config_path: str, use_wake_word: bool = True):
        self.cfg = config
        self._config_path = config_path
        self.use_wake_word = use_wake_word
        self.running = False
        self._last_interaction: float = time.monotonic()
        self.log = logging.getLogger("aria")

        self.log.info("╔══════════════════════════════════╗")
        self.log.info("║     Aria — Local AI Companion    ║")
        self.log.info("╚══════════════════════════════════╝")

        # ── Core engines & Tools ────────────────────────────────
        with suppress_stderr():
            self.tts = TTSEngine(config)
            self.stt = STTEngine(config)
            self.volume_tool = VolumeTool(config)
            self.music_tool = MusicTool(config)

        # Wire TTS duck/unduck to both volume and music
        def _on_start():
            self.volume_tool.duck()
            self.music_tool.duck()

        def _on_stop():
            self.volume_tool.unduck()
            self.music_tool.unduck()

        self.tts.on_start_speaking = _on_start
        self.tts.on_stop_speaking = _on_stop

        # Build system tool with speak + confirm callbacks
        self.system_tool = SystemTool(
            config,
            speak_callback=self.tts.speak,
            confirm_callback=self._ask_confirmation,
        )

        self.tools: dict = {
            MusicTool.NAME:         self.music_tool,
            SystemTool.NAME:        self.system_tool,
            VolumeTool.NAME:        self.volume_tool,
            WeatherTool.NAME:       WeatherTool(config),
            HomeAssistantTool.NAME: HomeAssistantTool(config),
            TodoTool.NAME:          TodoTool(config),
            NewsTool.NAME:          NewsTool(config),
            CalendarTool.NAME:      CalendarTool(config),
            NotesTool.NAME:         NotesTool(config),
            SystemMonitorTool.NAME: SystemMonitorTool(config),
            WebSearchTool.NAME:     WebSearchTool(config),
        }

        # ── Memory ──────────────────────────────────────────────
        self.memory = Memory(config)

        # ── Agent ───────────────────────────────────────────────
        self.agent = Agent(config, self.tools, self.memory)

        # ── Wake word ───────────────────────────────────────────
        with suppress_stderr():
            self.detector = WakeWordDetector(config) if use_wake_word else None

        # ── Supporting services ─────────────────────────────────
        self.heartbeat = Heartbeat(config, self.tts.speak, self.tools)

        def _get_combined_mic_time():
            # Check both STT and WakeWord detector last read times
            stt_time = self.stt.get_last_audio_time()
            ww_time = self.detector.get_last_read_time() if self.detector else 0
            return max(stt_time, ww_time)

        self.watchdog = Watchdog(
            config,
            get_mic_time=_get_combined_mic_time,
            speak=self.tts.speak,
        )

        # Dashboard for web UI
        self.dashboard = Dashboard(config, self.tools)

        # ── Signal handlers ─────────────────────────────────────
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGHUP,  self._handle_sighup)

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self):
        self.running = True
        name = self.cfg["assistant"]["name"]

        wait_for_ollama(self.cfg, self.tts.speak)

        self.heartbeat.start()
        self.watchdog.start()
        self.dashboard.start()

        self.tts.speak(f"Hi, I'm {name}. I'm ready to help.")
        self.log.info("Aria is running. Say '%s' to wake me up.",
                      self.cfg["assistant"]["name"])

        while self.running:
            try:
                await self._cycle()
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log.error("Unexpected error: %s", e, exc_info=True)
                await asyncio.sleep(1)

        self._shutdown()

    # ------------------------------------------------------------------
    # Single Listen → Think → Act → Speak cycle
    # ------------------------------------------------------------------

    async def _cycle(self):
        # ── Conversation reset after long silence ────────────────
        reset_after = self.cfg["assistant"].get("conversation_reset_after_silence_sec", 300)
        if time.monotonic() - self._last_interaction > reset_after:
            if self.agent._history:
                self.log.info("Resetting conversation after %.0f s silence.", reset_after)
                self.agent.reset()

        # ── Listen ───────────────────────────────────────────────
        user_text = None

        # 1. Check for dashboard text first (if any)
        if self.dashboard:
            user_text = self.dashboard.get_next_command(timeout=0.1)
            if user_text:
                self.log.info("DASHBOARD: %s", user_text)

        # 2. Otherwise, check for wake word and voice
        if not user_text:
            if self.detector:
                self.log.debug("Waiting for wake word …")
                self.dashboard.set_state("listening:wake")
                
                # Check for dashboard commands every 100ms within wait_for_wake_word
                triggered = self.detector.wait_for_wake_word(
                    interrupt_check=lambda: self.dashboard.get_next_command(timeout=0) is not None,
                    running_check=lambda: self.running
                )
                
                if triggered:
                    # Duck music immediately while we respond and listen
                    self.volume_tool.duck()
                    self.music_tool.duck()
                    
                    self.tts.speak("Yes?")
                else:
                    # We were interrupted by a dashboard command or shutdown, cycle to handle it
                    return

            self.log.info("Listening …")
            self.dashboard.set_state("listening")
            user_text = self.stt.listen(running_check=lambda: self.running)
            
            # If nothing was heard, unduck now. If something WAS heard, 
            # we keep it ducked until thinking and speaking are done.
            if not user_text:
                self.volume_tool.unduck()
                self.music_tool.unduck()

        if not user_text:
            if not self.detector:
                await asyncio.sleep(0.05)
            return

        self.log.info("USER: %s", user_text)
        self.dashboard.push_user(user_text)
        self._last_interaction = time.monotonic()

        # ── Interrupt command ────────────────────────────────────
        # Only interrupt if she's actually speaking, OR if it's a bare "stop"/"cancel"
        is_bare_stop = user_text.lower().strip() in ["stop", "cancel", "shut up", "quiet"]
        if self.tts.is_speaking() or is_bare_stop:
            if any(kw in user_text.lower() for kw in ["stop", "cancel", "shut up", "quiet"]):
                self.log.info("Interrupting speech.")
                self.tts.interrupt()
                # If it was just a "stop" to quiet her, we're done with this cycle.
                # If it was "stop radio", we let it fall through to the agent IF she wasn't speaking
                # OR if it's a more complex command.
                if is_bare_stop:
                    self.dashboard.set_state("idle")
                    return

        # ── Exit command ─────────────────────────────────────────
        if any(kw in user_text.lower() for kw in ["goodbye aria", "bye aria", "exit aria"]):
            self.tts.speak("Goodbye! Take care.")
            self.running = False
            return

        # ── Think ────────────────────────────────────────────────
        self.log.info("Thinking …")
        self.dashboard.set_state("thinking")
        try:
            reply = await self.agent.process(user_text)

            # ── Speak ────────────────────────────────────────────────
            self.log.info("ARIA: %s", reply)
            self.dashboard.set_state("speaking")
            self.dashboard.push_assistant(reply)
            self.tts.speak(reply)
            self.dashboard.set_state("idle")
        except Exception as e:
            self.log.error("AI processing or speech failed: %s", e, exc_info=True)
            err_msg = "I'm sorry, something went wrong. Let's try that again."
            self.dashboard.push_assistant(err_msg)
            self.tts.speak(err_msg)
            self.dashboard.set_state("idle")

    # ------------------------------------------------------------------
    # Confirmation helper (used by SystemTool for shutdown/reboot)
    # ------------------------------------------------------------------

    def _ask_confirmation(self, question: str) -> bool:
        self.tts.speak(question + " Say yes to confirm.")
        answer = self.stt.listen()
        if answer and any(w in answer.lower() for w in ["yes", "yeah", "confirm", "sure", "do it"]):
            return True
        self.tts.speak("OK, cancelled.")
        return False

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _handle_signal(self, signum, frame):
        self.log.info("Signal %d — stopping.", signum)
        self.running = False

    def _handle_sighup(self, signum, frame):
        """SIGHUP: hot-reload config.json without restarting."""
        self.log.info("SIGHUP received — reloading config …")
        try:
            new_cfg = load_config(self._config_path)
            self.cfg = new_cfg
            
            # Reload agent
            self.agent.reload_config(new_cfg)
            
            # Reload all tools
            for tool in self.tools.values():
                if hasattr(tool, "reload_config"):
                    tool.reload_config(new_cfg)
            
            self.log.info("Config reloaded successfully.")
        except Exception as e:
            self.log.error("Config reload failed: %s", e)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self):
        self.log.info("Shutting down …")
        self.heartbeat.stop()
        self.watchdog.stop()
        self.music_tool.shutdown()
        if self.detector:
            self.detector.close()
        self.stt.close()
        self.log.info("Aria stopped cleanly.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aria — Local AI Companion for Raspberry Pi 4")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--no-wake-word", action="store_true",
                        help="Listen continuously (useful for testing)")
    args = parser.parse_args()

    config = load_config(args.config)
    
    try:
        validate_config(config)
    except KeyError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    setup_logging(config)

    aria = Aria(config, args.config, use_wake_word=not args.no_wake_word)
    asyncio.run(aria.run())


if __name__ == "__main__":
    main()
