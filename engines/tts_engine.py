"""
tts_engine.py — Text-to-Speech via Piper
Supports:
  - Interrupt: call interrupt() to stop mid-sentence
  - Duck callbacks: on_start / on_stop hooks so music can lower its volume
"""

import logging
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, config: dict):
        self.cfg = config["tts"]
        self._interrupt_event = threading.Event()
        self._speaking = threading.Event()
        self._playback_proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

        # Optional callbacks — set by orchestrator after construction
        self.on_start_speaking: Optional[Callable[[], None]] = None
        self.on_stop_speaking: Optional[Callable[[], None]] = None

        self._check_binaries()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Synthesise and play text. Blocks until done or interrupted."""
        if not text or not text.strip():
            return

        if self._interrupt_event.is_set():
            logger.info("TTS: Speech cancelled by pre-existing interrupt.")
            self._interrupt_event.clear()
            return

        text = self._clean(text)
        logger.info("TTS ← %s", text)

        self._interrupt_event.clear()
        self._speaking.set()

        if self.on_start_speaking:
            try:
                self.on_start_speaking()
            except Exception as e:
                logger.debug("on_start_speaking callback error: %s", e)

        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name

            self._synthesise(text, wav_path)

            if not self._interrupt_event.is_set():
                self._play(wav_path)

        except Exception as e:
            logger.error("TTS error: %s", e)
        finally:
            self._speaking.clear()
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
            if self.on_stop_speaking:
                try:
                    self.on_stop_speaking()
                except Exception as e:
                    logger.debug("on_stop_speaking callback error: %s", e)

    def interrupt(self) -> None:
        """Stop TTS mid-playback."""
        self._interrupt_event.set()
        with self._lock:
            if self._playback_proc and self._playback_proc.poll() is None:
                self._playback_proc.terminate()
        logger.debug("TTS interrupted.")

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_binaries(self):
        for key, path in [("piper_binary", self.cfg["piper_binary"]),
                          ("model_path", self.cfg["model_path"])]:
            if not Path(path).exists():
                logger.warning("TTS: %s not found at %s — speech will be silent.", key, path)

    def _synthesise(self, text: str, wav_path: str) -> None:
        cmd = [
            self.cfg["piper_binary"],
            "--model", self.cfg["model_path"],
            "--config", self.cfg["model_config"],
            "--output_file", wav_path,
        ]
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=15,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Piper failed: {proc.stderr.decode()}")

    def _play(self, wav_path: str) -> None:
        device = self.cfg.get("output_device", "default")
        cmd = ["aplay", "-D", device, wav_path]
        with self._lock:
            self._playback_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        self._playback_proc.wait()

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"[*_`#>~]", "", text)
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
