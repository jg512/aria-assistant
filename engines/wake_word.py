"""
wake_word.py — Always-on wake-word detection via openwakeword
Supports multiple models simultaneously (any one can trigger).
"""

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np
import pyaudio
from openwakeword.model import Model as OWWModel

logger = logging.getLogger(__name__)

RATE = 16000
CHUNK = 1280   # ~80 ms — OWW preferred frame size

# OpenWakeWord ships with a small set of pretrained wake-word models.
# We support mapping user-friendly assistant names to supported models.
_MODEL_ALIAS = {
    "hey_aria": "hey_jarvis",
    "aria": "hey_jarvis",
    "hey_mycroft": "hey_mycroft",
    "alexa": "alexa",
    "hey_rhasspy": "hey_rhasspy",
}

SUPPORTED_MODELS = set(_MODEL_ALIAS.values())

class WakeWordDetector:
    def __init__(self, config: dict):
        self.cfg = config["wake_word"]
        self.threshold = self.cfg.get("threshold", 0.5)

        raw_models = self.cfg.get("models")
        if not raw_models:
            raw_models = []
            for phrase in config.get("assistant", {}).get("wake_words", []):
                alias = _MODEL_ALIAS.get(phrase.strip().lower())
                if alias:
                    raw_models.append(alias)

        mapped_models = []
        for model in raw_models or ["hey_jarvis"]:
            norm = model.strip().lower()
            mapped = _MODEL_ALIAS.get(norm, norm)
            if mapped not in SUPPORTED_MODELS:
                logger.warning(
                    "WakeWord model '%s' is not recognized; falling back to 'hey_jarvis'",
                    model,
                )
                mapped = "hey_jarvis"
            if mapped not in mapped_models:
                mapped_models.append(mapped)

        self._models_list = mapped_models or ["hey_jarvis"]

        logger.info("Loading OpenWakeWord models: %s", self._models_list)
        try:
            # Try new API first
            self.oww = OWWModel(
                wakeword_models=self._models_list,
            )
        except TypeError:
            # Fall back to simpler API for older versions
            logger.warning("Using legacy OpenWakeWord API")
            self.oww = OWWModel()

        self._pa = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None

        # Expose last-read timestamp for watchdog
        self._last_read: float = time.monotonic()
        logger.info("Wake-word detector ready. Models: %s", self._models_list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Open the microphone stream."""
        if self._stream is None:
            self._open_stream()
            logger.debug("WakeWord stream started.")

    def stop(self):
        """Close the microphone stream (but keep PyAudio alive)."""
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            logger.debug("WakeWord stream stopped.")

    def wait_for_wake_word(self, 
                           interrupt_check: Optional[Callable[[], bool]] = None,
                           running_check: Optional[Callable[[], bool]] = None) -> Optional[str]:
        """
        Block until any wake word is detected OR interrupt_check() returns True.
        Returns the name of the model that triggered, or None if interrupted/stopping.
        """
        self.start()  # ensure mic is open
        self.oww.reset()
        try:
            while True:
                # 🛑 Check for shutdown
                if running_check and not running_check():
                    return None

                # 1. Check for external interrupt (e.g. dashboard command)
                if interrupt_check and interrupt_check():
                    logger.debug("WakeWord detection interrupted by callback.")
                    return None

                audio = self._read_chunk()
                predictions = self.oww.predict(audio)

                for model_name, scores in predictions.items():
                    # Handle both native float and numpy.float32
                    if isinstance(scores, (float, np.floating)):
                        score = scores
                    else:
                        score = max(scores.values(), default=0.0)
                    
                    if score >= self.threshold:
                        logger.info("Wake word '%s' detected (score=%.2f)", model_name, score)
                        return model_name
        finally:
            self.stop()  # release mic for other engines

    def get_last_read_time(self) -> float:
        return self._last_read

    def close(self):
        self.stop()
        self._pa.terminate()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_stream(self):
        device_index = self.cfg.get("device_index")
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
        )

    def _read_chunk(self) -> np.ndarray:
        raw = self._stream.read(CHUNK, exception_on_overflow=False)
        self._last_read = time.monotonic()
        return np.frombuffer(raw, dtype=np.int16)
