"""
tools/volume.py — ALSA system volume control via amixer
Also provides duck() / unduck() for the TTS volume-ducking feature.
"""

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class VolumeTool:
    NAME = "volume"
    DESCRIPTION = (
        "Control system audio volume. "
        "Actions: set [level 0-100], get, mute, unmute, up [step?], down [step?]."
    )
    PARAMETERS = {
        "action": {
            "type": "string",
            "enum": ["set", "get", "mute", "unmute", "up", "down"],
        },
        "level": {
            "type": "integer",
            "description": "Volume percentage 0–100.",
        },
        "step": {
            "type": "integer",
            "description": "Step size for up/down (default 10).",
        },
    }

    def __init__(self, config: dict):
        self.cfg = config.get("tts", {})
        self._duck_volume = self.cfg.get("duck_volume", 20)
        self._pre_duck_level: Optional[int] = None
        self._mixer = self._detect_mixer()

    def get_status(self) -> str:
        level = self._get_level()
        return f"{level}%" if level is not None else "unknown"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, action: str, level: int = 50, step: int = 10, **_) -> str:
        action = action.lower().strip()
        if action == "set":
            return self._set(level)
        elif action == "get":
            return self._get()
        elif action == "mute":
            return self._mute()
        elif action == "unmute":
            return self._unmute()
        elif action == "up":
            return self._adjust(int(step))
        elif action == "down":
            return self._adjust(-int(step))
        return f"Unknown volume action: {action}"

    def duck(self):
        """Lower volume before TTS speaks. Called by on_start_speaking callback."""
        current = self._get_level()
        if current is not None and current > self._duck_volume:
            self._pre_duck_level = current
            self._set_raw(self._duck_volume)
            logger.debug("Volume ducked from %d%% to %d%%", current, self._duck_volume)

    def unduck(self):
        """Restore volume after TTS finishes. Called by on_stop_speaking callback."""
        if self._pre_duck_level is not None:
            self._set_raw(self._pre_duck_level)
            logger.debug("Volume restored to %d%%", self._pre_duck_level)
            self._pre_duck_level = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set(self, level: int) -> str:
        level = max(0, min(100, level))
        self._set_raw(level)
        return f"Volume set to {level} percent."

    def _get(self) -> str:
        level = self._get_level()
        return f"Volume is at {level} percent." if level is not None else "I couldn't read the volume."

    def _mute(self) -> str:
        subprocess.run(["amixer", "-q", "set", self._mixer, "mute"], capture_output=True)
        return "Audio muted."

    def _unmute(self) -> str:
        subprocess.run(["amixer", "-q", "set", self._mixer, "unmute"], capture_output=True)
        return "Audio unmuted."

    def _adjust(self, delta: int) -> str:
        current = self._get_level() or 50
        new_level = max(0, min(100, current + delta))
        self._set_raw(new_level)
        direction = "up" if delta > 0 else "down"
        return f"Volume {direction} to {new_level} percent."

    def _set_raw(self, level: int):
        subprocess.run(
            ["amixer", "-q", "set", self._mixer, f"{level}%"],
            capture_output=True,
        )

    def _get_level(self) -> Optional[int]:
        try:
            result = subprocess.run(
                ["amixer", "get", self._mixer],
                capture_output=True, text=True, timeout=2,
            )
            import re
            match = re.search(r"\[(\d+)%\]", result.stdout)
            return int(match.group(1)) if match else None
        except Exception:
            return None

    @staticmethod
    def _detect_mixer() -> str:
        """Try to detect the best ALSA mixer control name."""
        for name in ("Master", "PCM", "Speaker", "Headphone"):
            result = subprocess.run(
                ["amixer", "get", name], capture_output=True, text=True
            )
            if result.returncode == 0:
                return name
        return "Master"
