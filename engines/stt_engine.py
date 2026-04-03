"""
stt_engine.py — Speech-to-Text via faster-whisper
Records from the microphone until silence, transcribes with Whisper,
and filters out non-speech audio via no_speech_prob threshold.
"""

import io
import logging
import threading
import wave
from typing import Callable, Optional

import numpy as np
import pyaudio
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class STTEngine:
    def __init__(self, config: dict):
        self.cfg = config["stt"]
        self._last_audio_time = 0.0
        self._lock = threading.Lock()

        logger.info("Loading Whisper model '%s' …", self.cfg["model_size"])
        self.model = WhisperModel(
            self.cfg["model_size"],
            device=self.cfg["device"],
            compute_type=self.cfg["compute_type"],
        )
        self._pa = pyaudio.PyAudio()
        logger.info("STT ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def listen(self, running_check: Optional[Callable[[], bool]] = None) -> Optional[str]:
        """
        Open the mic, record until silence, transcribe.
        Returns None if nothing intelligible was captured or interrupted.
        """
        audio_data = self._record_until_silence(running_check)
        if audio_data is None:
            return None
        return self._transcribe(audio_data)

    def close(self):
        self._pa.terminate()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_until_silence(self, running_check: Optional[Callable[[], bool]] = None) -> Optional[bytes]:
        import time
        rate = 16000
        chunk = 1024
        channels = 1
        silence_thresh = self.cfg["silence_threshold"]
        silence_dur = self.cfg["silence_duration_sec"]
        max_dur = self.cfg["max_record_sec"]

        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
        )

        frames = []
        silent_chunks = 0
        silent_limit = int(rate / chunk * silence_dur)
        max_chunks = int(rate / chunk * max_dur)
        speech_started = False

        try:
            for _ in range(max_chunks):
                # 🛑 Check for shutdown
                if running_check and not running_check():
                    return None

                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                except Exception as e:
                    logger.error("STT: Microphone read error: %s", e)
                    break

                frames.append(data)

                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                rms = np.sqrt(np.mean(arr ** 2)) / 32768.0

                # Track last audio time for watchdog
                self._last_audio_time = time.monotonic()

                if rms > silence_thresh:
                    speech_started = True
                    silent_chunks = 0
                elif speech_started:
                    silent_chunks += 1
                    if silent_chunks >= silent_limit:
                        break
                else:
                    # Not yet started speech - if it's been too long without speech, exit
                    # (Safety to prevent infinite open mic if something is slightly noisy)
                    if len(frames) > max_chunks // 2: 
                        break
        except Exception as e:
            logger.error("STT: Unexpected recording error: %s", e)
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass

        if not speech_started:
            return None

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))
        buf.seek(0)
        return buf.read()

    def _transcribe(self, audio_bytes: bytes) -> Optional[str]:
        buf = io.BytesIO(audio_bytes)
        segments, info = self.model.transcribe(
            buf,
            language=self.cfg.get("language"),
            beam_size=5,
            vad_filter=True,
        )

        # Collect all segments
        segment_list = list(segments)

        # Filter by no_speech_prob — high prob means background noise / silence
        threshold = self.cfg.get("no_speech_prob_threshold", 0.6)
        good = [s for s in segment_list if s.no_speech_prob < threshold]

        if not good:
            logger.debug("STT filtered all segments (no_speech_prob too high).")
            return None

        text = " ".join(s.text.strip() for s in good).strip()
        if not text:
            return None

        logger.info("STT → %s", text)
        return text

    def get_last_audio_time(self) -> float:
        """Return monotonic timestamp of last audio chunk read (for watchdog)."""
        return self._last_audio_time
