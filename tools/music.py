"""
tools/music.py — mpv music controller with IPC socket
Features: volume duck/restore, crash detection & recovery, get_status().

Enhancements:
- Metadata-based search: artist, album, title via mutagen (gracefully degraded if unavailable)
- Internet radio / stream support: play URLs directly (Icecast, SHOUTcast, YouTube via yt-dlp)
- 'radio' action with named stations defined in config
"""

import glob
import json
import logging
import os
import random
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Try to import mutagen for ID3/FLAC/OGG tag reading
try:
    from mutagen import File as MutagenFile  # type: ignore
    _HAS_MUTAGEN = True
except ImportError:
    _HAS_MUTAGEN = False
    logger.info("mutagen not installed — metadata search disabled, falling back to filename search.")

# Try yt-dlp for YouTube/stream URL resolution
try:
    import yt_dlp as _yt_dlp  # type: ignore
    _HAS_YTDLP = True
except ImportError:
    _HAS_YTDLP = False
    logger.info("yt-dlp not installed — YouTube stream support disabled.")


class MusicTool:
    NAME = "music"
    DESCRIPTION = (
        "Control local music playback, internet radio, Spotify, and SoundCloud. "
        "Actions: play [query/shuffle], pause, resume, stop, next, "
        "volume [level 0-100], what_is_playing, radio [station_name_or_url], "
        "spotify [start/stop], soundcloud [query]."
    )
    PARAMETERS = {
        "action": {
            "type": "string",
            "enum": ["play", "pause", "resume", "stop", "next", "volume", "what_is_playing", "radio", "spotify", "soundcloud"],
        },
        "query": {
            "type": "string",
            "description": (
                "Song title, artist name, album name, genre keyword, or 'shuffle'. "
                "For action=radio: station name from config or a direct stream URL."
            ),
        },
        "level": {
            "type": "integer",
            "description": "Volume 0–100 (for action=volume).",
        },
    }

    def __init__(self, config: dict):
        self.cfg           = config["music"]
        self.socket_path   = self.cfg["mpv_socket"]

        # 🧹 Clean up orphaned socket from previous crashes
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        # Named radio stations from config, e.g.:
        #   "radio_stations": {"BBC World Service": "https://...", "Jazz Radio": "https://..."}
        self._stations: dict[str, str] = self.cfg.get("radio_stations", {})

        self._mpv_proc: Optional[subprocess.Popen] = None
        self._spotify_proc: Optional[subprocess.Popen] = None
        self._playlist: list[str]  = []
        self._current_index: int   = 0
        self._paused: bool         = False
        self._ducked_volume: Optional[int] = None
        self._is_stream: bool      = False   # True when playing a URL/radio stream
        self._stream_name: str     = ""       # Display name for the stream

        # Metadata & File cache
        self._meta_cache: dict[str, dict] = {}
        self._file_cache: list[str] = []
        self._last_scan_time: float = 0.0
        self._cache_ttl = 600  # 10 minutes

        # Crash-recovery watchdog
        self._watchdog_stop   = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._crash_watchdog, daemon=True, name="mpv-watchdog"
        )
        self._watchdog_thread.start()

    def reload_config(self, config: dict):
        """Update configuration and clear file cache to pick up new paths."""
        self.cfg = config["music"]
        self._stations = self.cfg.get("radio_stations", {})
        self._file_cache = []  # Force rescan
        logger.info("MusicTool: config reloaded.")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> str:
        if not self._mpv_proc or self._mpv_proc.poll() is not None:
            return "stopped"
        if self._paused:
            return f"paused — {self._current_track_name()}"
        return f"playing — {self._current_track_name()}"

    # ------------------------------------------------------------------
    # Volume ducking hooks
    # ------------------------------------------------------------------

    def duck(self):
        if self._is_running():
            self._ducked_volume = self.cfg["default_volume"]
            self._ipc_command({"command": ["set_property", "volume", 15]})

    def unduck(self):
        if self._is_running() and self._ducked_volume is not None:
            self._ipc_command({"command": ["set_property", "volume", self._ducked_volume]})
            self._ducked_volume = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, action: str, query: str = "shuffle", level: int = 80, **_) -> str:
        action = action.lower().strip()
        if action == "play":
            return self._play(query)
        elif action == "pause":
            return self._pause()
        elif action == "resume":
            return self._resume()
        elif action == "stop":
            return self._stop()
        elif action == "next":
            return self._next()
        elif action == "volume":
            return self._set_volume(level)
        elif action == "what_is_playing":
            return self._now_playing()
        elif action == "radio":
            return self._play_radio(query)
        elif action == "spotify":
            return self._spotify(query)
        elif action == "soundcloud":
            return self._play_soundcloud(query)
        return f"Unknown music action: {action}"

    def shutdown(self):
        self._watchdog_stop.set()
        self._stop_mpv()
        self._stop_spotify()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _play(self, query: str) -> str:
        self._is_stream = False
        files = self._find_files(query)
        if not files:
            return f"No music found matching '{query}'."
        random.shuffle(files)
        self._playlist      = files
        self._current_index = 0
        self._paused        = False
        self._stop_mpv()
        self._start_mpv(files[0])
        title = self._current_track_name()
        return f"Playing {title}." if query.lower() not in ("shuffle", "") else "Shuffling your music."

    def _play_radio(self, query: str) -> str:
        """
        Play an internet radio station or stream URL.
        Resolution order:
          1. Direct URL (starts with http:// or https://)
          2. Named station in config
          3. yt-dlp URL extraction (YouTube, SoundCloud, etc.)
        """
        url = ""
        name = query

        if query.startswith("http://") or query.startswith("https://"):
            url  = query
            name = query
        else:
            # Case-insensitive station lookup
            q_lower = query.lower()
            for station_name, station_url in self._stations.items():
                if q_lower in station_name.lower():
                    url  = station_url
                    name = station_name
                    break

            # Late check/import for yt-dlp
            global _HAS_YTDLP, _yt_dlp
            if not _HAS_YTDLP:
                try:
                    import yt_dlp
                    _yt_dlp = yt_dlp
                    _HAS_YTDLP = True
                except ImportError:
                    pass

            if not url and _HAS_YTDLP:
                # Try to extract a streamable URL via yt-dlp
                extracted = self._yt_dlp_extract(query)
                if extracted:
                    url  = extracted["url"]
                    name = extracted.get("title", query)

        if not url:
            available = ", ".join(self._stations.keys()) if self._stations else "none configured"
            tip = " (yt-dlp not installed — YouTube not supported)" if not _HAS_YTDLP else ""
            return (
                f"I couldn't find a stream for '{query}'. "
                f"Available stations: {available}.{tip}"
            )

        self._stop_mpv()
        self._is_stream   = True
        self._stream_name = name
        self._playlist    = []
        self._paused      = False
        self._start_mpv(url)
        return f"Playing {name}."

    def _play_soundcloud(self, query: str) -> str:
        """Search and play a track on SoundCloud via yt-dlp."""
        # Late check/import for yt-dlp in case it was installed while running
        global _HAS_YTDLP, _yt_dlp
        if not _HAS_YTDLP:
            try:
                import yt_dlp
                _yt_dlp = yt_dlp
                _HAS_YTDLP = True
            except ImportError:
                return "SoundCloud support is disabled because yt-dlp is not installed."
        
        if not query:
            return "Please provide a song title or artist to search on SoundCloud."

        # Prefix with scsearch1: to tell yt-dlp to search SoundCloud specifically
        # If it's already a URL, yt-dlp will handle it correctly
        target = query
        if not (query.startswith("http://") or query.startswith("https://")):
            target = f"scsearch1:{query}"

        extracted = self._yt_dlp_extract(target)
        if not extracted:
            return f"I couldn't find '{query}' on SoundCloud."

        self._stop_mpv()
        self._is_stream   = True
        self._stream_name = extracted.get("title", query)
        self._playlist    = []
        self._paused      = False
        self._start_mpv(extracted["url"])
        return f"Playing {self._stream_name} from SoundCloud."

    def _pause(self) -> str:
        if not self._is_running():
            return "Nothing is playing."
        self._ipc_command({"command": ["set_property", "pause", True]})
        self._paused = True
        return "Music paused."

    def _resume(self) -> str:
        if not self._is_running():
            if self._playlist:
                return self._play("shuffle")
            return "Nothing queued."
        self._ipc_command({"command": ["set_property", "pause", False]})
        self._paused = False
        return f"Resuming {self._current_track_name()}."

    def _stop(self) -> str:
        self._stop_mpv()
        self._playlist    = []
        self._is_stream   = False
        self._stream_name = ""
        return "Music stopped."

    def _next(self) -> str:
        if self._is_stream:
            return "Can't skip tracks on a radio stream."
        if not self._playlist:
            return "Nothing in the queue."
        self._current_index = (self._current_index + 1) % len(self._playlist)
        path = self._playlist[self._current_index]
        self._stop_mpv()
        self._start_mpv(path)
        return f"Skipping to {self._current_track_name()}."

    def _set_volume(self, level: int) -> str:
        level = max(0, min(100, level))
        self.cfg["default_volume"] = level
        self._ipc_command({"command": ["set_property", "volume", level]})
        return f"Music volume set to {level} percent."

    def _now_playing(self) -> str:
        if not self._is_running():
            return "Nothing is playing right now."
        state = "Paused on" if self._paused else "Playing"
        return f"{state}: {self._current_track_name()}."

    def _spotify(self, action: str) -> str:
        """Control librespot (Spotify Connect receiver)."""
        action = action.lower().strip()
        if action in ("start", "on", "play"):
            if self._spotify_proc and self._spotify_proc.poll() is None:
                return "Spotify receiver is already running."
            
            librespot_bin = self.cfg.get("librespot_path", "librespot")
            device_name = self.cfg.get("spotify_device_name", "Aria Assistant")
            
            cmd = [librespot_bin, "--name", device_name]
            
            # Optional: add credentials if provided
            if self.cfg.get("spotify_user") and self.cfg.get("spotify_password"):
                cmd += ["--username", self.cfg["spotify_user"], "--password", self.cfg["spotify_password"]]
            
            try:
                self._stop_mpv() # Stop local music if starting Spotify
                self._spotify_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"Spotify receiver '{device_name}' started. You can now connect from your Spotify app."
            except Exception as e:
                logger.error("Failed to start librespot: %s", e)
                return (
                    f"Error starting Spotify: {e}. It seems librespot is not installed or not in your path. "
                    "To fix this, run: 'sudo apt install librespot' or 'cargo install librespot'."
                )
        
        elif action in ("stop", "off"):
            self._stop_spotify()
            return "Spotify receiver stopped."
        
        return "Unknown spotify action. Try 'start' or 'stop'."

    def _stop_spotify(self):
        if self._spotify_proc and self._spotify_proc.poll() is None:
            self._spotify_proc.terminate()
            try:
                self._spotify_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._spotify_proc.kill()
        self._spotify_proc = None

    # ------------------------------------------------------------------
    # File discovery & metadata search
    # ------------------------------------------------------------------

    def _find_files(self, query: str) -> list[str]:
        now = time.monotonic()
        # Use cache if fresh
        if self._file_cache and (now - self._last_scan_time) < self._cache_ttl:
            all_files = self._file_cache
        else:
            logger.info("Scanning music directories...")
            exts = self.cfg["supported_formats"]
            dirs = self.cfg["directories"]
            all_files = []
            for d in dirs:
                if not os.path.isdir(d):
                    continue
                for ext in exts:
                    all_files += glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True)
            self._file_cache = all_files
            self._last_scan_time = now

        if not query or query.lower() == "shuffle":
            return all_files

        q = query.lower()

        # 1. Metadata search (artist / album / title tags via mutagen)
        if _HAS_MUTAGEN:
            meta_matches = self._search_by_metadata(all_files, q)
            if meta_matches:
                return meta_matches

        # 2. Fallback: filename / parent-folder substring match
        return [
            f for f in all_files
            if q in Path(f).stem.lower() or q in Path(f).parent.name.lower()
        ]

    def _get_metadata(self, path: str) -> dict:
        """Return cached metadata dict for a file path."""
        if path in self._meta_cache:
            return self._meta_cache[path]
        meta = {"title": "", "artist": "", "album": ""}
        if _HAS_MUTAGEN:
            try:
                audio = MutagenFile(path, easy=True)
                if audio:
                    meta["title"]  = str(audio.get("title",  [""])[0]).lower()
                    meta["artist"] = str(audio.get("artist", [""])[0]).lower()
                    meta["album"]  = str(audio.get("album",  [""])[0]).lower()
            except Exception:
                pass
        self._meta_cache[path] = meta
        return meta

    def _search_by_metadata(self, files: list[str], query: str) -> list[str]:
        """Return files whose tags match the query string."""
        matches: list[str] = []
        for f in files:
            meta = self._get_metadata(f)
            if (
                query in meta["title"]
                or query in meta["artist"]
                or query in meta["album"]
            ):
                matches.append(f)
        return matches

    # ------------------------------------------------------------------
    # yt-dlp stream extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _yt_dlp_extract(query: str) -> Optional[dict]:
        """
        Use yt-dlp to resolve a YouTube search or URL into a direct audio stream URL.
        Returns {"url": ..., "title": ...} or None.
        """
        if not _HAS_YTDLP:
            return None
        ydl_opts = {
            "format":        "bestaudio/best",
            "quiet":         True,
            "no_warnings":   True,
            "skip_download": True,
        }
        # If it looks like a URL use it directly; else prepend ytdl search
        if not (query.startswith("http://") or query.startswith("https://")):
            # Only prepend ytsearch if no other supported search prefix is present
            if not any(query.startswith(prefix) for prefix in ["ytsearch", "scsearch", "ytdl", "scsearch1", "ytsearch1"]):
                query = f"ytsearch1:{query}"

        try:
            with _yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(query, download=False)
                if "entries" in info:
                    info = info["entries"][0]
                return {"url": info["url"], "title": info.get("title", query)}
        except Exception as e:
            logger.warning("yt-dlp extraction failed for '%s': %s", query, e)
            return None

    # ------------------------------------------------------------------
    # mpv process management
    # ------------------------------------------------------------------

    def _start_mpv(self, path_or_url: str) -> None:
        cmd = [
            "mpv",
            f"--input-ipc-server={self.socket_path}",
            f"--volume={self.cfg['default_volume']}",
            "--no-video",
            "--really-quiet",
            path_or_url,
        ]
        self._mpv_proc = subprocess.Popen(cmd)
        time.sleep(0.3)

    def _stop_mpv(self) -> None:
        if self._mpv_proc and self._mpv_proc.poll() is None:
            self._ipc_command({"command": ["quit"]})
            try:
                self._mpv_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._mpv_proc.kill()
        self._mpv_proc = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

    def _is_running(self) -> bool:
        return bool(self._mpv_proc and self._mpv_proc.poll() is None)

    def _ipc_command(self, payload: dict) -> Optional[dict]:
        if not os.path.exists(self.socket_path):
            return None
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(self.socket_path)
                s.sendall((json.dumps(payload) + "\n").encode())
                try:
                    return json.loads(s.recv(4096))
                except (socket.timeout, json.JSONDecodeError):
                    return None
        except OSError:
            return None

    def _current_track_name(self) -> str:
        if self._is_stream:
            return self._stream_name or "radio stream"
        if self._playlist and 0 <= self._current_index < len(self._playlist):
            path = self._playlist[self._current_index]
            # Prefer tag title if available
            if _HAS_MUTAGEN:
                meta = self._get_metadata(path)
                parts = []
                if meta["artist"]:
                    parts.append(meta["artist"].title())
                if meta["title"]:
                    parts.append(meta["title"].title())
                if parts:
                    return " – ".join(parts)
            return Path(path).stem
        return "unknown"

    # ------------------------------------------------------------------
    # Crash-recovery watchdog
    # ------------------------------------------------------------------

    def _crash_watchdog(self) -> None:
        interval = self.cfg.get("crash_check_interval_sec", 5)
        while not self._watchdog_stop.is_set():
            self._watchdog_stop.wait(interval)
            if self._watchdog_stop.is_set():
                break
            if (
                self._mpv_proc
                and self._mpv_proc.poll() is not None
                and not self._paused
                and not self._is_stream
            ):
                logger.warning("mpv crashed — attempting auto-next.")
                self._mpv_proc = None
                if self._playlist:
                    self._current_index = (self._current_index + 1) % len(self._playlist)
                    self._start_mpv(self._playlist[self._current_index])