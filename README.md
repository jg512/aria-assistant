# Aria — Privacy-First Local AI Companion for Raspberry Pi 4

> **100% offline. No cloud. No tracking. Your home, your rules.**

[![CI](https://github.com/yourname/aria/actions/workflows/ci.yml/badge.svg)](https://github.com/yourname/aria/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%204-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What is Aria?

Aria is a voice-controlled AI assistant that runs entirely on a Raspberry Pi 4. No subscriptions, no cloud, no microphone data leaving your home. It listens for your wake word, understands speech, thinks with a local LLM, and talks back — all in under 15 seconds on modest hardware.

```
You: "Hey Aria, play some jazz and set a timer for 20 minutes."
Aria: "Playing jazz. Timer set for 20 minutes."
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ALWAYS ON (< 5% CPU)                              │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │      OpenWakeWord  — "hey aria" / custom wake words          │    │
│  └────────────────────────┬─────────────────────────────────────┘    │
│                           │ WAKE DETECTED                            │
│                           ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │   faster-whisper STT  (base.en · int8 · no_speech filter)   │    │
│  └────────────────────────┬─────────────────────────────────────┘    │
│                           │ transcript                               │
│                           ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │   TinyLlama 1B via Ollama  (localhost:11434)                 │    │
│  │   + long-term memory injection + tool-call JSON routing      │    │
│  └──────────┬──────────────────────────────────────┬────────────┘    │
│             │ tool call                            │ plain text      │
│             ▼                                     │                  │
│  ┌────────────────────────────┐                   │                  │
│  │  TOOLS                     │──── result ──────►│                  │
│  │  music (mpv IPC)           │                   ▼                  │
│  │  system (alarms · timer)   │     ┌─────────────────────────────┐  │
│  │  weather (Open-Meteo)      │     │  Piper TTS → aplay/USB out  │  │
│  │  home_assistant (HA REST)  │     │  + volume ducking callbacks  │  │
│  │  todo (JSON)               │     └─────────────────────────────┘  │
│  │  news (RSS)                │                                       │
│  │  calendar (.ics)           │     ┌─────────────────────────────┐  │
│  │  volume (amixer)           │     │  Dashboard  :5000           │  │
│  └────────────────────────────┘     │  (Flask · live history)     │  │
│                                     └─────────────────────────────┘  │
│  Background: Watchdog · Heartbeat · Alarm loop · mpv crash recovery  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Hardware

| Component | Recommended | Minimum |
|-----------|------------|---------|
| Board | Raspberry Pi 4 **8 GB** | Pi 4 4 GB |
| Microphone | USB mic (Blue Snowball iCE, MAONO AU-PM421) | Any USB mic |
| Speaker | USB speaker or 3.5 mm to portable speaker | Any ALSA output |
| Storage | 32 GB microSD (A2 rated) | 16 GB |
| OS | Raspberry Pi OS **Bookworm 64-bit** | Bookworm 32-bit |
| Power | Official Pi 4 PSU (5 V / 3 A) | — |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yourname/aria ~/aria
cd ~/aria

# 2. Install everything (one command)
chmod +x setup.sh && ./setup.sh

# 3. Add music (optional)
cp ~/Downloads/*.mp3 ~/Music/

# 4. Start
sudo systemctl start aria

# — Test without wake word ——
source venv/bin/activate
python3 main.py --no-wake-word
```

**That's it.** Aria will speak when ready.

---

## Project Structure

```
aria/
├── main.py               ← Orchestration — full Listen→Think→Act→Speak loop
├── agent.py              ← LLM brain with tool routing and memory injection
├── stt_engine.py         ← Whisper STT with no_speech_prob filtering
├── tts_engine.py         ← Piper TTS with interrupt and duck callbacks
├── wake_word.py          ← OpenWakeWord multi-model detector
├── memory.py             ← Persistent long-term memory (JSON)
├── heartbeat.py          ← Scheduled tasks & morning briefing
├── watchdog.py           ← Hardware health monitor (mic + Ollama)
├── dashboard.py          ← Flask web dashboard at :5000
├── config.json           ← All settings — edit this to customise
├── requirements.txt
├── setup.sh              ← One-shot Pi installer
├── tools/
│   ├── music.py          ← mpv controller (IPC socket, crash recovery, ducking)
│   ├── system.py         ← Timer, alarm (persistent), time/date, shutdown
│   ├── weather.py        ← Open-Meteo (free, no API key, cached)
│   ├── home_assistant.py ← Home Assistant REST API
│   ├── todo.py           ← To-do & shopping lists (persistent JSON)
│   ├── news.py           ← RSS headline reader (cached)
│   ├── calendar_tool.py  ← Local .ics calendar reader
│   └── volume.py         ← amixer volume control + duck/unduck
└── tests/
    ├── conftest.py        ← Mock hardware libs so tests run on any machine
    └── test_tools.py      ← pytest unit tests (no Pi required)
```

---

## Voice Commands

| You say | What happens |
|---------|-------------|
| `Hey Aria, play some music` | Shuffles `~/Music` via mpv |
| `Hey Aria, play jazz` | Searches your library for jazz |
| `Hey Aria, pause / resume / next` | Controls playback |
| `Hey Aria, volume 60` | Sets system + music volume |
| `Hey Aria, what's playing?` | Names the current track |
| `Hey Aria, what's the weather?` | Fetches Open-Meteo data |
| `Hey Aria, what's the forecast?` | 3-day forecast |
| `Hey Aria, add milk to my shopping list` | Persistent todo list |
| `Hey Aria, read me my shopping list` | Reads the list |
| `Hey Aria, turn on the kitchen light` | Home Assistant entity |
| `Hey Aria, read me the news` | RSS headlines |
| `Hey Aria, what's on my calendar today?` | Today's .ics events |
| `Hey Aria, set a timer for 10 minutes` | Fires TTS when done |
| `Hey Aria, set an alarm for 07:30` | Persists across reboots |
| `Hey Aria, what time is it?` | Tells the time |
| `Hey Aria, remember my favourite colour is blue` | Stores in memory |
| `Stop` / `Cancel` | Interrupts Aria mid-sentence |
| `Goodbye Aria` | Clean shutdown of assistant |
| `Hey Aria, shut down` | Asks to confirm, then shuts down Pi |

---

## Configuration

All settings live in `config.json`. Key sections:

### Weather
```json
"weather": {
  "latitude": 52.52,
  "longitude": 13.41,
  "location_name": "Berlin",
  "units": "celsius"
}
```
Uses [Open-Meteo](https://open-meteo.com/) — free, no API key, works offline-ish (caches for 30 min).

### Home Assistant
```json
"home_assistant": {
  "enabled": true,
  "host": "http://homeassistant.local:8123",
  "token": "your_long_lived_access_token"
}
```
Get a token in HA → Profile → Long-lived access tokens.

### USB Speaker
```bash
aplay -l        # find your card number
# e.g. card 2 → use "plughw:2,0" or "plughw:CARD=Speaker,DEV=0"
```
Update `tts.output_device` in `config.json`.

### Morning Briefing
```json
"heartbeat": {
  "enabled": true,
  "morning_briefing_time": "08:00",
  "briefing_components": ["time", "weather", "calendar", "todo"]
}
```

### Wake Words
```json
"wake_word": {
  "models": ["hey_jarvis", "hey_mycroft"],
  "threshold": 0.5
}
```
Supported: `hey_jarvis`, `alexa`, `hey_mycroft`, `hey_rhasspy`.

If you want to use `hey aria` (assistant name alias), the code now maps `hey_aria`/`aria` to `hey_jarvis` automatically, but you can also keep `models` in `wake_word` set to a supported model explicitly.

### Calendar
Export your Google/Apple calendar as `.ics` and save to `~/calendar.ics`.

---

## Adding a Custom Tool

1. Create `tools/my_tool.py`:

```python
class MyTool:
    NAME = "my_tool"
    DESCRIPTION = "What this tool does, in one sentence."
    PARAMETERS = {
        "action": {"type": "string", "enum": ["do_thing"]},
    }

    def __init__(self, config: dict):
        ...

    def run(self, action: str, **kwargs) -> str:
        if action == "do_thing":
            return "Done!"
        return "Unknown action."

    def get_status(self) -> str:   # optional — shown in dashboard
        return "ready"
```

2. Register it in `main.py`:

```python
from tools.my_tool import MyTool
self.tools["my_tool"] = MyTool(config)
```

The agent automatically includes it in its system prompt.

---

## Performance on Pi 4 (4 GB)

| Stage | Typical time |
|-------|-------------|
| Wake word (always on) | < 5% CPU |
| STT — base.en, ~5 s utterance | 3–4 s |
| LLM — TinyLlama, 100 tokens | 4–8 s |
| TTS — Piper, ~10 words | < 1 s |
| **Total round-trip** | **~8–13 s** |

**To reduce latency:**
- Switch to `tiny.en` Whisper model (faster, slightly less accurate)
- Use `en_US-ryan-low` Piper voice (smallest, fastest)
- Reduce `max_response_tokens` to 80 in `config.json`

---

## Operations

```bash
# Start / stop / restart
sudo systemctl start aria
sudo systemctl stop aria
sudo systemctl restart aria

# Live logs
journalctl -u aria -f

# Hot-reload config without restarting
kill -HUP $(pgrep -f main.py)

# Run tests (no Pi hardware needed)
source venv/bin/activate
pytest tests/ -v

# Manual run (no wake word, for development)
python3 main.py --no-wake-word

# Dashboard
open http://raspberrypi.local:5000
```

---

## Privacy

| Component | Network usage |
|-----------|--------------|
| OpenWakeWord | None — runs 100% locally |
| faster-whisper | None — runs 100% locally |
| TinyLlama / Ollama | None — localhost only |
| Piper TTS | None — standalone binary |
| Weather (Open-Meteo) | Outbound HTTPS on demand only |
| Home Assistant | Local network only |
| RSS News | Outbound HTTPS on demand only |

The microphone is **only** read by the STT engine after the wake word fires. OpenWakeWord processes raw audio frames in-process; they never leave the Pi.

---

## Troubleshooting

**No audio output**
```bash
aplay -l                          # list devices
aplay -D plughw:1,0 /usr/share/sounds/alsa/Front_Center.wav
```

**Ollama not starting**
```bash
systemctl status ollama
ollama list      # should show tinyllama
```

**Wake word not triggering**
- Lower `wake_word.threshold` to `0.3` in `config.json`
- Check mic: `arecord -d 3 /tmp/test.wav && aplay /tmp/test.wav`

**STT transcribing noise**
- Raise `stt.no_speech_prob_threshold` to `0.8`
- Check mic placement (USB mics work best 30–60 cm away)

---

## License

MIT — do whatever you want with it.
