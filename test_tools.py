"""
tests/test_tools.py — Unit tests for all Aria tools
Run with: pytest tests/ -v
No hardware required — all external calls are mocked.
"""

import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, date
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield {
            "assistant": {
                "name": "Aria",
                "personality": "Test assistant.",
                "max_response_tokens": 100,
                "temperature": 0.7,
                "conversation_reset_after_silence_sec": 300,
                "shutdown_confirmation": False,
            },
            "ollama": {"host": "http://localhost:11434", "model": "tinyllama", "timeout": 5,
                       "startup_retries": 1, "startup_retry_delay_sec": 0},
            "stt": {"model_size": "base.en", "device": "cpu", "compute_type": "int8",
                    "language": "en", "silence_threshold": 0.01, "silence_duration_sec": 1.5,
                    "max_record_sec": 30, "no_speech_prob_threshold": 0.6},
            "tts": {"piper_binary": "/tmp/piper", "model_path": "/tmp/model.onnx",
                    "model_config": "/tmp/model.onnx.json", "output_device": "default",
                    "duck_volume": 20, "duck_restore_delay_sec": 0.1},
            "music": {"directories": [tmpdir], "supported_formats": ["mp3", "flac"],
                      "default_volume": 80, "mpv_socket": "/tmp/test-mpv.sock",
                      "crash_check_interval_sec": 9999},
            "weather": {"latitude": 52.52, "longitude": 13.41, "location_name": "Berlin",
                        "units": "celsius", "cache_minutes": 30},
            "home_assistant": {"enabled": False, "host": "http://ha.local:8123",
                               "token": "test-token", "timeout": 5},
            "todo": {"file_path": os.path.join(tmpdir, "todo.json")},
            "news": {"feeds": [{"name": "Test", "url": "http://example.com/rss"}],
                     "max_headlines": 3, "cache_minutes": 60},
            "calendar": {"ics_path": os.path.join(tmpdir, "cal.ics"), "lookahead_days": 7},
            "memory": {"file_path": os.path.join(tmpdir, "memory.json"), "max_facts": 10},
            "alarms": {"file_path": os.path.join(tmpdir, "alarms.json")},
            "watchdog": {"enabled": False},
            "heartbeat": {"enabled": False},
            "dashboard": {"enabled": False},
            "system": {"log_level": "WARNING", "log_file": None,
                       "conversation_history_limit": 10,
                       "data_dir": tmpdir},
        }


# ── Memory tests ──────────────────────────────────────────────────────────────

class TestMemory:
    def test_remember_and_recall(self, base_config):
        from core.memory import Memory
        m = Memory(base_config)
        m.remember("favourite colour", "blue")
        assert m.recall("favourite colour") == "blue"

    def test_forget(self, base_config):
        from core.memory import Memory
        m = Memory(base_config)
        m.remember("pet name", "Biscuit")
        m.forget("pet name")
        assert m.recall("pet name") is None

    def test_as_prompt_context_empty(self, base_config):
        from core.memory import Memory
        m = Memory(base_config)
        assert m.as_prompt_context() == ""

    def test_as_prompt_context_filled(self, base_config):
        from core.memory import Memory
        m = Memory(base_config)
        m.remember("name", "Alice")
        ctx = m.as_prompt_context()
        assert "name" in ctx
        assert "Alice" in ctx

    def test_persistence(self, base_config):
        from core.memory import Memory
        m1 = Memory(base_config)
        m1.remember("city", "Berlin")
        m2 = Memory(base_config)   # re-load from same file
        assert m2.recall("city") == "Berlin"

    def test_max_facts_pruning(self, base_config):
        from core.memory import Memory
        m = Memory(base_config)
        for i in range(15):
            m.remember(f"key{i}", f"value{i}")
        assert m.count() <= base_config["memory"]["max_facts"]


# ── Todo tests ────────────────────────────────────────────────────────────────

class TestTodo:
    def test_add_and_list(self, base_config):
        from tools.todo import TodoTool
        t = TodoTool(base_config)
        t.run("add", item="milk")
        result = t.run("list")
        assert "milk" in result

    def test_remove(self, base_config):
        from tools.todo import TodoTool
        t = TodoTool(base_config)
        t.run("add", item="eggs")
        t.run("remove", item="eggs")
        result = t.run("list")
        assert "eggs" not in result

    def test_clear(self, base_config):
        from tools.todo import TodoTool
        t = TodoTool(base_config)
        t.run("add", item="butter")
        t.run("clear")
        result = t.run("list")
        assert "empty" in result.lower()

    def test_summary_empty(self, base_config):
        from tools.todo import TodoTool
        t = TodoTool(base_config)
        result = t.run("summary")
        assert "empty" in result.lower()

    def test_summary_with_items(self, base_config):
        from tools.todo import TodoTool
        t = TodoTool(base_config)
        t.run("add", item="apples")
        t.run("add", item="bread")
        result = t.run("summary")
        assert "2" in result or "two" in result.lower() or "item" in result

    def test_multiple_lists(self, base_config):
        from tools.todo import TodoTool
        t = TodoTool(base_config)
        t.run("add", item="call dentist", list="todo")
        t.run("add", item="milk", list="shopping")
        assert "call dentist" in t.run("list", list="todo")
        assert "milk" in t.run("list", list="shopping")

    def test_persistence(self, base_config):
        from tools.todo import TodoTool
        t1 = TodoTool(base_config)
        t1.run("add", item="persistent item")
        t2 = TodoTool(base_config)
        assert "persistent item" in t2.run("list")


# ── System tool tests ─────────────────────────────────────────────────────────

class TestSystemTool:
    def test_get_time(self, base_config):
        from tools.system import SystemTool
        s = SystemTool(base_config)
        result = s.run("get_time")
        assert any(p in result for p in ["AM", "PM", "It's"])

    def test_get_date(self, base_config):
        from tools.system import SystemTool
        s = SystemTool(base_config)
        result = s.run("get_date")
        assert "Today" in result

    def test_set_timer(self, base_config):
        from tools.system import SystemTool
        fired = []
        s = SystemTool(base_config, speak_callback=lambda t: fired.append(t))
        result = s.run("set_timer", seconds=1)
        assert "Timer" in result
        time.sleep(1.5)
        assert any("done" in f.lower() for f in fired)

    def test_set_and_list_alarms(self, base_config):
        from tools.system import SystemTool
        s = SystemTool(base_config)
        s.run("set_alarm", time_str="07:30")
        result = s.run("list_alarms")
        assert "07:30" in result

    def test_cancel_alarm(self, base_config):
        from tools.system import SystemTool
        s = SystemTool(base_config)
        s.run("set_alarm", time_str="08:00")
        s.run("cancel_alarm", time_str="08:00")
        result = s.run("list_alarms")
        assert "08:00" not in result

    def test_alarm_persistence(self, base_config):
        from tools.system import SystemTool
        s1 = SystemTool(base_config)
        s1.run("set_alarm", time_str="09:15")
        s2 = SystemTool(base_config)
        assert "09:15" in s2.run("list_alarms")

    def test_shutdown_no_confirmation(self, base_config):
        from tools.system import SystemTool
        with patch("subprocess.Popen") as mock_popen:
            s = SystemTool(base_config)
            result = s.run("shutdown")
            assert "Shutting" in result or "shut" in result.lower()
            mock_popen.assert_called_once()


# ── Weather tool tests ────────────────────────────────────────────────────────

class TestWeatherTool:
    def _mock_response(self):
        return {
            "current_weather": {"temperature": 18.5, "weathercode": 1, "windspeed": 12.0},
            "daily": {
                "temperature_2m_max": [19.0, 21.0, 17.0],
                "temperature_2m_min": [12.0, 13.0, 11.0],
                "weathercode": [1, 3, 61],
                "precipitation_sum": [0.0, 0.0, 4.2],
            },
        }

    def test_current_weather(self, base_config):
        from tools.weather import WeatherTool
        w = WeatherTool(base_config)
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = self._mock_response()
            result = w.run("current")
        assert "Berlin" in result
        assert "18" in result or "18.5" in result

    def test_forecast(self, base_config):
        from tools.weather import WeatherTool
        w = WeatherTool(base_config)
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = self._mock_response()
            result = w.run("forecast")
        assert "Today" in result or "Tomorrow" in result

    def test_cache(self, base_config):
        from tools.weather import WeatherTool
        w = WeatherTool(base_config)
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = self._mock_response()
            w.run("current")
            w.run("current")
            assert mock_get.call_count == 1   # second call uses cache


# ── Calendar tool tests ───────────────────────────────────────────────────────

class TestCalendarTool:
    _ics_content = """\
BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:Doctor Appointment
DTSTART:{today}
DTEND:{today}
END:VEVENT
END:VCALENDAR
""".format(today=datetime.now().strftime("%Y%m%d"))

    def test_today_event(self, base_config, tmp_path):
        ics = tmp_path / "cal.ics"
        ics.write_text(self._ics_content)
        base_config["calendar"]["ics_path"] = str(ics)
        from tools.calendar_tool import CalendarTool
        c = CalendarTool(base_config)
        result = c.run("today")
        assert "Doctor" in result

    def test_no_events_today(self, base_config, tmp_path):
        ics = tmp_path / "cal.ics"
        ics.write_text("BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR\n")
        base_config["calendar"]["ics_path"] = str(ics)
        from tools.calendar_tool import CalendarTool
        c = CalendarTool(base_config)
        result = c.run("today")
        assert "no events" in result.lower()

    def test_missing_ics(self, base_config):
        base_config["calendar"]["ics_path"] = "/nonexistent/calendar.ics"
        from tools.calendar_tool import CalendarTool
        c = CalendarTool(base_config)
        result = c.run("today")
        assert "no" in result.lower() or "couldn't" in result.lower() or "0" in result


# ── Home Assistant tests ──────────────────────────────────────────────────────

class TestHomeAssistantTool:
    def test_disabled_returns_message(self, base_config):
        from tools.home_assistant import HomeAssistantTool
        h = HomeAssistantTool(base_config)
        result = h.run("turn_on", entity="light.kitchen")
        assert "not enabled" in result.lower()

    def test_turn_on(self, base_config):
        base_config["home_assistant"]["enabled"] = True
        from tools.home_assistant import HomeAssistantTool
        h = HomeAssistantTool(base_config)
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.raise_for_status = MagicMock()
            result = h.run("turn_on", entity="light.kitchen")
        assert "kitchen" in result.lower() or "turned on" in result.lower()

    def test_get_state(self, base_config):
        base_config["home_assistant"]["enabled"] = True
        from tools.home_assistant import HomeAssistantTool
        h = HomeAssistantTool(base_config)
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.raise_for_status = MagicMock()
            mock_get.return_value.json.return_value = {
                "state": "on",
                "attributes": {"friendly_name": "Kitchen Light"},
            }
            result = h.run("get_state", entity="light.kitchen")
        assert "Kitchen Light" in result
        assert "on" in result


# ── Agent tests ───────────────────────────────────────────────────────────────

class TestAgent:
    def test_plain_reply(self, base_config):
        from core.agent import Agent
        a = Agent(base_config, {})
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "message": {"content": "Hello there!"}
            }
            result = a.process("Hello")
        assert result == "Hello there!"

    def test_tool_call_parsed(self, base_config):
        from core.agent import Agent
        mock_tool = MagicMock()
        mock_tool.DESCRIPTION = "A test tool."
        mock_tool.run.return_value = "Tool result text."

        a = Agent(base_config, {"mock": mock_tool})
        responses = iter([
            {"message": {"content": '{"tool": "mock", "action": "test"}'}},
            {"message": {"content": "Done!"}},
        ])

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.side_effect = lambda: next(responses)
            result = a.process("Do the thing")

        mock_tool.run.assert_called_once_with(action="test")
        assert result == "Done!"

    def test_history_reset(self, base_config):
        from core.agent import Agent
        a = Agent(base_config, {})
        a._history = [{"role": "user", "content": "test"}]
        a.reset()
        assert a._history == []

    def test_history_trimmed(self, base_config):
        from core.agent import Agent
        a = Agent(base_config, {})
        limit = base_config["system"]["conversation_history_limit"]
        a._history = [{"role": "user", "content": f"msg{i}"} for i in range(100)]
        a._trim_history()
        assert len(a._history) <= limit * 2

    def test_memory_injected_in_prompt(self, base_config):
        from core.agent import Agent
        from core.memory import Memory
        m = Memory(base_config)
        m.remember("favourite food", "pizza")
        a = Agent(base_config, {}, memory=m)
        prompt = a._build_system_prompt()
        assert "pizza" in prompt


# ── Volume tool tests ─────────────────────────────────────────────────────────

class TestVolumeTool:
    def test_set_volume(self, base_config):
        from tools.volume import VolumeTool
        v = VolumeTool(base_config)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = v.run("set", level=75)
        assert "75" in result

    def test_duck_unduck(self, base_config):
        from tools.volume import VolumeTool
        v = VolumeTool(base_config)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="[50%]", stderr="")
            v.duck()
            assert v._pre_duck_level is not None
            v.unduck()
            assert v._pre_duck_level is None


# ── TTS engine tests ──────────────────────────────────────────────────────────

class TestTTSEngine:
    def test_speak_calls_piper_and_aplay(self, base_config, tmp_path):
        # Create dummy piper binary and model so _check_binaries doesn't warn
        piper_bin = tmp_path / "piper"
        model = tmp_path / "model.onnx"
        cfg = tmp_path / "model.onnx.json"
        piper_bin.touch()
        model.touch()
        cfg.touch()
        base_config["tts"]["piper_binary"] = str(piper_bin)
        base_config["tts"]["model_path"] = str(model)
        base_config["tts"]["model_config"] = str(cfg)

        from engines.tts_engine import TTSEngine
        t = TTSEngine(base_config)

        with patch.object(t, "_synthesise") as mock_synth, \
             patch.object(t, "_play") as mock_play:
            t.speak("Hello world")
            mock_synth.assert_called_once()
            mock_play.assert_called_once()

    def test_interrupt_stops_playback(self, base_config, tmp_path):
        from engines.tts_engine import TTSEngine
        t = TTSEngine(base_config)
        t._interrupt_event.set()
        with patch.object(t, "_synthesise"), patch.object(t, "_play") as mock_play:
            t.speak("This should not play")
            mock_play.assert_not_called()

    def test_clean_strips_markdown(self, base_config):
        from engines.tts_engine import TTSEngine
        result = TTSEngine._clean("**Hello** _world_ `code` # heading")
        assert "*" not in result
        assert "_" not in result
        assert "`" not in result
        assert "#" not in result
