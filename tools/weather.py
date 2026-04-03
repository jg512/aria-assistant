"""
tools/weather.py — Local weather via Open-Meteo (free, no API key)
Caches results to avoid hammering the API on repeated questions.
"""

import logging
import time
from datetime import datetime
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    80: "light showers", 81: "moderate showers", 82: "heavy showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
}


class WeatherTool:
    NAME = "weather"
    DESCRIPTION = (
        "Get current weather or a short forecast. "
        "Actions: current, forecast."
    )
    PARAMETERS = {
        "action": {
            "type": "string",
            "enum": ["current", "forecast"],
        }
    }

    def __init__(self, config: dict):
        self.cfg = config["weather"]
        self._cache: Optional[dict] = None
        self._cache_time: float = 0.0
        self._cache_ttl = self.cfg.get("cache_minutes", 30) * 60

    def reload_config(self, config: dict):
        self.cfg = config["weather"]
        self._cache_ttl = self.cfg.get("cache_minutes", 30) * 60
        self._cache = None  # Force refresh
        logger.info("WeatherTool: config reloaded.")

    def get_status(self) -> str:
        if self._cache:
            return f"{self._cache.get('temp', '?')}°  {self._cache.get('description', '')}"
        return "not fetched yet"

    def run(self, action: str = "current", **_) -> str:
        data = self._fetch()
        if data is None:
            return "I couldn't fetch the weather right now."

        if action == "current":
            return self._format_current(data)
        elif action == "forecast":
            return self._format_forecast(data)
        return "Unknown weather action."

    # ------------------------------------------------------------------

    def _fetch(self) -> Optional[dict]:
        now = time.monotonic()
        if self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        lat = self.cfg["latitude"]
        lon = self.cfg["longitude"]
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&hourly=relativehumidity_2m,precipitation_probability"
            f"&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&timezone=auto&forecast_days=3"
        )
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            logger.warning("Weather fetch failed: %s", e)
            return self._cache  # return stale cache if available

        cw = raw.get("current_weather", {})
        temp = cw.get("temperature")
        code = cw.get("weathercode", 0)
        wind = cw.get("windspeed")
        units = self.cfg.get("units", "celsius")
        temp_unit = "°C" if units == "celsius" else "°F"

        daily = raw.get("daily", {})
        self._cache = {
            "temp": temp,
            "temp_unit": temp_unit,
            "description": _WMO_CODES.get(code, "unknown conditions"),
            "wind_kph": wind,
            "location": self.cfg.get("location_name", "your location"),
            "daily_max": daily.get("temperature_2m_max", []),
            "daily_min": daily.get("temperature_2m_min", []),
            "daily_codes": daily.get("weathercode", []),
            "daily_precip": daily.get("precipitation_sum", []),
        }
        self._cache_time = now
        return self._cache

    def _format_current(self, d: dict) -> str:
        return (
            f"Currently in {d['location']}: {d['temp']}{d['temp_unit']} "
            f"with {d['description']} and winds of {d['wind_kph']} kilometres per hour."
        )

    def _format_forecast(self, d: dict) -> str:
        days = ["Today", "Tomorrow", "The day after"]
        parts = []
        for i, day in enumerate(days[:len(d["daily_max"])]):
            hi = d["daily_max"][i]
            lo = d["daily_min"][i]
            code = d["daily_codes"][i] if i < len(d["daily_codes"]) else 0
            desc = _WMO_CODES.get(code, "variable")
            parts.append(f"{day}: {lo}–{hi}{d['temp_unit']}, {desc}")
        return ". ".join(parts) + "."
