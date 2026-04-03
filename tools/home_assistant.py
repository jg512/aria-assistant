"""
tools/home_assistant.py — Control Home Assistant entities via REST API
Supports lights, switches, covers, climate, and generic state queries.
Set home_assistant.enabled=true and your token in config.json to activate.
"""

import logging
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


class HomeAssistantTool:
    NAME = "home_assistant"
    DESCRIPTION = (
        "Control smart home devices via Home Assistant. "
        "Actions: turn_on, turn_off, toggle, set_brightness [entity, brightness 0-255], "
        "set_temperature [entity, temperature], get_state [entity]."
    )
    PARAMETERS = {
        "action": {
            "type": "string",
            "enum": ["turn_on", "turn_off", "toggle", "set_brightness", "set_temperature", "get_state"],
        },
        "entity": {
            "type": "string",
            "description": "Home Assistant entity ID, e.g. light.living_room or switch.desk_lamp.",
        },
        "brightness": {
            "type": "integer",
            "description": "Brightness 0–255 for lights.",
        },
        "temperature": {
            "type": "number",
            "description": "Target temperature for climate entities.",
        },
    }

    def __init__(self, config: dict):
        self.cfg = config.get("home_assistant", {})
        self.reload_config(config)

    def reload_config(self, config: dict):
        self.cfg = config.get("home_assistant", {})
        self.enabled = self.cfg.get("enabled", False)
        self.host = self.cfg.get("host", "http://homeassistant.local:8123")
        self.token = self.cfg.get("token", "")
        self.timeout = self.cfg.get("timeout", 5)
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        logger.info("HomeAssistantTool: config reloaded.")

    def get_status(self) -> str:
        return "connected" if self.enabled else "disabled"

    def run(self, action: str, entity: str = "", brightness: int = 255,
            temperature: float = 21.0, **_) -> str:
        if not self.enabled:
            return "Home Assistant integration is not enabled. Set home_assistant.enabled=true in config.json."
        if not self.token or self.token == "YOUR_LONG_LIVED_ACCESS_TOKEN_HERE":
            return "Home Assistant token is not configured. Please add your token to config.json."

        try:
            if action == "get_state":
                return self._get_state(entity)
            elif action in ("turn_on", "turn_off", "toggle"):
                return self._call_service(action, entity)
            elif action == "set_brightness":
                return self._call_service("turn_on", entity, {"brightness": brightness})
            elif action == "set_temperature":
                return self._call_service("set_temperature", entity,
                                         {"temperature": temperature}, domain="climate")
            return f"Unknown action: {action}"
        except requests.exceptions.ConnectionError:
            return "I can't reach Home Assistant right now."
        except Exception as e:
            logger.error("HA error: %s", e)
            return f"Home Assistant error: {e}"

    # ------------------------------------------------------------------

    def _get_state(self, entity: str) -> str:
        r = requests.get(
            f"{self.host}/api/states/{entity}",
            headers=self._headers, timeout=self.timeout
        )
        r.raise_for_status()
        data = r.json()
        state = data.get("state", "unknown")
        friendly = data.get("attributes", {}).get("friendly_name", entity)
        attrs = data.get("attributes", {})

        extra = ""
        if "temperature" in attrs:
            extra = f" at {attrs['temperature']}°"
        elif "brightness" in attrs:
            pct = round(attrs["brightness"] / 255 * 100)
            extra = f" at {pct}% brightness"

        return f"{friendly} is {state}{extra}."

    def _call_service(self, service: str, entity: str,
                      extra_data: Optional[dict] = None, domain: str = "") -> str:
        # Use the universal 'homeassistant' domain for common power services
        # This works for any entity (light, switch, fan, etc.) in HA.
        if service in ("turn_on", "turn_off", "toggle"):
            service_domain = "homeassistant"
        else:
            service_domain = domain or (entity.split(".")[0] if "." in entity else "homeassistant")

        payload: dict[str, Any] = {"entity_id": entity}
        if extra_data:
            payload.update(extra_data)

        r = requests.post(
            f"{self.host}/api/services/{service_domain}/{service}",
            headers=self._headers, json=payload, timeout=self.timeout
        )
        r.raise_for_status()

        friendly = entity.replace("_", " ").replace(".", " ").title()
        action_past = {
            "turn_on": "turned on",
            "turn_off": "turned off",
            "toggle": "toggled",
            "set_temperature": "temperature set",
        }.get(service, service)

        return f"{friendly} {action_past}."
