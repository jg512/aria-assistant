"""
tools/news.py — RSS headline reader with per-feed caching
Uses only the standard library xml.etree for parsing (no feedparser dep).
"""

import logging
import time
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


class NewsTool:
    NAME = "news"
    DESCRIPTION = (
        "Read the latest news headlines. "
        "Actions: headlines [source?]. Source can be a feed name or 'all'."
    )
    PARAMETERS = {
        "action": {
            "type": "string",
            "enum": ["headlines"],
        },
        "source": {
            "type": "string",
            "description": "Feed name (e.g. 'BBC World') or 'all'.",
        },
    }

    def __init__(self, config: dict):
        self.cfg = config["news"]
        self._cache: dict[str, tuple[float, list[str]]] = {}  # url -> (timestamp, headlines)
        self._ttl = self.cfg.get("cache_minutes", 60) * 60
        self._max = self.cfg.get("max_headlines", 5)

    def reload_config(self, config: dict):
        self.cfg = config["news"]
        self._ttl = self.cfg.get("cache_minutes", 60) * 60
        self._max = self.cfg.get("max_headlines", 5)
        # Clear cache to pick up potentially new feeds
        self._cache = {}
        logger.info("NewsTool: config reloaded.")

    def run(self, action: str = "headlines", source: str = "all", **_) -> str:
        headlines = self._get_headlines(source)
        if not headlines:
            return "I couldn't fetch any news right now."
        if len(headlines) == 1:
            return f"The top story: {headlines[0]}."
        intro = "Here are the top headlines. "
        return intro + " Next: ".join(headlines[:self._max]) + "."

    # ------------------------------------------------------------------

    def _get_headlines(self, source: str) -> list[str]:
        feeds = self.cfg.get("feeds", [])
        if source and source.lower() != "all":
            feeds = [f for f in feeds if source.lower() in f["name"].lower()]

        all_headlines = []
        for feed in feeds:
            all_headlines.extend(self._fetch_feed(feed["url"]))

        return all_headlines[:self._max]

    def _fetch_feed(self, url: str) -> list[str]:
        now = time.monotonic()
        cached = self._cache.get(url)
        if cached and (now - cached[0]) < self._ttl:
            return cached[1]

        try:
            with urlopen(url, timeout=5) as resp:
                content = resp.read()
            
            # Defensive parsing: strip leading/trailing whitespace which breaks some XML parsers
            content_str = content.decode("utf-8", errors="replace").strip()
            root = ET.fromstring(content_str)
        except Exception as e:
            logger.warning("News: failed to fetch or parse %s — %s", url, e)
            return cached[1] if cached else []

        headlines = []
        # Support common namespaces
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "dc": "http://purl.org/dc/elements/1.1/"
        }

        # 1. Try RSS format (item -> title)
        for item in root.iter("item"):
            title = item.findtext("title", "").strip()
            if not title:
                # Some feeds use dc:title
                title = item.findtext("{http://purl.org/dc/elements/1.1/}title", "").strip()
            if title:
                headlines.append(title)

        # 2. Try Atom format (entry -> title) if no RSS items found
        if not headlines:
            # Look for atom entries
            entries = root.findall(".//atom:entry", ns) or root.findall(".//entry")
            for entry in entries:
                t_elem = entry.find("atom:title", ns) or entry.find("title")
                if t_elem is not None and t_elem.text:
                    headlines.append(t_elem.text.strip())

        # Clean up any HTML entities that might be in the titles
        import html
        headlines = [html.unescape(h) for h in headlines]

        self._cache[url] = (now, headlines)
        return headlines
