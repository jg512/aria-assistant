"""
tools/web_search.py — Privacy-friendly web search via DuckDuckGo
Enables Aria to answer questions about current events or general knowledge
without a local knowledge base.

Uses: duckduckgo_search (pip install duckduckgo_search)
"""

import logging
from typing import Optional

try:
    from duckduckgo_search import DDGS
    _HAS_DDGS = True
except ImportError:
    _HAS_DDGS = False
    logging.warning("duckduckgo_search not installed — web search tool disabled.")

logger = logging.getLogger(__name__)


class WebSearchTool:
    NAME = "web_search"
    DESCRIPTION = (
        "Search the web for current information or general knowledge. "
        "Actions: search [query], news [query]."
    )
    PARAMETERS = {
        "action": {
            "type": "string",
            "enum": ["search", "news"],
        },
        "query": {
            "type": "string",
            "description": "The search query (e.g., 'who won the Super Bowl?', 'current weather in London').",
        },
    }

    def __init__(self, config: dict):
        self.cfg = config.get("web_search", {})
        self.max_results = self.cfg.get("max_results", 3)

    def run(self, action: str, query: str, **_) -> str:
        if not _HAS_DDGS:
            return "Web search is currently unavailable because the necessary library is not installed."
        
        if not query:
            return "Please provide a search query."

        action = action.lower().strip()
        if action == "search":
            return self._search(query)
        elif action == "news":
            return self._news(query)
        
        return f"Unknown web search action: {action}"

    def _search(self, query: str) -> str:
        """Perform a general web search."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                if not results:
                    return f"I couldn't find any results for '{query}'."
                
                parts = []
                for i, r in enumerate(results, 1):
                    # Concatenate title and body for the LLM to summarize
                    parts.append(f"Result {i}: {r['title']}. {r['body']}")
                
                context = "\n".join(parts)
                return f"Search results for '{query}':\n{context}"
        except Exception as e:
            logger.error("Web search failed: %s", e)
            return f"I'm sorry, I encountered an error while searching: {e}"

    def _news(self, query: str) -> str:
        """Perform a news-specific search."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=self.max_results))
                if not results:
                    return f"I couldn't find any news for '{query}'."
                
                parts = []
                for i, r in enumerate(results, 1):
                    parts.append(f"News {i}: {r['title']} ({r['date']}). {r['body']}")
                
                context = "\n".join(parts)
                return f"Latest news for '{query}':\n{context}"
        except Exception as e:
            logger.error("News search failed: %s", e)
            return f"I'm sorry, I encountered an error while fetching news: {e}"
