"""
Search Tools - Web search and information retrieval tools.
"""


from .registry import ToolRegistry, get_default_tool_registry


def get_search_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """
    Get search-related tools.

    Args:
        registry: Optional registry to add tools to

    Returns:
        ToolRegistry with search tools
    """
    if registry is None:
        registry = get_default_tool_registry()

    def web_search(query: str, num_results: int = 5) -> str:
        """
        Search the web for information.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            Search results as formatted string
        """
        try:
            import os

            from tavily import TavilyClient

            api_key = os.environ.get("TAVILY_API_KEY")
            if api_key:
                client = TavilyClient(api_key=api_key)
                results = client.search(query, max_results=num_results)
                return str(results)
            else:
                return f"Web search for: {query}\n\n(No Tavily API key configured. Set TAVILY_API_KEY environment variable.)"
        except ImportError:
            raise ImportError(
                "Tavily package is required for web_search. "
                "Install it with: pip install tavily-python"
            )

    def duckduckgo_search(query: str) -> str:
        """
        Search using DuckDuckGo.

        Args:
            query: Search query

        Returns:
            Search results
        """
        try:
            from langchain_community.tools import DuckDuckGoSearchResults

            search = DuckDuckGoSearchResults(num_results=5)
            result = search.run(query)
            return result
        except ImportError:
            raise ImportError(
                "langchain-community package is required for DuckDuckGo search. "
                "Install it with: pip install langchain-community"
            )

    registry.register(
        name="web_search",
        func=web_search,
        description="Search the web for information using Tavily",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    )

    registry.register(
        name="duckduckgo_search",
        func=duckduckgo_search,
        description="Search the web using DuckDuckGo",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    )

    return registry


def get_default_tools() -> ToolRegistry:
    """Get all default tools including search."""
    registry = get_default_tool_registry()
    registry = get_search_tools(registry)
    return registry
