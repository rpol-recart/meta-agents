"""Tests for the tool registry module."""

from src.tools.registry import ToolRegistry, get_default_tool_registry


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register(self):
        """Test registering a tool."""
        registry = ToolRegistry()

        def sample_func(x: int) -> str:
            return str(x)

        tool_def = registry.register(
            name="sample_tool",
            func=sample_func,
            description="A sample tool",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        )

        assert tool_def.name == "sample_tool"
        assert "sample_tool" in registry
        assert len(registry) == 1

    def test_get(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()

        def sample_func():
            return "result"

        registry.register(
            name="sample_tool",
            func=sample_func,
            description="A sample tool",
        )

        tool_def = registry.get("sample_tool")

        assert tool_def is not None
        assert tool_def.name == "sample_tool"

    def test_unregister(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()

        def sample_func():
            return "result"

        registry.register(
            name="sample_tool",
            func=sample_func,
            description="A sample tool",
        )

        removed = registry.unregister("sample_tool")

        assert removed is True
        assert "sample_tool" not in registry

    def test_list(self):
        """Test listing all tools."""
        registry = ToolRegistry()

        registry.register(name="tool1", func=lambda: None, description="Tool 1")
        registry.register(name="tool2", func=lambda: None, description="Tool 2")

        tools = registry.list()

        assert len(tools) == 2

    def test_list_names(self):
        """Test listing all tool names."""
        registry = ToolRegistry()

        registry.register(name="tool1", func=lambda: None, description="Tool 1")
        registry.register(name="tool2", func=lambda: None, description="Tool 2")

        names = registry.list_names()

        assert len(names) == 2
        assert "tool1" in names
        assert "tool2" in names

    def test_clear(self):
        """Test clearing all tools."""
        registry = ToolRegistry()

        registry.register(name="tool1", func=lambda: None, description="Tool 1")
        registry.register(name="tool2", func=lambda: None, description="Tool 2")

        registry.clear()

        assert len(registry) == 0

    def test_len(self):
        """Test length of registry."""
        registry = ToolRegistry()

        assert len(registry) == 0

        registry.register(name="tool1", func=lambda: None, description="Tool 1")
        assert len(registry) == 1

        registry.register(name="tool2", func=lambda: None, description="Tool 2")
        assert len(registry) == 2


class TestGetDefaultToolRegistry:
    """Tests for get_default_tool_registry function."""

    def test_returns_registry(self):
        """Test that it returns a ToolRegistry."""
        registry = get_default_tool_registry()

        assert isinstance(registry, ToolRegistry)

    def test_has_weather_tool(self):
        """Test that default registry has weather tool."""
        registry = get_default_tool_registry()

        assert "get_weather" in registry
        tool_def = registry.get("get_weather")
        assert "weather" in tool_def.description.lower()

    def test_has_calculate_tool(self):
        """Test that default registry has calculate tool."""
        registry = get_default_tool_registry()

        assert "calculate" in registry
        tool_def = registry.get("calculate")
        assert (
            "math" in tool_def.description.lower()
            or "calculation" in tool_def.description.lower()
        )

    def test_has_time_tool(self):
        """Test that default registry has time tool."""
        registry = get_default_tool_registry()

        assert "current_time" in registry

    def test_has_random_tool(self):
        """Test that default registry has random tool."""
        registry = get_default_tool_registry()

        assert "random_number" in registry
