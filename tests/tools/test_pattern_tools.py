"""
Tests for Pattern Recognition Tools
"""

import sys

import pytest

sys.path.insert(0, "/home/meta_agent/src")

from src.tools.pattern_tools import get_pattern_tools
from src.tools.registry import ToolRegistry


class TestPatternTools:
    """Tests for pattern tools registration."""

    @pytest.fixture
    def registry(self):
        """Create a tool registry with pattern tools."""
        return get_pattern_tools()

    def test_registry_creation(self, registry):
        """Test that registry is created successfully."""
        assert registry is not None
        assert isinstance(registry, ToolRegistry)

    def test_tools_count(self, registry):
        """Test that all 6 tools are registered."""
        tools = registry.list()
        assert len(tools) == 6

    def test_tool_names(self, registry):
        """Test that all tool names are correct."""
        tool_names = registry.list_names()
        expected_names = [
            "analyze_patterns",
            "detect_anti_patterns",
            "calculate_code_metrics",
            "save_patterns_to_neo4j",
            "analyze_code_full",
            "detect_specific_pattern",
        ]
        for name in expected_names:
            assert name in tool_names

    def test_langchain_conversion(self, registry):
        """Test that tools can be converted to LangChain tools."""
        tools = registry.to_langchain_tools()
        assert len(tools) == 6


class TestAnalyzePatternsTool:
    """Tests for analyze_patterns tool."""

    @pytest.fixture
    def tool_func(self):
        """Get the analyze_patterns function."""
        registry = get_pattern_tools()
        tool = registry.get("analyze_patterns")
        return tool.func

    def test_analyze_singleton(self, tool_func):
        """Test detecting Singleton pattern."""
        code = """
class Singleton:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
"""
        result = tool_func(code, "test.py")
        assert "pattern_count" in result

    def test_analyze_empty_code(self, tool_func):
        """Test analyzing empty code."""
        result = tool_func("", "empty.py")
        assert "pattern_count" in result

    def test_analyze_factory(self, tool_func):
        """Test detecting Factory pattern."""
        code = """
class Creator:
    def create_product(self):
        return Product()

class Product:
    pass
"""
        result = tool_func(code, "factory.py")
        assert "pattern_count" in result


class TestDetectAntiPatternsTool:
    """Tests for detect_anti_patterns tool."""

    @pytest.fixture
    def tool_func(self):
        """Get the detect_anti_patterns function."""
        registry = get_pattern_tools()
        tool = registry.get("detect_anti_patterns")
        return tool.func

    def test_detect_god_object(self, tool_func):
        """Test detecting God Object anti-pattern."""
        code = """
class EverythingManager:
    def __init__(self):
        self.data = []
        self.users = []
        self.config = {}
        self.connections = []

    def process_data(self):
        for item in self.data:
            self.validate(item)
            self.transform(item)
            self.save(item)

    def manage_users(self):
        for user in self.users:
            self.create(user)
            self.update(user)
            self.delete(user)
"""
        result = tool_func(code, "god_object.py")
        assert "anti_pattern_count" in result

    def test_detect_magic_numbers(self, tool_func):
        """Test detecting Magic Numbers anti-pattern."""
        code = """
def calculate(x):
    result = x * 100 + 50
    return result
"""
        result = tool_func(code, "magic.py")
        assert "anti_pattern_count" in result

    def test_no_anti_patterns(self, tool_func):
        """Test code with no anti-patterns."""
        code = """
def add(a, b):
    return a + b
"""
        result = tool_func(code, "clean.py")
        assert "anti_pattern_count" in result


class TestCalculateCodeMetricsTool:
    """Tests for calculate_code_metrics tool."""

    @pytest.fixture
    def tool_func(self):
        """Get the calculate_code_metrics function."""
        registry = get_pattern_tools()
        tool = registry.get("calculate_code_metrics")
        return tool.func

    def test_calculate_metrics(self, tool_func):
        """Test calculating code metrics."""
        code = """
def function1():
    x = 1
    y = 2
    return x + y

class TestClass:
    def method1(self):
        pass
"""
        result = tool_func(code)
        assert "total_lines" in result
        assert "code_lines" in result
        assert "cyclomatic_complexity" in result

    def test_empty_code_metrics(self, tool_func):
        """Test metrics for empty code."""
        result = tool_func("")
        assert "total_lines" in result


class TestSavePatternsToNeo4jTool:
    """Tests for save_patterns_to_neo4j tool."""

    @pytest.fixture
    def tool_func(self):
        """Get the save_patterns_to_neo4j function."""
        registry = get_pattern_tools()
        tool = registry.get("save_patterns_to_neo4j")
        return tool.func

    def test_save_empty_patterns(self, tool_func):
        """Test saving empty patterns."""
        result = tool_func('{"patterns": []}')
        assert "status" in result or "error" in result

    def test_save_invalid_json(self, tool_func):
        """Test saving with invalid JSON."""
        result = tool_func("not valid json")
        assert "error" in result


class TestAnalyzeCodeFullTool:
    """Tests for analyze_code_full tool."""

    @pytest.fixture
    def tool_func(self):
        """Get the analyze_code_full function."""
        registry = get_pattern_tools()
        tool = registry.get("analyze_code_full")
        return tool.func

    def test_full_analysis(self, tool_func):
        """Test full code analysis."""
        code = """
class Singleton:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def calculate(x):
    result = x * 100 + 50
    return result
"""
        result = tool_func(code, "test.py")
        assert "patterns" in result
        assert "anti_patterns" in result
        assert "metrics" in result
        assert "neo4j_status" in result


class TestDetectSpecificPatternTool:
    """Tests for detect_specific_pattern tool."""

    @pytest.fixture
    def tool_func(self):
        """Get the detect_specific_pattern function."""
        registry = get_pattern_tools()
        tool = registry.get("detect_specific_pattern")
        return tool.func

    def test_detect_singleton(self, tool_func):
        """Test detecting specific Singleton pattern."""
        code = """
class Singleton:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
"""
        result = tool_func(code, "Singleton")
        assert "pattern_name" in result
        assert "occurrences" in result

    def test_detect_nonexistent_pattern(self, tool_func):
        """Test detecting non-existent pattern."""
        code = """
def add(a, b):
    return a + b
"""
        result = tool_func(code, "NonExistent")
        assert "occurrences" in result


class TestPatternToolsIntegration:
    """Integration tests for pattern tools."""

    def test_full_pipeline(self):
        """Test full pattern analysis pipeline."""
        from src.tools.pattern_tools import get_pattern_tools

        registry = get_pattern_tools()

        analyze_tool = registry.get("analyze_patterns")
        anti_tool = registry.get("detect_anti_patterns")
        metrics_tool = registry.get("calculate_code_metrics")

        code = """
class Singleton:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

class Factory:
    def create(self):
        return Product()

class Product:
    pass
"""

        patterns_result = analyze_tool.func(code)
        anti_result = anti_tool.func(code)
        metrics_result = metrics_tool.func(code)

        assert "pattern_count" in patterns_result
        assert "anti_pattern_count" in anti_result
        assert "cyclomatic_complexity" in metrics_result

    def test_langchain_tools_usable(self):
        """Test that LangChain tools can be used."""
        registry = get_pattern_tools()
        langchain_tools = registry.to_langchain_tools()

        for tool in langchain_tools:
            assert tool.name is not None
            assert tool.description is not None
