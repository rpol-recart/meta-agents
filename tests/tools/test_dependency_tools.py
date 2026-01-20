"""
Tests for Dependency Analysis Tools
"""

import sys

import pytest

sys.path.insert(0, "/home/meta_agent/src")

from src.tools.dependency_tools import get_dependency_tools
from src.tools.registry import ToolRegistry


class TestDependencyTools:
    """Tests for dependency tools registration."""

    @pytest.fixture
    def registry(self):
        """Create a tool registry with dependency tools."""
        return get_dependency_tools()

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
            "find_imports",
            "analyze_file_dependencies",
            "build_dependency_graph",
            "detect_circular_dependencies",
            "calculate_dependency_metrics",
            "save_dependencies_to_neo4j",
        ]
        for name in expected_names:
            assert name in tool_names

    def test_langchain_conversion(self, registry):
        """Test that tools can be converted to LangChain tools."""
        tools = registry.to_langchain_tools()
        assert len(tools) == 6


class TestFindImportsTool:
    """Tests for find_imports tool."""

    @pytest.fixture
    def tool_func(self):
        """Get the find_imports function."""
        registry = get_dependency_tools()
        tool = registry.get("find_imports")
        return tool.func

    def test_find_standard_lib_imports(self, tool_func):
        """Test finding standard library imports."""
        code = """
import os
import sys
import json
from datetime import datetime
"""
        result = tool_func(code)
        assert "import_count" in result
        assert "imports" in result

    def test_find_third_party_imports(self, tool_func):
        """Test finding third-party imports."""
        code = """
import requests
from flask import Flask
from langchain import OpenAI
"""
        result = tool_func(code)
        assert "third_party" in result

    def test_find_local_imports(self, tool_func):
        """Test finding local imports."""
        code = """
from .module import Class
from ..parent import Function
"""
        result = tool_func(code)
        assert "local" in result or "relative" in result

    def test_find_no_imports(self, tool_func):
        """Test code with no imports."""
        code = """
def hello():
    return "Hello, World!"
"""
        result = tool_func(code)
        assert "import_count" in result


class TestAnalyzeFileDependenciesTool:
    """Tests for analyze_file_dependencies tool."""

    @pytest.fixture
    def tool_func(self):
        """Get the analyze_file_dependencies function."""
        registry = get_dependency_tools()
        tool = registry.get("analyze_file_dependencies")
        return tool.func

    def test_analyze_simple_file(self, tool_func):
        """Test analyzing a simple file."""
        code = """
import os
from datetime import datetime

class MyClass:
    def method(self):
        pass
"""
        result = tool_func(code, "test.py")
        assert "file_path" in result
        assert "dependencies" in result

    def test_analyze_empty_file(self, tool_func):
        """Test analyzing an empty file."""
        result = tool_func("", "empty.py")
        assert "file_path" in result


class TestBuildDependencyGraphTool:
    """Tests for build_dependency_graph tool."""

    @pytest.fixture
    def tool_func(self):
        """Get the build_dependency_graph function."""
        registry = get_dependency_tools()
        tool = registry.get("build_dependency_graph")
        return tool.func

    def test_build_simple_graph(self, tool_func):
        """Test building a simple dependency graph."""
        files = {
            "main.py": "import module_a",
            "module_a.py": "import module_b",
            "module_b.py": "",
        }
        result = tool_func(files)
        assert "nodes" in result
        assert "edges" in result
        assert "node_count" in result
        assert "edge_count" in result

    def test_build_empty_graph(self, tool_func):
        """Test building a graph with no files."""
        result = tool_func({})
        assert "node_count" in result
        assert "edge_count" in result


class TestDetectCircularDependenciesTool:
    """Tests for detect_circular_dependencies tool."""

    @pytest.fixture
    def tool_func(self):
        """Get the detect_circular_dependencies function."""
        registry = get_dependency_tools()
        tool = registry.get("detect_circular_dependencies")
        return tool.func

    def test_detect_no_circular(self, tool_func):
        """Test detecting no circular dependencies."""
        files = {
            "main.py": "import a",
            "a.py": "import b",
            "b.py": "",
        }
        result = tool_func(files)
        import json

        data = json.loads(result)
        assert not data["has_circular_dependencies"]

    def test_detect_circular(self, tool_func):
        """Test detecting circular dependencies."""
        files = {
            "main.py": "from module_a import A",
            "module_a.py": "from main import MainClass",
        }
        result = tool_func(files)
        import json

        data = json.loads(result)
        assert "has_circular_dependencies" in data
        assert "circular_count" in data

    def test_detect_empty(self, tool_func):
        """Test detecting with empty files dict."""
        result = tool_func({})
        assert "has_circular_dependencies" in result


class TestCalculateDependencyMetricsTool:
    """Tests for calculate_dependency_metrics tool."""

    @pytest.fixture
    def tool_func(self):
        """Get the calculate_dependency_metrics function."""
        registry = get_dependency_tools()
        tool = registry.get("calculate_dependency_metrics")
        return tool.func

    def test_calculate_metrics(self, tool_func):
        """Test calculating dependency metrics."""
        files = {
            "main.py": "import os import sys",
            "a.py": "import requests",
            "b.py": "from main import func",
        }
        result = tool_func(files)
        assert "total_files" in result
        assert "unique_external_imports" in result
        assert "average_imports_per_file" in result

    def test_calculate_empty_metrics(self, tool_func):
        """Test calculating metrics for empty project."""
        result = tool_func({})
        import json

        data = json.loads(result)
        assert "total_files" in data
        assert data["total_files"] == 0


class TestSaveDependenciesToNeo4jTool:
    """Tests for save_dependencies_to_neo4j tool."""

    @pytest.fixture
    def tool_func(self):
        """Get the save_dependencies_to_neo4j function."""
        registry = get_dependency_tools()
        tool = registry.get("save_dependencies_to_neo4j")
        return tool.func

    def test_save_empty_dependencies(self, tool_func):
        """Test saving empty dependencies."""
        result = tool_func('{"nodes": [], "edges": []}')
        assert "status" in result or "error" in result

    def test_save_invalid_json(self, tool_func):
        """Test saving with invalid JSON."""
        result = tool_func("not valid json")
        assert "error" in result


class TestDependencyToolsIntegration:
    """Integration tests for dependency tools."""

    def test_full_pipeline(self):
        """Test full dependency analysis pipeline."""
        from src.tools.dependency_tools import get_dependency_tools

        registry = get_dependency_tools()

        files = {
            "main.py": "import os\nfrom datetime import datetime\nfrom module_a import ClassA",
            "module_a.py": "import requests\nfrom module_b import ClassB",
            "module_b.py": "import json",
        }

        find_tool = registry.get("find_imports")
        graph_tool = registry.get("build_dependency_graph")
        circular_tool = registry.get("detect_circular_dependencies")
        metrics_tool = registry.get("calculate_dependency_metrics")

        imports_result = find_tool.func(files["main.py"])
        graph_result = graph_tool.func(files)
        circular_result = circular_tool.func(files)
        metrics_result = metrics_tool.func(files)

        assert "import_count" in imports_result
        assert "node_count" in graph_result
        assert "has_circular_dependencies" in circular_result
        assert "total_files" in metrics_result

    def test_langchain_tools_usable(self):
        """Test that LangChain tools can be used."""
        registry = get_dependency_tools()
        langchain_tools = registry.to_langchain_tools()

        for tool in langchain_tools:
            assert tool.name is not None
            assert tool.description is not None
