"""
Integration tests for agent tool workflows.

These tests verify that agent tools work correctly together
in realistic multi-step analysis scenarios.
"""

import sys
import os
import pytest
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from src.tools.registry import ToolRegistry
from src.tools.analysis import get_analysis_tools
from src.tools.pattern_tools import get_pattern_tools
from src.tools.dependency_tools import get_dependency_tools


class TestIntegrationAnalysisToGraph:
    """Test workflow: analysis -> entity extraction -> graph population."""

    def test_full_text_analysis_pipeline(self):
        """Test end-to-end text analysis with entity extraction and sentiment."""
        registry = ToolRegistry()
        tools = get_analysis_tools(registry)
        assert registry.get("extract_entities") is not None
        assert registry.get("analyze_sentiment") is not None
        assert registry.get("analyze_text_full") is not None

        extract_tool = registry.get("extract_entities")
        sentiment_tool = registry.get("analyze_sentiment")

        text = "OpenAI released GPT-4, a powerful language model. Google responded with new AI features."

        entities_result = extract_tool.func(text)
        entities = json.loads(entities_result)
        assert entities["entity_count"] > 0
        assert any(e["type"] == "technology" for e in entities["entities"])
        assert "AI" in entities["technologies"]

        sentiment_result = sentiment_tool.func(text)
        sentiment = json.loads(sentiment_result)
        assert "sentiment" in sentiment
        assert "score" in sentiment

    def test_analyze_text_full_combines_results(self):
        """Test that full analysis combines all individual tool results."""
        registry = ToolRegistry()
        tools = get_analysis_tools(registry)

        full_tool = registry.get("analyze_text_full")

        text = "Microsoft announced Azure AI services. IBM also announced new Watson features."

        result = full_tool.func(text)
        result_dict = json.loads(result)

        assert "entities" in result_dict
        assert "sentiment" in result_dict


class TestIntegrationPatternRecognition:
    """Test pattern recognition workflows."""

    def test_full_code_analysis_pipeline(self):
        """Test comprehensive code analysis combining patterns, anti-patterns, and metrics."""
        registry = ToolRegistry()
        tools = get_pattern_tools(registry)

        full_tool = registry.get("analyze_code_full")

        code = """
class DatabaseManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def connect(self):
        pass
    
    def query(self):
        pass

class UserManager(DatabaseManager):
    def __init__(self):
        self.users = []
        self.cache = []
        self.temp = []
        self.data = []
        self.buffer = []
        self.session_id = "user_session_id"
        self.config_version = 123
        self.max_retries = 5
        self.timeout_ms = 30000
        self.retry_count = 0
"""

        result = full_tool.func(code)
        result_dict = json.loads(result)

        assert "patterns" in result_dict
        assert "anti_patterns" in result_dict
        assert "metrics" in result_dict

    def test_detect_anti_patterns_in_code(self):
        """Test detecting anti-patterns in code."""
        registry = ToolRegistry()
        tools = get_pattern_tools(registry)

        anti_pattern_tool = registry.get("detect_anti_patterns")

        code = """
class DataManager:
    def __init__(self):
        self.data = []
        self.temp = []
        self.cache = []
        self.buffer = []
        self.session_id = "abc123"
        self.max_retries = 5
        self.timeout = 30000
        self.count = 0
"""

        result = anti_pattern_tool.func(code)
        result_dict = json.loads(result)

        assert "anti_patterns" in result_dict
        assert len(result_dict["anti_patterns"]) > 0


class TestIntegrationDependencyAnalysis:
    """Test dependency analysis workflows."""

    def test_dependencies_analyze_and_metrics(self):
        """Test analyzing dependencies and calculating metrics."""
        registry = ToolRegistry()
        tools = get_dependency_tools(registry)

        code = """import os
import sys
from typing import Dict, List
import json
import yaml

def load_config():
    pass

class DataProcessor:
    def __init__(self):
        self.data = {}
"""

        analyze_tool = registry.get("analyze_file_dependencies")
        build_graph_tool = registry.get("build_dependency_graph")
        metrics_tool = registry.get("calculate_dependency_metrics")

        analyze_result = analyze_tool.func(code)
        analyze_dict = json.loads(analyze_result)

        graph_result = build_graph_tool.func(code)
        graph_dict = json.loads(graph_result)

        metrics_result = metrics_tool.func(code)
        metrics_dict = json.loads(metrics_result)
        if "error" not in metrics_dict:
            assert "metrics" in metrics_dict

    def test_find_imports_in_code(self):
        """Test finding imports in code."""
        registry = ToolRegistry()
        tools = get_dependency_tools(registry)

        find_imports_tool = registry.get("find_imports")

        code = """import os
import sys
from typing import Dict, List, Optional
import json
import yaml
import pandas as pd
from datetime import datetime
"""

        result = find_imports_tool.func(code)
        result_dict = json.loads(result)

        assert "standard_lib" in result_dict["imports"]
        assert "third_party" in result_dict["imports"]
        assert "os" in result_dict["imports"]["standard_lib"]


class TestIntegrationMultiToolWorkflows:
    """Test workflows that use multiple tool categories together."""

    def test_code_analysis_with_dependency_check(self):
        """Test combining pattern analysis with dependency analysis."""
        pattern_registry = ToolRegistry()
        dependency_registry = ToolRegistry()

        get_pattern_tools(pattern_registry)
        get_dependency_tools(dependency_registry)

        code = """
import json
from datetime import datetime

class Logger:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def log(self, message):
        print(f"[{datetime.now()}] {message}")

class DataService:
    def __init__(self):
        self.logger = Logger.get_instance()
        self.cache = {}
    
    def get_data(self, key):
        if key in self.cache:
            return self.cache[key]
        result = self._fetch_from_db(key)
        self.cache[key] = result
        return result
    
    def _fetch_from_db(self, key):
        pass
"""

        pattern_tool = pattern_registry.get("analyze_code_full")
        dependency_tool = dependency_registry.get("analyze_file_dependencies")

        pattern_result = pattern_tool.func(code)
        pattern_dict = json.loads(pattern_result)

        dependency_result = dependency_tool.func(code)
        dependency_dict = json.loads(dependency_result)

        assert "patterns" in pattern_dict
        assert "dependencies" in dependency_dict

    def test_langchain_tool_conversion_all_categories(self):
        """Test that all tool categories can be converted to LangChain tools."""
        analysis_registry = ToolRegistry()
        pattern_registry = ToolRegistry()
        dependency_registry = ToolRegistry()

        get_analysis_tools(analysis_registry)
        get_pattern_tools(pattern_registry)
        get_dependency_tools(dependency_registry)

        analysis_lc = analysis_registry.to_langchain_tools()
        pattern_lc = pattern_registry.to_langchain_tools()
        dependency_lc = dependency_registry.to_langchain_tools()

        assert len(analysis_lc) == 4
        assert len(pattern_lc) == 6
        assert len(dependency_lc) == 6

        all_tools = analysis_lc + pattern_lc + dependency_lc
        tool_names = [tool.name for tool in all_tools]

        assert "extract_entities" in tool_names
        assert "analyze_patterns" in tool_names
        assert "find_imports" in tool_names


class TestIntegrationEndToEndScenarios:
    """Real-world end-to-end scenarios using agent tools."""

    def test_analyze_python_file_complete(self):
        """Complete analysis of a Python file using multiple tool types."""
        pattern_registry = ToolRegistry()
        dependency_registry = ToolRegistry()

        get_pattern_tools(pattern_registry)
        get_dependency_tools(dependency_registry)

        code = '''
"""
Data processing module for handling user analytics.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime

class UserAnalytics:
    """Singleton for user analytics tracking."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.data = {}
        return cls._instance
    
    def track_event(self, user_id: str, event: str, properties: Dict = None):
        """Track a user event."""
        if user_id not in self.data:
            self.data[user_id] = []
        self.data[user_id].append({
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "properties": properties or {}
        })
    
    def get_events(self, user_id: str) -> List[Dict]:
        """Get all events for a user."""
        return self.data.get(user_id, [])

class DataProcessor:
    """Process and transform data."""
    
    DEFAULT_BATCH_SIZE = 1000
    MAX_RETRY_COUNT = 3
    
    def __init__(self, batch_size: int = None):
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.retry_count = 0
        self.temp_storage = []
        self.cache_data = []
        self.buffer = []
        self.session = "session_12345"
'''

        pattern_tool = pattern_registry.get("analyze_code_full")
        dependency_tool = dependency_registry.get("analyze_file_dependencies")

        pattern_result = pattern_tool.func(code)
        pattern_dict = json.loads(pattern_result)

        dependency_result = dependency_tool.func(code)
        dependency_dict = json.loads(dependency_result)

        assert len(pattern_dict["patterns"]) > 0
        assert len(pattern_dict["anti_patterns"]) > 0
        assert "dependencies" in dependency_dict

    def test_analyze_text_for_entities_and_sentiment(self):
        """Analyze text for both sentiment and entity extraction."""
        registry = ToolRegistry()
        get_analysis_tools(registry)

        full_tool = registry.get("analyze_text_full")

        text = """Apple Inc. today announced record-breaking quarterly earnings,
surpassing analyst expectations. The tech giant reported revenue of $89.6 billion,
driven by strong iPhone and services sales. CEO Tim Cook stated that customer
demand remains robust across all product categories. Meanwhile, Samsung Electronics
reported moderate growth in their semiconductor division."""

        result = full_tool.func(text)
        result_dict = json.loads(result)

        assert "sentiment" in result_dict
        assert "entities" in result_dict
        assert len(result_dict["entities"]) > 0

    def test_calculate_code_metrics(self):
        """Test calculating code metrics for a code snippet."""
        registry = ToolRegistry()
        get_pattern_tools(registry)

        metrics_tool = registry.get("calculate_code_metrics")

        code = """
class MyClass:
    def __init__(self):
        self.x = 1
        self.y = 2
    
    def method_a(self):
        return self.x + self.y
    
    def method_b(self):
        return self.x * self.y
    
    def method_c(self):
        return self.x - self.y
"""

        result = metrics_tool.func(code)
        result_dict = json.loads(result)

        assert "code_lines" in result_dict
        assert "comment_lines" in result_dict
        assert result_dict["code_lines"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
