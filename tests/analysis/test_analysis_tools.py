"""
Tests for Analysis Tools Module
"""

import sys

import pytest

sys.path.insert(0, "/home/meta_agent/src")

from src.tools.analysis import get_analysis_tools
from src.tools.registry import ToolRegistry


class TestAnalysisTools:
    """Tests for analysis tools registration."""

    @pytest.fixture
    def registry(self):
        """Create a tool registry with analysis tools."""
        return get_analysis_tools()

    def test_registry_creation(self, registry):
        """Test that registry is created successfully."""
        assert registry is not None
        assert isinstance(registry, ToolRegistry)

    def test_tools_count(self, registry):
        """Test that all 4 tools are registered."""
        tools = registry.list()
        assert len(tools) == 4

    def test_tool_names(self, registry):
        """Test that all tool names are correct."""
        tool_names = registry.list_names()
        expected_names = [
            "extract_entities",
            "analyze_sentiment",
            "save_entities_to_neo4j",
            "analyze_text_full",
        ]
        for name in expected_names:
            assert name in tool_names

    def test_extract_entities_tool(self, registry):
        """Test extract_entities tool definition."""
        tool = registry.get("extract_entities")
        assert tool is not None
        assert "extract" in tool.description.lower()
        assert "text" in tool.description.lower()

    def test_analyze_sentiment_tool(self, registry):
        """Test analyze_sentiment tool definition."""
        tool = registry.get("analyze_sentiment")
        assert tool is not None
        assert "sentiment" in tool.description.lower()

    def test_save_entities_to_neo4j_tool(self, registry):
        """Test save_entities_to_neo4j tool definition."""
        tool = registry.get("save_entities_to_neo4j")
        assert tool is not None
        assert "neo4j" in tool.description.lower()

    def test_analyze_text_full_tool(self, registry):
        """Test analyze_text_full tool definition."""
        tool = registry.get("analyze_text_full")
        assert tool is not None
        assert "full" in tool.description.lower() or "complete" in tool.description.lower()

    def test_langchain_conversion(self, registry):
        """Test that tools can be converted to LangChain tools."""
        tools = registry.to_langchain_tools()
        assert len(tools) == 4
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "invoke")

    def test_empty_registry_creation(self):
        """Test creating registry with None."""
        registry = get_analysis_tools(None)
        assert registry is not None
        assert len(registry) == 4


class TestExtractEntitiesTool:
    """Tests for extract_entities tool function."""

    @pytest.fixture
    def tool_func(self):
        """Get the extract_entities function."""
        registry = get_analysis_tools()
        tool = registry.get("extract_entities")
        return tool.func

    def test_extract_technology_entities(self, tool_func):
        """Test extracting technology entities."""
        text = "The AI system uses machine learning algorithms."
        result = tool_func(text)

        assert "entity_count" in result
        assert "entities" in result

    def test_extract_company_entities(self, tool_func):
        """Test extracting company entities."""
        text = "Apple Inc. develops innovative technology."
        result = tool_func(text)

        assert "entity_count" in result

    def test_extract_material_entities(self, tool_func):
        """Test extracting material entities."""
        text = "Aluminum and steel are commonly used materials."
        result = tool_func(text)

        assert "entity_count" in result

    def test_empty_text(self, tool_func):
        """Test extracting from empty text."""
        result = tool_func("")
        assert "entity_count" in result


class TestAnalyzeSentimentTool:
    """Tests for analyze_sentiment tool function."""

    @pytest.fixture
    def tool_func(self):
        """Get the analyze_sentiment function."""
        registry = get_analysis_tools()
        tool = registry.get("analyze_sentiment")
        return tool.func

    def test_positive_sentiment(self, tool_func):
        """Test positive sentiment analysis."""
        text = "This is an innovative breakthrough with excellent results."
        result = tool_func(text)

        assert "sentiment" in result
        assert "score" in result

    def test_negative_sentiment(self, tool_func):
        """Test negative sentiment analysis."""
        text = "This technology has many problems and challenges."
        result = tool_func(text)

        assert "sentiment" in result

    def test_neutral_sentiment(self, tool_func):
        """Test neutral sentiment analysis."""
        text = "The process involves heating and cooling."
        result = tool_func(text)

        assert "sentiment" in result

    def test_intent_detection(self, tool_func):
        """Test intent detection in sentiment analysis."""
        text = "Company announced new AI technology today."
        result = tool_func(text)

        assert "detected_intents" in result


class TestSaveEntitiesToNeo4jTool:
    """Tests for save_entities_to_neo4j tool function."""

    @pytest.fixture
    def tool_func(self):
        """Get the save_entities_to_neo4j function."""
        registry = get_analysis_tools()
        tool = registry.get("save_entities_to_neo4j")
        return tool.func

    def test_save_without_neo4j_config(self, tool_func):
        """Test saving when Neo4j is not configured."""
        import json

        entities = json.dumps({"entities": [{"text": "AI", "type": "technology"}]})
        result = tool_func(entities)

        assert "error" in result or "success" in result

    def test_save_empty_entities(self, tool_func):
        """Test saving empty entities list."""
        result = tool_func('{"entities": []}')

        assert "error" in result or "saved" in result

    def test_save_invalid_json(self, tool_func):
        """Test saving with invalid JSON."""
        result = tool_func("not valid json")

        assert "error" in result


class TestAnalyzeTextFullTool:
    """Tests for analyze_text_full tool function."""

    @pytest.fixture
    def tool_func(self):
        """Get the analyze_text_full function."""
        registry = get_analysis_tools()
        tool = registry.get("analyze_text_full")
        return tool.func

    def test_full_analysis(self, tool_func):
        """Test full text analysis."""
        text = "Apple Inc. announced new AI technology. This is an innovative breakthrough."
        result = tool_func(text)

        assert "entities" in result
        assert "sentiment" in result
        assert "neo4j_status" in result

    def test_full_analysis_empty_text(self, tool_func):
        """Test full analysis with empty text."""
        result = tool_func("")

        assert "entities" in result
        assert "sentiment" in result

    def test_full_analysis_with_source(self, tool_func):
        """Test full analysis with custom source name."""
        text = "AI technology is transforming industries."
        result = tool_func(text, "test_source")

        assert "neo4j_status" in result


class TestToolIntegration:
    """Integration tests for analysis tools."""

    def test_full_pipeline(self):
        """Test full analysis pipeline."""
        from src.tools.analysis import get_analysis_tools

        registry = get_analysis_tools()

        extract_tool = registry.get("extract_entities")
        sentiment_tool = registry.get("analyze_sentiment")

        text = "Google develops AI technology using advanced machine learning."

        entities_result = extract_tool.func(text)
        sentiment_result = sentiment_tool.func(text)

        assert "entity_count" in entities_result
        assert "sentiment" in sentiment_result

    def test_langchain_tools_usable(self):
        """Test that LangChain tools can be used."""
        from src.tools.analysis import get_analysis_tools

        registry = get_analysis_tools()
        langchain_tools = registry.to_langchain_tools()

        for tool in langchain_tools:
            assert tool.name is not None
            assert tool.description is not None
