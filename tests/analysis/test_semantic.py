"""
Tests for Semantic Analyzer Module
"""

import pytest
import sys

sys.path.insert(0, "/home/meta_agent/src")

from src.analysis.semantic import (
    SemanticAnalyzer,
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
    SentimentType,
    SentimentAnalysis,
)


class TestSemanticAnalyzer:
    """Tests for SemanticAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a SemanticAnalyzer instance."""
        return SemanticAnalyzer()

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert hasattr(analyzer, "_company_patterns")
        assert hasattr(analyzer, "_technology_patterns")

    def test_extract_entities_basic(self, analyzer):
        """Test basic entity extraction."""
        text = "Apple Inc. develops AI technology using advanced algorithms."
        entities = analyzer.extract_entities(text)

        assert len(entities) > 0
        entity_texts = [e.text for e in entities]
        assert any("Apple" in t or "Inc" in t for t in entity_texts)

    def test_extract_technology_entities(self, analyzer):
        """Test technology entity extraction."""
        text = "The new AI system uses machine learning and natural language processing."
        entities = analyzer.extract_entities(text)

        entity_types = [e.type for e in entities]
        assert EntityType.TECHNOLOGY in entity_types

    def test_extract_material_entities(self, analyzer):
        """Test material entity extraction."""
        text = "Aluminum and steel are commonly used materials in manufacturing."
        entities = analyzer.extract_entities(text)

        entity_texts = [e.text.lower() for e in entities]
        assert any("aluminum" in t for t in entity_texts)
        assert any("steel" in t for t in entity_texts)

    def test_extract_process_entities(self, analyzer):
        """Test process entity extraction."""
        text = "The extrusion process involves heat treatment and forging."
        entities = analyzer.extract_entities(text)

        entity_types = [e.type for e in entities]
        assert EntityType.PROCESS in entity_types

    def test_entity_deduplication(self, analyzer):
        """Test that duplicate entities are removed."""
        text = "AI is short for Artificial Intelligence. AI transforms industries."
        entities = analyzer.extract_entities(text)

        unique_entities = set(e.text.lower() for e in entities)
        assert len(unique_entities) <= len(entities)

    def test_classify_entity(self, analyzer):
        """Test entity classification based on context."""
        entity = Entity(
            text="Neural Network",
            type=EntityType.UNKNOWN,
            start_pos=0,
            end_pos=14,
        )
        context = "This technology uses deep neural networks for processing."

        classified = analyzer.classify_entity(entity, context)

        assert classified == EntityType.TECHNOLOGY

    def test_normalize_entity(self, analyzer):
        """Test entity normalization."""
        entity = Entity(
            text="AI",
            type=EntityType.TECHNOLOGY,
            start_pos=0,
            end_pos=2,
        )

        normalized = analyzer.normalize_entity(entity)

        assert normalized == "Artificial Intelligence"

    def test_analyze_sentiment_positive(self, analyzer):
        """Test positive sentiment analysis."""
        text = "This is an innovative breakthrough with excellent results and promising future."
        sentiment = analyzer.analyze_sentiment(text)

        assert sentiment.sentiment in [SentimentType.POSITIVE, SentimentType.MIXED]
        assert sentiment.score > 0

    def test_analyze_sentiment_negative(self, analyzer):
        """Test negative sentiment analysis."""
        text = "This technology has many problems and challenges that are difficult to solve."
        sentiment = analyzer.analyze_sentiment(text)

        assert sentiment.sentiment in [SentimentType.NEGATIVE, SentimentType.MIXED]

    def test_analyze_sentiment_neutral(self, analyzer):
        """Test neutral sentiment analysis."""
        text = "The process involves heating and cooling the material."
        sentiment = analyzer.analyze_sentiment(text)

        assert sentiment.sentiment == SentimentType.NEUTRAL

    def test_extract_intent_announcement(self, analyzer):
        """Test intent extraction for announcements."""
        text = "Company XYZ announced new AI technology today."
        intent = analyzer.extract_intent(text)

        assert "announcement" in intent["detected_intents"]

    def test_extract_intent_research(self, analyzer):
        """Test intent extraction for research."""
        text = "The study found that the new method improves efficiency."
        intent = analyzer.extract_intent(text)

        assert "research" in intent["detected_intents"]

    def test_calculate_similarity(self, analyzer):
        """Test semantic similarity calculation."""
        text1 = "Machine learning algorithms for data analysis"
        text2 = "ML algorithms for processing data"

        similarity = analyzer.calculate_similarity(text1, text2)

        assert 0 <= similarity <= 1
        assert similarity > 0.3

    def test_calculate_similarity_identical(self, analyzer):
        """Test similarity for identical texts."""
        text = "Artificial intelligence is transforming industries"

        similarity = analyzer.calculate_similarity(text, text)

        assert similarity == 1.0

    def test_calculate_similarity_different(self, analyzer):
        """Test similarity for very different texts."""
        text1 = "Weather forecast for tomorrow"
        text2 = "Quantum physics principles"

        similarity = analyzer.calculate_similarity(text1, text2)

        assert similarity < 0.5

    def test_cluster_entities(self, analyzer):
        """Test entity clustering."""
        entities = [
            Entity(text="AI", type=EntityType.TECHNOLOGY, start_pos=0, end_pos=2),
            Entity(text="ML", type=EntityType.TECHNOLOGY, start_pos=0, end_pos=2),
            Entity(text="Aluminum", type=EntityType.MATERIAL, start_pos=0, end_pos=8),
            Entity(text="Steel", type=EntityType.MATERIAL, start_pos=0, end_pos=5),
        ]

        clusters = analyzer.cluster_entities(entities)

        assert len(clusters) == len(entities)

    def test_analyze_text_comprehensive(self, analyzer):
        """Test comprehensive text analysis."""
        text = """
        Apple Inc. announced a breakthrough in AI technology.
        The new system uses advanced machine learning algorithms.
        This development promises to improve efficiency significantly.
        """

        result = analyzer.analyze_text(text)

        assert "entities" in result
        assert "relationships" in result
        assert "sentiment" in result
        assert "entity_count" in result
        assert result["entity_count"] > 0


class TestEntity:
    """Tests for Entity class."""

    def test_entity_creation(self):
        """Test entity creation."""
        entity = Entity(
            text="Neural Network",
            type=EntityType.TECHNOLOGY,
            start_pos=0,
            end_pos=14,
            confidence=0.9,
        )

        assert entity.text == "Neural Network"
        assert entity.type == EntityType.TECHNOLOGY
        assert entity.confidence == 0.9

    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = Entity(
            text="AI",
            type=EntityType.TECHNOLOGY,
            start_pos=0,
            end_pos=2,
            properties={"source": "text"},
        )

        data = entity.to_dict()

        assert data["text"] == "AI"
        assert data["type"] == "technology"
        assert data["properties"] == {"source": "text"}

    def test_entity_from_dict(self):
        """Test entity deserialization."""
        data = {
            "text": "ML",
            "type": "technology",
            "start_pos": 0,
            "end_pos": 2,
            "confidence": 0.8,
            "properties": {},
        }

        entity = Entity.from_dict(data)

        assert entity.text == "ML"
        assert entity.type == EntityType.TECHNOLOGY


class TestRelationship:
    """Tests for Relationship class."""

    def test_relationship_creation(self):
        """Test relationship creation."""
        source = Entity(
            text="Company A",
            type=EntityType.COMPANY,
            start_pos=0,
            end_pos=9,
        )
        target = Entity(
            text="AI Technology",
            type=EntityType.TECHNOLOGY,
            start_pos=0,
            end_pos=13,
        )

        relationship = Relationship(
            source_entity=source,
            target_entity=target,
            relationship_type=RelationshipType.DEVELOPS,
            confidence=0.85,
        )

        assert relationship.relationship_type == RelationshipType.DEVELOPS
        assert relationship.confidence == 0.85

    def test_relationship_to_dict(self):
        """Test relationship serialization."""
        source = Entity(
            text="Apple",
            type=EntityType.COMPANY,
            start_pos=0,
            end_pos=5,
        )
        target = Entity(
            text="iPhone",
            type=EntityType.PRODUCT,
            start_pos=0,
            end_pos=7,
        )

        relationship = Relationship(
            source_entity=source,
            target_entity=target,
            relationship_type=RelationshipType.PRODUCEs,
        )

        data = relationship.to_dict()

        assert data["source"] == "Apple"
        assert data["target"] == "iPhone"
        assert data["type"] == "produces"


class TestSentimentAnalysis:
    """Tests for SentimentAnalysis class."""

    def test_sentiment_to_dict(self):
        """Test sentiment analysis serialization."""
        sentiment = SentimentAnalysis(
            sentiment=SentimentType.POSITIVE,
            score=0.8,
            confidence=0.9,
            strategic_implications=["Growth opportunity"],
        )

        data = sentiment.to_dict()

        assert data["sentiment"] == "positive"
        assert data["score"] == 0.8
        assert "Growth opportunity" in data["strategic_implications"]
