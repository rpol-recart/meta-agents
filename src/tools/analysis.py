"""
Analysis Tools for Agents - Integrates SemanticAnalyzer and PatternRecognizer
with Neo4j knowledge graph.
"""

import json
import logging
import os

from src.analysis import EntityType, SemanticAnalyzer
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def get_analysis_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Register all analysis tools for agents."""
    if registry is None:
        registry = ToolRegistry()

    semantic_analyzer = SemanticAnalyzer()

    def extract_entities(text: str) -> str:
        """Extract named entities (technologies, companies, materials, processes) from text."""
        try:
            entities = semantic_analyzer.extract_entities(text)
            result = {
                "entity_count": len(entities),
                "entities": [e.to_dict() for e in entities],
                "technologies": [e.text for e in entities if e.type == EntityType.TECHNOLOGY],
                "companies": [e.text for e in entities if e.type == EntityType.COMPANY],
                "materials": [e.text for e in entities if e.type == EntityType.MATERIAL],
                "processes": [e.text for e in entities if e.type == EntityType.PROCESS],
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return f'{{"error": "{e}"}}'

    def analyze_sentiment(text: str) -> str:
        """Analyze sentiment and strategic implications of technical text."""
        try:
            sentiment = semantic_analyzer.analyze_sentiment(text)
            intent = semantic_analyzer.extract_intent(text)
            result = {
                "sentiment": sentiment.sentiment.value,
                "score": sentiment.score,
                "confidence": sentiment.confidence,
                "strategic_implications": sentiment.strategic_implications,
                "detected_intents": intent["detected_intents"],
                "has_strategy": intent["has_strategy"],
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return f'{{"error": "{e}"}}'

    def save_entities_to_neo4j(entities_json: str, source: str = "agent") -> str:
        """Save extracted entities to Neo4j knowledge graph."""
        try:
            from neo4j import GraphDatabase

            uri = os.getenv("NEO4J_URI") or os.getenv("AURA_INSTANCEID", "")
            username = os.getenv("NEO4J_USERNAME", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            database = os.getenv("NEO4J_DATABASE", "neo4j")

            if not uri:
                return '{"error": "NEO4J_URI or AURA_INSTANCEID not configured"}'

            entities_data = json.loads(entities_json)
            if isinstance(entities_data, dict) and "entities" in entities_data:
                entities = entities_data["entities"]
            elif isinstance(entities_data, list):
                entities = entities_data
            else:
                entities = [entities_data]

            driver = GraphDatabase.driver(uri, auth=(username, password))
            with driver.session(database=database) as session:
                for entity in entities:
                    entity_text = entity.get("text", "")
                    if not entity_text:
                        continue
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type,
                            e.confidence = $confidence,
                            e.source = $source,
                            e.updated_at = datetime()
                        """,
                        name=entity_text,
                        type=entity.get("type", "unknown"),
                        confidence=entity.get("confidence", 1.0),
                        source=source,
                    )

            driver.close()
            return f'{{"status": "success", "saved": {len(entities)}}}'
        except ImportError:
            logger.error("neo4j driver not installed")
            return '{"error": "neo4j driver not installed"}'
        except Exception as e:
            logger.error(f"Neo4j save failed: {e}")
            return f'{{"error": "{e}"}}'

    def analyze_text_full(text: str, source_name: str = "agent") -> str:
        """Perform full semantic analysis: entities + sentiment + save to Neo4j."""
        try:
            entities = semantic_analyzer.extract_entities(text)
            sentiment = semantic_analyzer.analyze_sentiment(text)
            intent = semantic_analyzer.extract_intent(text)

            result = {
                "entities": [e.to_dict() for e in entities],
                "sentiment": sentiment.to_dict(),
                "intent": intent,
            }

            entities_json = json.dumps({"entities": result["entities"]}, ensure_ascii=False)
            save_result = save_entities_to_neo4j(entities_json, source_name)

            result["neo4j_status"] = save_result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Full analysis failed: {e}")
            return f'{{"error": "{e}"}}'

    registry.register(
        name="extract_entities",
        func=extract_entities,
        description="Extract named entities (technologies, companies, materials, processes) from text",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to analyze"}},
            "required": ["text"],
        },
    )

    registry.register(
        name="analyze_sentiment",
        func=analyze_sentiment,
        description="Analyze sentiment and strategic implications of technical text",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to analyze"}},
            "required": ["text"],
        },
    )

    registry.register(
        name="save_entities_to_neo4j",
        func=save_entities_to_neo4j,
        description="Save extracted entities to Neo4j knowledge graph",
        parameters={
            "type": "object",
            "properties": {
                "entities_json": {"type": "string", "description": "JSON with entities array"},
                "source": {"type": "string", "description": "Source identifier"},
            },
            "required": ["entities_json"],
        },
    )

    registry.register(
        name="analyze_text_full",
        func=analyze_text_full,
        description="Full semantic analysis: entities + sentiment + auto-save to Neo4j",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"},
                "source_name": {"type": "string", "description": "Source identifier for Neo4j"},
            },
            "required": ["text"],
        },
    )

    return registry
