"""
Pattern Recognition Tools for Agents

This module provides tools for detecting design patterns, anti-patterns,
and code quality metrics.
"""

import json
import logging

from src.analysis import PatternRecognizer
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def get_pattern_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Register all pattern recognition tools for agents."""
    if registry is None:
        registry = ToolRegistry()

    recognizer = PatternRecognizer()

    def analyze_patterns(code: str, file_path: str = "unknown") -> str:
        """Detect design patterns in code (Singleton, Factory, Observer, etc.)."""
        try:
            patterns = recognizer.detect_patterns(code, file_path)
            result = {
                "pattern_count": len(patterns),
                "patterns": [p.to_dict() for p in patterns],
                "pattern_types": list(set(p.pattern_type.value for p in patterns)),
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return f'{{"error": "{e}"}}'

    def detect_anti_patterns(code: str, file_path: str = "unknown") -> str:
        """Detect anti-patterns in code (God Object, Spaghetti Code, Magic Numbers, etc.)."""
        try:
            anti_patterns = recognizer.detect_anti_patterns(code, file_path)
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
            for ap in anti_patterns:
                severity_counts[ap.severity.value] += 1

            result = {
                "anti_pattern_count": len(anti_patterns),
                "anti_patterns": [a.to_dict() for a in anti_patterns],
                "severity_counts": severity_counts,
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Anti-pattern detection failed: {e}")
            return f'{{"error": "{e}"}}'

    def calculate_code_metrics(code: str) -> str:
        """Calculate code quality metrics (complexity, lines, comments, etc.)."""
        try:
            metrics = recognizer.calculate_code_metrics(code)
            return json.dumps(metrics, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Code metrics calculation failed: {e}")
            return f'{{"error": "{e}"}}'

    def save_patterns_to_neo4j(patterns_json: str, source: str = "agent") -> str:
        """Save detected patterns to Neo4j knowledge graph."""
        try:
            import os

            from neo4j import GraphDatabase

            uri = os.getenv("NEO4J_URI") or os.getenv("AURA_INSTANCEID", "")
            username = os.getenv("NEO4J_USERNAME", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            database = os.getenv("NEO4J_DATABASE", "neo4j")

            if not uri:
                return '{"error": "NEO4J_URI or AURA_INSTANCEID not configured"}'

            patterns_data = json.loads(patterns_json)
            patterns = patterns_data.get("patterns", patterns_data)

            if not patterns:
                return '{"status": "success", "saved": 0}'

            driver = GraphDatabase.driver(uri, auth=(username, password))
            with driver.session(database=database) as session:
                for pattern in patterns:
                    pattern_name = pattern.get("name", "")
                    if not pattern_name:
                        continue
                    session.run(
                        """
                        MERGE (p:Pattern {name: $name})
                        SET p.type = $type,
                            p.description = $description,
                            p.file_path = $file_path,
                            p.confidence = $confidence,
                            p.source = $source,
                            p.updated_at = datetime()
                        """,
                        name=pattern_name,
                        type=pattern.get("type", "unknown"),
                        description=pattern.get("description", ""),
                        file_path=pattern.get("file_path", ""),
                        confidence=pattern.get("confidence", 1.0),
                        source=source,
                    )

            driver.close()
            return f'{{"status": "success", "saved": {len(patterns)}}}'
        except ImportError:
            logger.error("neo4j driver not installed")
            return '{"error": "neo4j driver not installed"}'
        except Exception as e:
            logger.error(f"Neo4j save failed: {e}")
            return f'{{"error": "{e}"}}'

    def analyze_code_full(code: str, file_path: str = "agent") -> str:
        """Full pattern analysis: patterns + anti-patterns + metrics + auto-save to Neo4j."""
        try:
            result = recognizer.analyze_code(code, file_path)

            patterns_json = json.dumps({"patterns": result["patterns"]}, ensure_ascii=False)
            save_result = save_patterns_to_neo4j(patterns_json, source=file_path)
            result["neo4j_status"] = save_result

            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Full pattern analysis failed: {e}")
            return f'{{"error": "{e}"}}'

    def detect_specific_pattern(code: str, pattern_name: str) -> str:
        """Detect a specific design pattern by name."""
        try:
            patterns = recognizer.detect_patterns(code, "unknown")
            matching = [p.to_dict() for p in patterns if p.name.lower() == pattern_name.lower()]
            return json.dumps(
                {
                    "pattern_name": pattern_name,
                    "occurrences": len(matching),
                    "matches": matching,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            logger.error(f"Specific pattern detection failed: {e}")
            return f'{{"error": "{e}"}}'

    registry.register(
        name="analyze_patterns",
        func=analyze_patterns,
        description="Detect design patterns in code (Singleton, Factory, Observer, Strategy, etc.)",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Source code to analyze"},
                "file_path": {"type": "string", "description": "File path for reference"},
            },
            "required": ["code"],
        },
    )

    registry.register(
        name="detect_anti_patterns",
        func=detect_anti_patterns,
        description="Detect anti-patterns in code (God Object, Spaghetti Code, Magic Numbers, Long Method, etc.)",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Source code to analyze"},
                "file_path": {"type": "string", "description": "File path for reference"},
            },
            "required": ["code"],
        },
    )

    registry.register(
        name="calculate_code_metrics",
        func=calculate_code_metrics,
        description="Calculate code quality metrics (lines, complexity, comments, indentation, docstrings)",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Source code to analyze"},
            },
            "required": ["code"],
        },
    )

    registry.register(
        name="save_patterns_to_neo4j",
        func=save_patterns_to_neo4j,
        description="Save detected patterns to Neo4j knowledge graph",
        parameters={
            "type": "object",
            "properties": {
                "patterns_json": {"type": "string", "description": "JSON with patterns array"},
                "source": {"type": "string", "description": "Source identifier"},
            },
            "required": ["patterns_json"],
        },
    )

    registry.register(
        name="analyze_code_full",
        func=analyze_code_full,
        description="Full pattern analysis: patterns + anti-patterns + metrics + auto-save to Neo4j",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Source code to analyze"},
                "file_path": {"type": "string", "description": "Source identifier for Neo4j"},
            },
            "required": ["code"],
        },
    )

    registry.register(
        name="detect_specific_pattern",
        func=detect_specific_pattern,
        description="Detect a specific design pattern by name (e.g., 'Singleton', 'Factory', 'Observer')",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Source code to analyze"},
                "pattern_name": {"type": "string", "description": "Pattern name to detect"},
            },
            "required": ["code", "pattern_name"],
        },
    )

    return registry
