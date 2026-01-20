"""
Analysis Module for Project Analysis Tool

This module provides semantic analysis and pattern recognition capabilities
for extracting insights from technical content.
"""

from .graph_population import GraphPopulationManager, GraphPopulationStats
from .patterns import (
    AntiPattern,
    Pattern,
    PatternRecognizer,
    PatternType,
    Severity,
)
from .semantic import (
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
    SemanticAnalyzer,
    SentimentAnalysis,
    SentimentType,
)
from .vector_integration import (
    ClusterResult,
    EmbeddingResult,
    EmbeddingService,
    SearchResult,
    VectorIntegrationService,
    VectorStore,
)

__all__ = [
    "SemanticAnalyzer",
    "Entity",
    "EntityType",
    "Relationship",
    "RelationshipType",
    "SentimentAnalysis",
    "SentimentType",
    "PatternRecognizer",
    "Pattern",
    "PatternType",
    "AntiPattern",
    "Severity",
    "GraphPopulationManager",
    "GraphPopulationStats",
    "VectorIntegrationService",
    "EmbeddingService",
    "VectorStore",
    "EmbeddingResult",
    "SearchResult",
    "ClusterResult",
]
