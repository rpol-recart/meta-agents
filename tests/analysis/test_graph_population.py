"""
Tests for Graph Population Manager Module
"""

import pytest
import sys

sys.path.insert(0, "/home/meta_agent/src")

from src.analysis.graph_population import (
    GraphPopulationManager,
    GraphPopulationStats,
)


class TestGraphPopulationManager:
    """Tests for GraphPopulationManager class."""

    @pytest.fixture
    def manager(self):
        """Create a GraphPopulationManager instance."""
        return GraphPopulationManager()

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert manager.graph is not None
        assert manager.stats is not None
        assert manager._entity_node_map == {}

    def test_populate_from_entities(self, manager):
        """Test population from entities."""
        entities = [
            {
                "text": "Neural Network",
                "type": "technology",
                "confidence": 0.9,
                "properties": {"source": "article"},
            },
            {
                "text": "Deep Learning",
                "type": "technology",
                "confidence": 0.85,
            },
        ]

        stats = manager.populate_from_entities(entities, "source_id")

        assert stats.nodes_created == 2
        assert manager.graph.node_count == 2

    def test_populate_from_entities_with_mapping(self, manager):
        """Test entity type mapping."""
        entities = [
            {"text": "Apple Inc.", "type": "company", "confidence": 0.9},
            {"text": "Extrusion Process", "type": "process", "confidence": 0.8},
            {"text": "Aluminum", "type": "material", "confidence": 0.95},
        ]

        manager.populate_from_entities(entities)

        assert manager.graph.node_count == 3

    def test_populate_from_relationships(self, manager):
        """Test population from relationships."""
        entities = [
            {"text": "Company A", "type": "company"},
            {"text": "AI Technology", "type": "technology"},
        ]
        manager.populate_from_entities(entities)

        relationships = [
            {
                "source": "Company A",
                "target": "AI Technology",
                "type": "develops",
                "confidence": 0.9,
                "context": "Company A develops AI Technology",
            },
        ]

        stats = manager.populate_from_relationships(relationships)

        assert stats.edges_created == 1
        assert manager.graph.edge_count == 1

    def test_populate_from_patterns(self, manager):
        """Test population from patterns."""
        patterns = [
            {
                "name": "Singleton",
                "type": "creational",
                "description": "Ensure single instance",
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 10,
                "confidence": 0.9,
            },
        ]

        stats = manager.populate_from_patterns(patterns)

        assert stats.nodes_created >= 1

    def test_populate_from_anti_patterns(self, manager):
        """Test population from anti-patterns."""
        anti_patterns = [
            {
                "name": "God Object",
                "description": "Class knows too much",
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 50,
                "severity": "high",
                "impact": "High coupling",
                "remediation": "Split into smaller classes",
                "confidence": 0.9,
            },
        ]

        stats = manager.populate_from_anti_patterns(anti_patterns)

        assert stats.nodes_created >= 1

    def test_populate_full_analysis(self, manager):
        """Test population from complete analysis result."""
        analysis_result = {
            "entities": [
                {"text": "Neural Network", "type": "technology", "confidence": 0.9},
            ],
            "relationships": [],
            "patterns": [
                {
                    "name": "Observer",
                    "type": "behavioral",
                    "description": "Observer pattern",
                    "file_path": "observer.py",
                    "start_line": 1,
                    "end_line": 20,
                    "confidence": 0.85,
                },
            ],
            "anti_patterns": [],
        }

        stats = manager.populate_full_analysis(analysis_result, "test_source")

        assert manager.graph.node_count > 0

    def test_incremental_update(self, manager):
        """Test incremental update functionality."""
        initial_entities = [
            {"text": "AI", "type": "technology"},
        ]
        manager.populate_from_entities(initial_entities)
        initial_count = manager.graph.node_count

        new_entities = [
            {"text": "AI", "type": "technology"},
            {"text": "ML", "type": "technology"},
        ]
        new_relationships = [
            {"source": "AI", "target": "ML", "type": "uses"},
        ]

        stats = manager.incremental_update(new_entities, new_relationships)

        assert manager.graph.node_count == initial_count + 1
        assert stats.nodes_created == 1

    def test_detect_conflicts_no_conflicts(self, manager):
        """Test conflict detection with no conflicts."""
        entities = [
            {"text": "AI", "type": "technology"},
            {"text": "ML", "type": "technology"},
        ]
        manager.populate_from_entities(entities)

        conflicts = manager.detect_conflicts()

        assert len(conflicts) == 0

    def test_get_graph_stats(self, manager):
        """Test graph statistics retrieval."""
        entities = [
            {"text": "AI", "type": "technology"},
            {"text": "ML", "type": "technology"},
            {"text": "Company A", "type": "company"},
        ]
        manager.populate_from_entities(entities)

        stats = manager.get_graph_stats()

        assert "total_nodes" in stats
        assert "node_types" in stats
        assert stats["total_nodes"] == 3


class TestGraphPopulationStats:
    """Tests for GraphPopulationStats class."""

    def test_stats_initialization(self):
        """Test stats initialization."""
        stats = GraphPopulationStats()

        assert stats.nodes_created == 0
        assert stats.nodes_updated == 0
        assert stats.edges_created == 0
        assert stats.edges_updated == 0
        assert stats.conflicts_resolved == 0
        assert stats.errors == []

    def test_stats_to_dict(self):
        """Test stats serialization."""
        stats = GraphPopulationStats()
        stats.nodes_created = 5
        stats.edges_created = 3
        stats.errors.append("Test error")

        data = stats.to_dict()

        assert data["nodes_created"] == 5
        assert data["edges_created"] == 3
        assert "Test error" in data["errors"]

    def test_stats_duration(self):
        """Test stats duration calculation."""
        from datetime import datetime

        stats = GraphPopulationStats()
        stats.start_time = datetime.now()
        stats.end_time = datetime.now()

        data = stats.to_dict()

        assert data["duration_seconds"] is not None
        assert data["duration_seconds"] >= 0
