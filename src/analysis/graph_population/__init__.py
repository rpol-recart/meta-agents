"""
Graph Population Manager for Project Analysis Tool

This module provides mechanisms for integrating analysis results
into the Neo4j knowledge graph.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.graph import Edge, EdgeType, HybridGraph, Node, NodeType

logger = logging.getLogger(__name__)


@dataclass
class GraphPopulationStats:
    """Statistics for graph population operations."""

    nodes_created: int = 0
    nodes_updated: int = 0
    edges_created: int = 0
    edges_updated: int = 0
    conflicts_resolved: int = 0
    errors: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes_created": self.nodes_created,
            "nodes_updated": self.nodes_updated,
            "edges_created": self.edges_created,
            "edges_updated": self.edges_updated,
            "conflicts_resolved": self.conflicts_resolved,
            "errors": self.errors,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.start_time and self.end_time
                else None
            ),
        }


class GraphPopulationManager:
    """
    Manages population of the knowledge graph with analysis results.

    This class provides:
    - Incremental graph updates
    - Conflict detection and resolution
    - Batch population operations
    - Graph consistency management
    """

    def __init__(self, graph: HybridGraph | None = None):
        """
        Initialize the graph population manager.

        Args:
            graph: Optional HybridGraph instance to manage
        """
        self.graph = graph or HybridGraph()
        self.stats = GraphPopulationStats()
        self._node_id_counter = 0
        self._entity_node_map: dict[str, int] = {}

    def populate_from_entities(
        self,
        entities: list[dict[str, Any]],
        source_id: str | None = None,
    ) -> GraphPopulationStats:
        """
        Populate graph with entities from semantic analysis.

        Args:
            entities: List of entity dictionaries
            source_id: Optional source identifier for the entities

        Returns:
            GraphPopulationStats with operation statistics
        """
        self.stats = GraphPopulationStats()
        self.stats.start_time = datetime.now()

        for entity_data in entities:
            try:
                self._add_entity_to_graph(entity_data, source_id)
            except Exception as e:
                self.stats.errors.append(f"Failed to add entity: {e}")

        self.stats.end_time = datetime.now()
        return self.stats

    def _add_entity_to_graph(self, entity_data: dict[str, Any], source_id: str | None) -> None:
        """Add a single entity to the graph."""
        entity_text = entity_data.get("text", "")
        entity_type_str = entity_data.get("type", "unknown")
        entity_type = self._map_entity_type(entity_type_str)

        node_id = self._get_or_create_node_id(entity_text, entity_type)

        existing_node = self.graph.get_node(node_id)
        if existing_node:
            existing_node.properties["last_seen"] = datetime.now().isoformat()
            self.stats.nodes_updated += 1
        else:
            node = Node(
                id=node_id,
                type=entity_type,
                name=entity_text,
                filepath=source_id,
            )
            node.properties = {
                "confidence": entity_data.get("confidence", 1.0),
                "created_at": datetime.now().isoformat(),
                "normalized_form": entity_data.get("normalized_form"),
            }
            node.properties.update(entity_data.get("properties", {}))
            self.graph.add_node(node)
            self.stats.nodes_created += 1

    def _get_or_create_node_id(self, entity_text: str, entity_type: NodeType) -> int:
        """Get existing node ID or create new one."""
        key = f"{entity_type.value}:{entity_text.lower()}"
        if key in self._entity_node_map:
            return self._entity_node_map[key]

        self._node_id_counter += 1
        self._entity_node_map[key] = self._node_id_counter
        return self._node_id_counter

    def _map_entity_type(self, entity_type_str: str) -> NodeType:
        """Map entity type string to NodeType enum."""
        type_mapping = {
            "technology": NodeType.CLASS,
            "material": NodeType.CLASS,
            "company": NodeType.CLASS,
            "organization": NodeType.CLASS,
            "process": NodeType.FUNCTION,
            "methodology": NodeType.FUNCTION,
            "research_institution": NodeType.CLASS,
            "equipment": NodeType.CLASS,
            "machinery": NodeType.CLASS,
            "product": NodeType.CLASS,
            "standard": NodeType.CLASS,
            "person": NodeType.CLASS,
            "location": NodeType.CLASS,
            "concept": NodeType.CLASS,
        }
        return type_mapping.get(entity_type_str.lower(), NodeType.CLASS)

    def populate_from_relationships(
        self,
        relationships: list[dict[str, Any]],
    ) -> GraphPopulationStats:
        """
        Populate graph with relationships from semantic analysis.

        Args:
            relationships: List of relationship dictionaries

        Returns:
            GraphPopulationStats with operation statistics
        """
        self.stats = GraphPopulationStats()
        self.stats.start_time = datetime.now()

        for rel_data in relationships:
            try:
                self._add_relationship_to_graph(rel_data)
            except Exception as e:
                self.stats.errors.append(f"Failed to add relationship: {e}")

        self.stats.end_time = datetime.now()
        return self.stats

    def _add_relationship_to_graph(self, rel_data: dict[str, Any]) -> None:
        """Add a single relationship to the graph."""
        source_text = rel_data.get("source", "")
        target_text = rel_data.get("target", "")
        rel_type_str = rel_data.get("type", "unknown")

        source_node_id = self._get_node_id_by_text(source_text)
        target_node_id = self._get_node_id_by_text(target_text)

        if source_node_id is None or target_node_id is None:
            logger.warning(
                f"Skipping relationship: missing node for {source_text} -> {target_text}"
            )
            return

        edge_type = self._map_relationship_type(rel_type_str)

        existing_edge = self.graph.get_edge(source_node_id, target_node_id, edge_type)
        if existing_edge:
            existing_edge.properties["last_seen"] = datetime.now().isoformat()
            self.stats.edges_updated += 1
        else:
            self.graph.add_edge(
                source_id=source_node_id,
                target_id=target_node_id,
                edge_type=edge_type,
                properties={
                    "confidence": rel_data.get("confidence", 1.0),
                    "context": rel_data.get("context"),
                },
            )
            self.stats.edges_created += 1

    def _get_node_id_by_text(self, text: str) -> int | None:
        """Get node ID by entity text."""
        for node_id, node in self.graph.node_map.items():
            if node.name.lower() == text.lower():
                return node_id
        return None

    def _map_relationship_type(self, rel_type_str: str) -> EdgeType:
        """Map relationship type string to EdgeType enum."""
        type_mapping = {
            "develops": EdgeType.CALLS,
            "uses": EdgeType.USES,
            "produces": EdgeType.CALLS,
            "applies": EdgeType.USES,
            "researches": EdgeType.REFERENCES,
            "partners": EdgeType.REFERENCES,
            "competes": EdgeType.DEPENDS_ON,
            "supplies": EdgeType.USES,
            "implements": EdgeType.CALLS,
            "improves": EdgeType.REFERENCES,
            "replaces": EdgeType.DEPENDS_ON,
            "depends_on": EdgeType.DEPENDS_ON,
            "references": EdgeType.REFERENCES,
            "contains": EdgeType.CONTAINS,
        }
        return type_mapping.get(rel_type_str.lower(), EdgeType.REFERENCES)

    def populate_from_patterns(
        self,
        patterns: list[dict[str, Any]],
    ) -> GraphPopulationStats:
        """
        Populate graph with detected patterns.

        Args:
            patterns: List of pattern dictionaries

        Returns:
            GraphPopulationStats with operation statistics
        """
        self.stats = GraphPopulationStats()
        self.stats.start_time = datetime.now()

        pattern_type_map = {
            "creational": NodeType.CLASS,
            "structural": NodeType.CLASS,
            "behavioral": NodeType.FUNCTION,
            "architectural": NodeType.MODULE,
            "domain_specific": NodeType.CLASS,
        }

        for pattern_data in patterns:
            try:
                pattern_name = pattern_data.get("name", "Unknown")
                pattern_type = pattern_data.get("type", "domain_specific")
                file_path = pattern_data.get("file_path", "")
                confidence = pattern_data.get("confidence", 1.0)

                node_id = self._get_or_create_node_id(f"Pattern: {pattern_name}", NodeType.CLASS)

                existing_node = self.graph.get_node(node_id)
                if existing_node:
                    self.stats.nodes_updated += 1
                else:
                    node = Node(
                        id=node_id,
                        type=pattern_type_map.get(pattern_type, NodeType.CLASS),
                        name=f"Pattern: {pattern_name}",
                        filepath=file_path,
                    )
                    node.properties = {
                        "pattern_type": pattern_type,
                        "description": pattern_data.get("description", ""),
                        "confidence": confidence,
                        "start_line": pattern_data.get("start_line"),
                        "end_line": pattern_data.get("end_line"),
                        "created_at": datetime.now().isoformat(),
                    }
                    self.graph.add_node(node)
                    self.stats.nodes_created += 1

                if file_path:
                    file_node_id = self._get_or_create_node_id(
                        f"File: {file_path}", NodeType.MODULE
                    )
                    self.graph.add_edge(
                        source_id=file_node_id,
                        target_id=node_id,
                        edge_type=EdgeType.CONTAINS,
                    )
                    self.stats.edges_created += 1

            except Exception as e:
                self.stats.errors.append(f"Failed to add pattern: {e}")

        self.stats.end_time = datetime.now()
        return self.stats

    def populate_from_anti_patterns(
        self,
        anti_patterns: list[dict[str, Any]],
    ) -> GraphPopulationStats:
        """
        Populate graph with detected anti-patterns.

        Args:
            anti_patterns: List of anti-pattern dictionaries

        Returns:
            GraphPopulationStats with operation statistics
        """
        self.stats = GraphPopulationStats()
        self.stats.start_time = datetime.now()

        severity_map = {
            "critical": NodeType.ISSUE,
            "high": NodeType.ISSUE,
            "medium": NodeType.ISSUE,
            "low": NodeType.RECOMMENDATION,
            "info": NodeType.RECOMMENDATION,
        }

        severity_edge_map = {
            "critical": EdgeType.DEPENDS_ON,
            "high": EdgeType.DEPENDS_ON,
            "medium": EdgeType.USES,
            "low": EdgeType.USES,
            "info": EdgeType.REFERENCES,
        }

        for ap_data in anti_patterns:
            try:
                ap_name = ap_data.get("name", "Unknown Anti-Pattern")
                severity = ap_data.get("severity", "medium")
                file_path = ap_data.get("file_path", "")
                confidence = ap_data.get("confidence", 1.0)

                node_id = self._get_or_create_node_id(f"AntiPattern: {ap_name}", NodeType.ISSUE)

                existing_node = self.graph.get_node(node_id)
                if existing_node:
                    self.stats.nodes_updated += 1
                else:
                    node = Node(
                        id=node_id,
                        type=severity_map.get(severity, NodeType.ISSUE),
                        name=f"AntiPattern: {ap_name}",
                        filepath=file_path,
                    )
                    node.properties = {
                        "severity": severity,
                        "description": ap_data.get("description", ""),
                        "impact": ap_data.get("impact", ""),
                        "remediation": ap_data.get("remediation", ""),
                        "confidence": confidence,
                        "start_line": ap_data.get("start_line"),
                        "end_line": ap_data.get("end_line"),
                        "created_at": datetime.now().isoformat(),
                    }
                    self.graph.add_node(node)
                    self.stats.nodes_created += 1

                if file_path:
                    file_node_id = self._get_or_create_node_id(
                        f"File: {file_path}", NodeType.MODULE
                    )
                    self.graph.add_edge(
                        source_id=node_id,
                        target_id=file_node_id,
                        edge_type=severity_edge_map.get(severity, EdgeType.USES),
                    )
                    self.stats.edges_created += 1

            except Exception as e:
                self.stats.errors.append(f"Failed to add anti-pattern: {e}")

        self.stats.end_time = datetime.now()
        return self.stats

    def populate_full_analysis(
        self,
        analysis_result: dict[str, Any],
        source_id: str | None = None,
    ) -> GraphPopulationStats:
        """
        Populate graph from complete analysis result.

        Args:
            analysis_result: Complete analysis result dictionary
            source_id: Optional source identifier

        Returns:
            GraphPopulationStats with operation statistics
        """
        self.stats = GraphPopulationStats()
        self.stats.start_time = datetime.now()

        entities = analysis_result.get("entities", [])
        relationships = analysis_result.get("relationships", [])
        patterns = analysis_result.get("patterns", [])
        anti_patterns = analysis_result.get("anti_patterns", [])

        self.populate_from_entities(entities, source_id)
        self.populate_from_relationships(relationships)
        self.populate_from_patterns(patterns)
        self.populate_from_anti_patterns(anti_patterns)

        self.stats.end_time = datetime.now()
        return self.stats

    def incremental_update(
        self,
        new_entities: list[dict[str, Any]],
        new_relationships: list[dict[str, Any]],
    ) -> GraphPopulationStats:
        """
        Perform incremental update of the graph.

        Args:
            new_entities: New entities to add
            new_relationships: New relationships to add

        Returns:
            GraphPopulationStats with operation statistics
        """
        self.stats = GraphPopulationStats()
        self.stats.start_time = datetime.now()

        existing_entity_texts = self._get_all_entity_texts()
        existing_relationships = self._get_all_relationships()

        for entity in new_entities:
            entity_text = entity.get("text", "")
            if entity_text.lower() not in existing_entity_texts:
                self._add_entity_to_graph(entity, None)
                existing_entity_texts.add(entity_text.lower())

        for rel in new_relationships:
            rel_key = (
                rel.get("source", "").lower(),
                rel.get("target", "").lower(),
                rel.get("type", "").lower(),
            )
            if rel_key not in existing_relationships:
                self._add_relationship_to_graph(rel)

        self.stats.end_time = datetime.now()
        return self.stats

    def _get_all_entity_texts(self) -> set[str]:
        """Get set of all entity texts in the graph."""
        return {node.name.lower() for node in self.graph.node_map.values()}

    def _get_all_relationships(self) -> set[tuple[str, str, str]]:
        """Get set of all relationship keys in the graph."""
        relationships = set()
        for edge in self.graph.edge_map.values():
            source = self.graph.get_node(edge.source)
            target = self.graph.get_node(edge.target)
            if source and target:
                relationships.add(
                    (
                        source.name.lower(),
                        target.name.lower(),
                        edge.type.value.lower(),
                    )
                )
        return relationships

    def detect_conflicts(self) -> list[dict[str, Any]]:
        """
        Detect potential conflicts in the graph.

        Returns:
            List of detected conflicts with details
        """
        conflicts = []

        duplicate_nodes: dict[str, list[int]] = {}
        for node_id, node in self.graph.node_map.items():
            key = f"{node.type.value}:{node.name.lower()}"
            if key not in duplicate_nodes:
                duplicate_nodes[key] = []
            duplicate_nodes[key].append(node_id)

        for key, node_ids in duplicate_nodes.items():
            if len(node_ids) > 1:
                conflicts.append(
                    {
                        "type": "duplicate_nodes",
                        "key": key,
                        "node_ids": node_ids,
                        "message": f"Found {len(node_ids)} nodes with same type and name",
                    }
                )

        orphaned_edges = []
        for edge_key, edge in self.graph.edge_map.items():
            if edge.source not in self.graph.node_map:
                orphaned_edges.append(("source", edge_key))
            if edge.target not in self.graph.node_map:
                orphaned_edges.append(("target", edge_key))

        if orphaned_edges:
            conflicts.append(
                {
                    "type": "orphaned_edges",
                    "edges": orphaned_edges,
                    "message": f"Found {len(orphaned_edges)} edges with missing nodes",
                }
            )

        return conflicts

    def resolve_conflict(self, conflict: dict[str, Any]) -> bool:
        """
        Resolve a detected conflict.

        Args:
            conflict: Conflict dictionary to resolve

        Returns:
            True if resolution was successful
        """
        conflict_type = conflict.get("type")

        if conflict_type == "duplicate_nodes":
            node_ids = conflict.get("node_ids", [])
            if len(node_ids) > 1:
                primary = node_ids[0]
                for node_id in node_ids[1:]:
                    self._merge_nodes(primary, node_id)
                self.stats.conflicts_resolved += 1
                return True

        return False

    def _merge_nodes(self, primary_id: int, secondary_id: int) -> None:
        """Merge secondary node into primary node."""
        primary_node = self.graph.get_node(primary_id)
        secondary_node = self.graph.get_node(secondary_id)

        if not primary_node or not secondary_node:
            return

        for prop_name, prop_value in secondary_node.properties.items():
            if prop_name not in primary_node.properties:
                primary_node.properties[prop_name] = prop_value

        for edge_key, edge in list(self.graph.edge_map.items()):
            if edge.source == secondary_id:
                edge.source = primary_id
            if edge.target == secondary_id:
                edge.target = primary_id

        self.graph.remove_node(secondary_id)

    def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the current graph state."""
        node_type_counts: dict[str, int] = {}
        for node in self.graph.node_map.values():
            type_key = node.type.value
            node_type_counts[type_key] = node_type_counts.get(type_key, 0) + 1

        edge_type_counts: dict[str, int] = {}
        for edge in self.graph.edge_map.values():
            type_key = edge.type.value
            edge_type_counts[type_key] = edge_type_counts.get(type_key, 0) + 1

        return {
            "total_nodes": self.graph.node_count,
            "total_edges": self.graph.edge_count,
            "node_types": node_type_counts,
            "edge_types": edge_type_counts,
        }
