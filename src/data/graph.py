"""
Enhanced Graph Data Structures for Project Analysis Tool

This module implements the HybridGraph data structure and related components
as specified in the architectural plan.
"""

import numpy as np
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Set, Optional, Any
import logging

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Enumeration of node types for the knowledge graph."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    PACKAGE = "package"
    PATTERN = "pattern"
    METRIC = "metric"
    ISSUE = "issue"
    RECOMMENDATION = "recommendation"


class EdgeType(Enum):
    """Enumeration of edge types for relationships in the knowledge graph."""
    CALLS = "calls"
    INHERITS = "inherits"
    IMPORTS = "imports"
    CONTAINS = "contains"
    USES = "uses"
    DEPENDS_ON = "depends_on"
    INSTANTIATES = "instantiates"
    REFERENCES = "references"
    IMPLEMENTS = "implements"
    PATTERN_OF = "pattern_of"
    SUGGESTS = "suggests"


class Node:
    """Represents a node in the knowledge graph."""
    
    def __init__(self, id: int, type: NodeType, name: str, filepath: Optional[str] = None):
        self.id = id
        self.type = type
        self.name = name
        self.filepath = filepath
        self.properties = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.type.value,
            'name': self.name,
            'filepath': self.filepath,
            'properties': self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create node from dictionary."""
        node = cls(
            id=data['id'],
            type=NodeType(data['type']),
            name=data['name'],
            filepath=data.get('filepath')
        )
        node.properties = data.get('properties', {})
        return node


class Edge:
    """Represents an edge in the knowledge graph."""
    
    def __init__(self, source: int, target: int, type: EdgeType):
        self.source = source
        self.target = target
        self.type = type
        self.properties = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization."""
        return {
            'source': self.source,
            'target': self.target,
            'type': self.type.value,
            'properties': self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Edge':
        """Create edge from dictionary."""
        edge = cls(
            source=data['source'],
            target=data['target'],
            type=EdgeType(data['type'])
        )
        edge.properties = data.get('properties', {})
        return edge


class RelationshipIndex:
    """Multi-dimensional index for fast relationship queries."""
    
    def __init__(self):
        self.by_source = defaultdict(list)
        self.by_target = defaultdict(list)
        self.by_type = defaultdict(list)
        self.by_property = defaultdict(lambda: defaultdict(list))


class PropertyIndex:
    """Index for efficient property-based filtering."""
    
    def __init__(self):
        self.indexes = defaultdict(lambda: defaultdict(set))
    
    def add_property(self, entity_id: int, prop_name: str, prop_value):
        """Add a property to the index."""
        self.indexes[prop_name][prop_value].add(entity_id)
    
    def query(self, prop_name: str, prop_value) -> Set[int]:
        """Query entities by property."""
        return self.indexes[prop_name][prop_value].copy()


class HybridGraph:
    """Memory-efficient graph representation combining multiple data structures."""
    
    def __init__(self):
        # Adjacency list for general traversal
        self.adj_list = defaultdict(set)
        
        # CSR format for matrix operations (to be computed when needed)
        self.csr_data = None
        
        # Hash maps for O(1) lookups
        self.node_map = {}  # node_id -> Node
        self.edge_map = {}  # (source, target, type) -> Edge
        
        # Indexes for fast querying
        self.nodes_by_type = defaultdict(set)  # NodeType -> Set[node_id]
        self.edges_by_type = defaultdict(set)  # EdgeType -> Set[(source, target)]
        self.property_index = PropertyIndex()
        self.relationship_index = RelationshipIndex()
        
        # Metadata
        self.node_count = 0
        self.edge_count = 0
        
        # Dirty flags for caching
        self.dirty_flags = set()
    
    def add_node(self, node: Node):
        """Add a node to the graph."""
        self.node_map[node.id] = node
        self.nodes_by_type[node.type].add(node.id)
        self.node_count += 1
        
        # Index node properties
        for prop_name, prop_value in node.properties.items():
            self.property_index.add_property(node.id, prop_name, prop_value)
    
    def add_edge(self, source_id: int, target_id: int, edge_type: EdgeType, 
                 properties: Optional[Dict[str, Any]] = None):
        """Add an edge to the graph."""
        # Verify nodes exist
        if source_id not in self.node_map or target_id not in self.node_map:
            raise ValueError("Source or target node does not exist")
        
        # Add to adjacency list
        self.adj_list[source_id].add(target_id)
        
        # Store edge information
        edge_key = (source_id, target_id, edge_type)
        edge = Edge(source_id, target_id, edge_type)
        if properties:
            edge.properties.update(properties)
        
        self.edge_map[edge_key] = edge
        self.edges_by_type[edge_type].add((source_id, target_id))
        
        # Update indexes
        self.relationship_index.by_source[source_id].append(edge)
        self.relationship_index.by_target[target_id].append(edge)
        self.relationship_index.by_type[edge_type].append(edge)
        
        # Index edge properties
        for prop_name, prop_value in edge.properties.items():
            self.property_index.add_property(id(edge), prop_name, prop_value)
        
        self.edge_count += 1
    
    def get_neighbors(self, node_id: int) -> Set[int]:
        """Get all neighbors of a node."""
        return self.adj_list.get(node_id, set())
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Get all nodes of a specific type."""
        node_ids = self.nodes_by_type.get(node_type, set())
        return [self.node_map[nid] for nid in node_ids]
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """Get all edges of a specific type."""
        edge_keys = self.edges_by_type.get(edge_type, set())
        return [self.edge_map[(src, tgt, edge_type)] for src, tgt in edge_keys]
    
    def get_node(self, node_id: int) -> Optional[Node]:
        """Get a node by ID."""
        return self.node_map.get(node_id)
    
    def get_edge(self, source_id: int, target_id: int, edge_type: EdgeType) -> Optional[Edge]:
        """Get an edge by source, target, and type."""
        edge_key = (source_id, target_id, edge_type)
        return self.edge_map.get(edge_key)
    
    def query_nodes_by_property(self, prop_name: str, prop_value) -> List[Node]:
        """Query nodes by property."""
        node_ids = self.property_index.query(prop_name, prop_value)
        return [self.node_map[nid] for nid in node_ids if nid in self.node_map]
    
    def remove_node(self, node_id: int):
        """Remove a node and all its edges."""
        if node_id not in self.node_map:
            return
        
        # Remove node
        node = self.node_map.pop(node_id)
        self.nodes_by_type[node.type].discard(node_id)
        self.node_count -= 1
        
        # Remove associated edges
        # Remove outgoing edges
        if node_id in self.adj_list:
            targets = list(self.adj_list[node_id])
            for target in targets:
                # Remove from adjacency list
                self.adj_list[node_id].discard(target)
                
                # Remove from edge map
                # Need to find all edge types between these nodes
                for edge_type in EdgeType:
                    edge_key = (node_id, target, edge_type)
                    if edge_key in self.edge_map:
                        del self.edge_map[edge_key]
                        self.edges_by_type[edge_type].discard((node_id, target))
                        self.edge_count -= 1
            
            # Remove adjacency list entry
            del self.adj_list[node_id]
        
        # Remove incoming edges
        for source, targets in self.adj_list.items():
            if node_id in targets:
                targets.discard(node_id)
                # Remove from edge map (find all edge types)
                for edge_type in EdgeType:
                    edge_key = (source, node_id, edge_type)
                    if edge_key in self.edge_map:
                        del self.edge_map[edge_key]
                        self.edges_by_type[edge_type].discard((source, node_id))
                        self.edge_count -= 1
    
    def to_dict(self) -> Dict:
        """Convert graph to dictionary representation for serialization."""
        return {
            'nodes': {nid: node.to_dict() for nid, node in self.node_map.items()},
            'edges': {str(key): edge.to_dict() for key, edge in self.edge_map.items()},
            'adj_list': {str(k): list(v) for k, v in self.adj_list.items()},
            'stats': {
                'node_count': self.node_count,
                'edge_count': self.edge_count
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HybridGraph':
        """Create graph from dictionary representation."""
        graph = cls()
        
        # Restore nodes
        for nid, node_data in data.get('nodes', {}).items():
            node = Node.from_dict(node_data)
            graph.node_map[node.id] = node
            graph.nodes_by_type[node.type].add(node.id)
        
        # Restore edges
        for edge_key_str, edge_data in data.get('edges', {}).items():
            edge = Edge.from_dict(edge_data)
            edge_key = (edge.source, edge.target, edge.type)
            graph.edge_map[edge_key] = edge
            graph.edges_by_type[edge.type].add((edge.source, edge.target))
        
        # Restore adjacency list
        for node_id_str, neighbors in data.get('adj_list', {}).items():
            node_id = int(node_id_str)
            graph.adj_list[node_id] = set(neighbors)
        
        # Restore stats
        stats = data.get('stats', {})
        graph.node_count = stats.get('node_count', 0)
        graph.edge_count = stats.get('edge_count', 0)
        
        return graph