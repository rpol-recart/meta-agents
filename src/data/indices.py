"""
Indexing system for efficient querying in the Project Analysis Tool.
"""

from collections import defaultdict
from typing import Dict, List, Set, Any


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