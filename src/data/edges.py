"""
Edge definitions for the Project Analysis Tool knowledge graph.
"""

from enum import Enum


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