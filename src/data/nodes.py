"""
Node definitions for the Project Analysis Tool knowledge graph.
"""

from enum import Enum
from typing import Dict, Any, Optional


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