"""
Integration test for Phase 1 Foundation components.
"""

import sys
import os
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data.graph import HybridGraph, Node
from src.data.nodes import NodeType
from src.data.edges import EdgeType
from src.rag.database import RAGDatabase
from src.orchestration.coordinator import AgentOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_hybrid_graph():
    """Test the HybridGraph implementation."""
    logger.info("Testing HybridGraph implementation...")

    # Create a graph
    graph = HybridGraph()

    # Add nodes
    node1 = Node(
        id=1, type=NodeType.MODULE, name="test_module.py", filepath="/path/to/test_module.py"
    )
    node2 = Node(id=2, type=NodeType.CLASS, name="TestClass", filepath="/path/to/test_module.py")
    node3 = Node(
        id=3, type=NodeType.FUNCTION, name="test_function", filepath="/path/to/test_module.py"
    )

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    # Add edges
    graph.add_edge(1, 2, EdgeType.CONTAINS)
    graph.add_edge(2, 3, EdgeType.CONTAINS)

    # Test queries
    module_nodes = graph.get_nodes_by_type(NodeType.MODULE)
    assert len(module_nodes) == 1
    assert module_nodes[0].name == "test_module.py"

    contains_edges = graph.get_edges_by_type(EdgeType.CONTAINS)
    assert len(contains_edges) == 2

    logger.info("HybridGraph test passed!")


def test_rag_database():
    """Test the RAG database integration."""
    logger.info("Testing RAG database integration...")

    # Create a mock database connection
    db_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}

    rag_db = RAGDatabase(db_config)

    # Test connection (will fail but shouldn't crash)
    connected = rag_db.connect()
    logger.info(f"Database connection test: {'PASSED' if connected else 'FAILED (expected)'}")

    # Test schema info
    schema = rag_db.get_schema_info()
    assert "labels" in schema
    assert "relationship_types" in schema

    logger.info("RAG database test passed!")


def main():
    """Run all integration tests."""
    logger.info("Running Phase 1 Foundation integration tests...")

    try:
        test_hybrid_graph()
        test_rag_database()
        logger.info("All integration tests passed!")
        return True
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
