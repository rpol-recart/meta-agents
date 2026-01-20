"""
Neo4j Tools - Graph database operations for Neo4j.

This module provides tools for interacting with Neo4j graph databases,
including executing Cypher queries, managing nodes and relationships,
and exploring graph structure.

Required environment variables:
    NEO4J_URI         - Neo4j database URI (default: bolt://localhost:7687)
    NEO4J_USER        - Neo4j username (default: neo4j)
    NEO4J_PASSWORD    - Neo4j password
    NEO4J_DATABASE    - Neo4j database name (default: neo4j)

Example usage:
    from src.tools.neo4j_tools import get_neo4j_tools

    # Register tools
    registry = get_neo4j_tools()

    # Or use directly
    from src.tools.neo4j_tools import execute_cypher_query

    result = execute_cypher_query(query="MATCH (n) RETURN n LIMIT 5")
"""

import logging
import os
from typing import Any

from .registry import ToolRegistry, get_default_tool_registry

logger = logging.getLogger(__name__)


def get_neo4j_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """
    Get Neo4j-related tools.

    Args:
        registry: Optional registry to add tools to

    Returns:
        ToolRegistry with Neo4j tools
    """
    if registry is None:
        registry = get_default_tool_registry()

    def execute_cypher_query(
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> str:
        """
        Execute a Cypher query against the Neo4j database.

        Args:
            query: Cypher query to execute
            parameters: Optional query parameters as dictionary
            database: Database name (uses NEO4J_DATABASE env var if not provided)

        Returns:
            Query results as formatted string
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j package is required for Neo4j operations. Install it with: pip install neo4j"
            )

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")

        if not password:
            return "Error: Neo4j password not configured. Set NEO4J_PASSWORD environment variable."

        db_name = database or os.environ.get("NEO4J_DATABASE", "neo4j")

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()

            with driver.session(database=db_name) as session:
                result = session.run(query, parameters or {})
                records = list(result)

                if not records:
                    return "Query executed successfully. No results returned."

                output = "Neo4j Query Results\n"
                output += "-" * 50 + "\n\n"
                output += f"Query: {query}\n"
                output += f"Records: {len(records)}\n\n"

                for i, record in enumerate(records, 1):
                    output += f"Record {i}:\n"
                    for key, value in record.items():
                        output += f"  {key}: {value}\n"
                    output += "\n"

                logger.info(f"Executed Cypher query, returned {len(records)} records")
                return output

        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return f"Error: {str(e)}"

    def neo4j_database_info() -> str:
        """
        Get information about the connected Neo4j database.

        Returns:
            Database information as formatted string
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j package is required for Neo4j operations. Install it with: pip install neo4j"
            )

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        db_name = os.environ.get("NEO4J_DATABASE", "neo4j")

        if not password:
            return "Error: Neo4j password not configured. Set NEO4J_PASSWORD environment variable."

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()

            with driver.session(database=db_name) as session:
                info_result = session.run("CALL db.info()").single()
                size_result = session.run(
                    "CALL dbms.queryJmx('org.neo4j:name=Neo4j,type=Database') "
                    "YIELD attributes "
                    "RETURN attributes['HeapMemoryUsage'] as heap, "
                    "attributes['NonHeapMemoryUsage'] as non_heap"
                ).single()

                output = "Neo4j Database Information\n"
                output += "-" * 50 + "\n\n"

                output += f"Connection URI: {uri}\n"
                output += f"Database: {db_name}\n"
                output += f"User: {user}\n\n"

                if info_result:
                    output += "Database Info:\n"
                    for key, value in info_result.items():
                        output += f"  {key}: {value}\n"
                    output += "\n"

                if size_result:
                    output += "Memory Usage:\n"
                    heap = size_result.get("heap", {})
                    non_heap = size_result.get("non_heap", {})
                    if isinstance(heap, dict):
                        output += f"  Heap Max: {heap.get('max', 'N/A')}\n"
                        output += f"  Heap Used: {heap.get('used', 'N/A')}\n"
                    if isinstance(non_heap, dict):
                        output += f"  Non-Heap Max: {non_heap.get('max', 'N/A')}\n"
                        output += f"  Non-Heap Used: {non_heap.get('used', 'N/A')}\n"

                logger.info("Retrieved Neo4j database info")
                return output

        except Exception as e:
            logger.error(f"Failed to get Neo4j database info: {e}")
            return f"Error: {str(e)}"

    def neo4j_list_labels() -> str:
        """
        List all labels in the Neo4j database.

        Returns:
            List of labels as formatted string
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j package is required for Neo4j operations. Install it with: pip install neo4j"
            )

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        db_name = os.environ.get("NEO4J_DATABASE", "neo4j")

        if not password:
            return "Error: Neo4j password not configured. Set NEO4J_PASSWORD environment variable."

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()

            with driver.session(database=db_name) as session:
                result = session.run("CALL db.labels()")
                labels = [record[0] for record in result]

                output = "Neo4j Database Labels\n"
                output += "-" * 50 + "\n\n"
                output += f"Total Labels: {len(labels)}\n\n"

                if labels:
                    for label in sorted(labels):
                        count_result = session.run(
                            f"MATCH (n:`{label}`) RETURN count(n) as count"
                        ).single()
                        count = count_result["count"] if count_result else 0
                        output += f"  :{label} - {count} nodes\n"
                else:
                    output += "  No labels found in database.\n"

                logger.info(f"Listed {len(labels)} Neo4j labels")
                return output

        except Exception as e:
            logger.error(f"Failed to list Neo4j labels: {e}")
            return f"Error: {str(e)}"

    def neo4j_list_relationship_types() -> str:
        """
        List all relationship types in the Neo4j database.

        Returns:
            List of relationship types as formatted string
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j package is required for Neo4j operations. Install it with: pip install neo4j"
            )

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        db_name = os.environ.get("NEO4J_DATABASE", "neo4j")

        if not password:
            return "Error: Neo4j password not configured. Set NEO4J_PASSWORD environment variable."

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()

            with driver.session(database=db_name) as session:
                result = session.run("CALL db.relationshipTypes()")
                rel_types = [record[0] for record in result]

                output = "Neo4j Relationship Types\n"
                output += "-" * 50 + "\n\n"
                output += f"Total Types: {len(rel_types)}\n\n"

                if rel_types:
                    for rel_type in sorted(rel_types):
                        count_result = session.run(
                            f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count"
                        ).single()
                        count = count_result["count"] if count_result else 0
                        output += f"  :{rel_type} - {count} relationships\n"
                else:
                    output += "  No relationship types found in database.\n"

                logger.info(f"Listed {len(rel_types)} Neo4j relationship types")
                return output

        except Exception as e:
            logger.error(f"Failed to list Neo4j relationship types: {e}")
            return f"Error: {str(e)}"

    def neo4j_get_node_count() -> str:
        """
        Get the total count of nodes in the database.

        Returns:
            Node count as formatted string
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j package is required for Neo4j operations. Install it with: pip install neo4j"
            )

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        db_name = os.environ.get("NEO4J_DATABASE", "neo4j")

        if not password:
            return "Error: Neo4j password not configured. Set NEO4J_PASSWORD environment variable."

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()

            with driver.session(database=db_name) as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]

                output = "Neo4j Node Count\n"
                output += "-" * 50 + "\n\n"
                output += f"Total Nodes: {count}\n"

                logger.info(f"Got node count: {count}")
                return output

        except Exception as e:
            logger.error(f"Failed to get Neo4j node count: {e}")
            return f"Error: {str(e)}"

    def neo4j_get_relationship_count() -> str:
        """
        Get the total count of relationships in the database.

        Returns:
            Relationship count as formatted string
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j package is required for Neo4j operations. Install it with: pip install neo4j"
            )

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        db_name = os.environ.get("NEO4J_DATABASE", "neo4j")

        if not password:
            return "Error: Neo4j password not configured. Set NEO4J_PASSWORD environment variable."

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()

            with driver.session(database=db_name) as session:
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                count = result.single()["count"]

                output = "Neo4j Relationship Count\n"
                output += "-" * 50 + "\n\n"
                output += f"Total Relationships: {count}\n"

                logger.info(f"Got relationship count: {count}")
                return output

        except Exception as e:
            logger.error(f"Failed to get Neo4j relationship count: {e}")
            return f"Error: {str(e)}"

    def neo4j_schema() -> str:
        """
        Get the database schema (indexes and constraints).

        Returns:
            Schema information as formatted string
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j package is required for Neo4j operations. Install it with: pip install neo4j"
            )

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        db_name = os.environ.get("NEO4J_DATABASE", "neo4j")

        if not password:
            return "Error: Neo4j password not configured. Set NEO4J_PASSWORD environment variable."

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()

            with driver.session(database=db_name) as session:
                indexes = session.run("CALL db.indexes()").data()
                constraints = session.run("CALL db.constraints()").data()

                output = "Neo4j Database Schema\n"
                output += "-" * 50 + "\n\n"

                output += "Indexes:\n"
                if indexes:
                    for idx in indexes:
                        output += f"  {idx}\n"
                else:
                    output += "  No indexes found.\n"
                output += "\n"

                output += "Constraints:\n"
                if constraints:
                    for constr in constraints:
                        output += f"  {constr}\n"
                else:
                    output += "  No constraints found.\n"

                logger.info("Retrieved Neo4j schema")
                return output

        except Exception as e:
            logger.error(f"Failed to get Neo4j schema: {e}")
            return f"Error: {str(e)}"

    def neo4j_config_status() -> str:
        """
        Check Neo4j configuration and connectivity status.

        Returns:
            Configuration status and database info
        """
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        db_name = os.environ.get("NEO4J_DATABASE", "neo4j")

        result = "Neo4j Configuration Status\n"
        result += "-" * 40 + "\n\n"

        result += "Environment Variables:\n"
        result += f"  NEO4J_URI:      {uri}\n"
        result += f"  NEO4J_USER:     {user}\n"
        result += (
            f"  NEO4J_PASSWORD: {'*' * len(password) if password else 'NOT SET'}\n"
        )
        result += f"  NEO4J_DATABASE: {db_name}\n\n"

        if not password:
            result += "Status: INCOMPLETE - Missing NEO4J_PASSWORD\n"
            result += "\nTo configure, set these environment variables:\n"
            result += "  export NEO4J_URI=bolt://localhost:7687\n"
            result += "  export NEO4J_USER=neo4j\n"
            result += "  export NEO4J_PASSWORD=your_password\n"
            result += "  export NEO4J_DATABASE=neo4j\n"
            return result

        result += "Status: CONFIGURATION COMPLETE\n\n"

        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()

            result += "Connection: SUCCESS\n"

            with driver.session(database=db_name) as session:
                node_result = session.run("MATCH (n) RETURN count(n) as count").single()
                rel_result = session.run(
                    "MATCH ()-[r]->() RETURN count(r) as count"
                ).single()

                result += f"Database: {db_name}\n"
                result += f"Nodes: {node_result['count']}\n"
                result += f"Relationships: {rel_result['count']}\n"

        except ImportError:
            result += "Connection: SKIPPED (neo4j package not installed)\n"
            result += "\nInstall with: pip install neo4j\n"
        except Exception as e:
            result += f"Connection: FAILED\nError: {str(e)}\n"

        return result

    registry.register(
        name="execute_cypher_query",
        func=execute_cypher_query,
        description="Execute a Cypher query against the Neo4j database",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Cypher query to execute",
                },
                "parameters": {
                    "type": "object",
                    "description": "Optional query parameters as dictionary",
                },
                "database": {
                    "type": "string",
                    "description": "Database name (uses NEO4J_DATABASE env var if not provided)",
                },
            },
            "required": ["query"],
        },
    )

    registry.register(
        name="neo4j_database_info",
        func=neo4j_database_info,
        description="Get information about the connected Neo4j database",
    )

    registry.register(
        name="neo4j_list_labels",
        func=neo4j_list_labels,
        description="List all labels in the Neo4j database with node counts",
    )

    registry.register(
        name="neo4j_list_relationship_types",
        func=neo4j_list_relationship_types,
        description="List all relationship types in the Neo4j database",
    )

    registry.register(
        name="neo4j_get_node_count",
        func=neo4j_get_node_count,
        description="Get the total count of nodes in the database",
    )

    registry.register(
        name="neo4j_get_relationship_count",
        func=neo4j_get_relationship_count,
        description="Get the total count of relationships in the database",
    )

    registry.register(
        name="neo4j_schema",
        func=neo4j_schema,
        description="Get the database schema (indexes and constraints)",
    )

    registry.register(
        name="neo4j_config_status",
        func=neo4j_config_status,
        description="Check Neo4j configuration and connectivity status",
    )

    return registry


def get_default_neo4j_tools() -> ToolRegistry:
    """Get all default tools including Neo4j."""
    registry = get_default_tool_registry()
    registry = get_neo4j_tools(registry)
    return registry
