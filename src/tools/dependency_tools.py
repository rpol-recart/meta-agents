"""
Dependency Analysis Tools for Agents

This module provides tools for analyzing code dependencies,
imports, and relationships between modules.
"""

import json
import logging
import re

from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def get_dependency_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Register all dependency analysis tools for agents."""
    if registry is None:
        registry = ToolRegistry()

    def find_imports(code: str) -> str:
        """Extract all imports from source code."""
        try:
            imports = {
                "standard_lib": [],
                "third_party": [],
                "local": [],
                "relative": [],
            }

            import_patterns = {
                "standard_lib": [
                    r"^import\s+(\w+)",
                    r"^from\s+(\w+)\s+import",
                ],
                "third_party": r"(?:from\s+|import\s+)([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)",
                "local": r"(?:from\s+|import\s+)\.([a-z0-9_]*)",
                "relative": r"(?:from\s+|import\s+)\.\.([a-z0-9_]*)",
            }

            lines = code.split("\n")
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                for lib in [
                    "os",
                    "sys",
                    "json",
                    "re",
                    "logging",
                    "datetime",
                    "collections",
                    "itertools",
                ]:
                    if re.match(rf"^import\s+{lib}\b", stripped) or re.match(
                        rf"^from\s+{lib}\b", stripped
                    ):
                        if lib not in imports["standard_lib"]:
                            imports["standard_lib"].append(lib)
                        break

                match = re.search(import_patterns["third_party"], stripped)
                if match:
                    module = match.group(1)
                    if module.split(".")[0] not in imports["standard_lib"]:
                        if module not in imports["third_party"]:
                            imports["third_party"].append(module)

                if re.search(import_patterns["local"], stripped):
                    match = re.search(import_patterns["local"], stripped)
                    module = match.group(1)
                    if module not in imports["local"]:
                        imports["local"].append(module)

                if re.search(import_patterns["relative"], stripped):
                    match = re.search(import_patterns["relative"], stripped)
                    module = match.group(1)
                    if module not in imports["relative"]:
                        imports["relative"].append(module)

            return json.dumps(
                {
                    "import_count": (
                        len(imports["standard_lib"])
                        + len(imports["third_party"])
                        + len(imports["local"])
                        + len(imports["relative"])
                    ),
                    "imports": imports,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            logger.error(f"Import extraction failed: {e}")
            return f'{{"error": "{e}"}}'

    def analyze_file_dependencies(code: str, file_path: str = "unknown") -> str:
        """Analyze dependencies for a single file."""
        try:
            imports_result = find_imports(code)

            patterns = {
                "classes": len(re.findall(r"class\s+(\w+)", code)),
                "functions": len(re.findall(r"def\s+(\w+)", code)),
                "methods": len(re.findall(r"^\s+def\s+(\w+)", code, re.MULTILINE)),
                "imports": json.loads(imports_result),
            }

            patterns["public_api"] = [
                m for m in re.findall(r"def\s+(\w+)", code) if not m.startswith("_")
            ]

            return json.dumps(
                {
                    "file_path": file_path,
                    "dependencies": patterns,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            return f'{{"error": "{e}"}}'

    def build_dependency_graph_from_files(files: dict[str, str]) -> str:
        """Build a dependency graph from multiple files."""
        try:
            graph = {
                "nodes": [],
                "edges": [],
                "orphans": [],
            }

            node_map = {}

            for file_path, code in files.items():
                node_id = file_path
                node_map[node_id] = {
                    "id": node_id,
                    "file_path": file_path,
                    "imports": [],
                }

                imports_match = re.findall(r"(?:from\s+|import\s+)([a-zA-Z_][a-zA-Z0-9_.]*)", code)
                for imp in imports_match:
                    if imp.startswith("."):
                        continue
                    node_map[node_id]["imports"].append(imp)

            for node_id, node_data in node_map.items():
                graph["nodes"].append(
                    {
                        "id": node_id,
                        "file_path": node_data["file_path"],
                        "import_count": len(node_data["imports"]),
                    }
                )

                for imp in node_data["imports"]:
                    found = False
                    for other_id, other_data in node_map.items():
                        if other_id == node_id:
                            continue
                        if imp in other_data["file_path"] or imp in other_data["file_path"].replace(
                            "/", "."
                        ).replace("\\", "."):
                            graph["edges"].append(
                                {
                                    "from": node_id,
                                    "to": other_id,
                                    "type": "imports",
                                }
                            )
                            found = True
                            break
                    if not found:
                        graph["edges"].append(
                            {
                                "from": node_id,
                                "to": imp,
                                "type": "external_import",
                            }
                        )

            graph["node_count"] = len(graph["nodes"])
            graph["edge_count"] = len(graph["edges"])

            return json.dumps(graph, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Dependency graph building failed: {e}")
            return f'{{"error": "{e}"}}'

    def detect_circular_dependencies(files: dict[str, str]) -> str:
        """Detect circular dependencies in a set of files."""
        try:
            graph = {}

            for file_path, code in files.items():
                graph[file_path] = set()
                imports_match = re.findall(r"(?:from\s+|import\s+)([a-zA-Z_][a-zA-Z0-9_.]*)", code)
                for imp in imports_match:
                    if imp.startswith("."):
                        continue
                    graph[file_path].add(imp)

            def find_circular(start, current, path, visited):
                if current in path:
                    return [path[path.index(current) :] + [current]]
                if current in visited:
                    return []

                path.append(current)
                visited.add(current)
                cycles = []

                for neighbor in graph.get(current, set()):
                    if neighbor in graph:
                        cycles.extend(find_circular(start, neighbor, path[:], visited.copy()))

                path.pop()
                return cycles

            all_cycles = []
            visited_global = set()
            for node in graph:
                if node not in visited_global:
                    all_cycles.extend(find_circular(node, node, [], set()))

            unique_cycles = []
            seen = set()
            for cycle in all_cycles:
                cycle_tuple = tuple(sorted(cycle))
                if cycle_tuple not in seen:
                    seen.add(cycle_tuple)
                    unique_cycles.append(cycle)

            return json.dumps(
                {
                    "has_circular_dependencies": len(unique_cycles) > 0,
                    "circular_count": len(unique_cycles),
                    "circular_chains": unique_cycles[:10],
                },
                ensure_ascii=False,
            )
        except Exception as e:
            logger.error(f"Circular dependency detection failed: {e}")
            return f'{{"error": "{e}"}}'

    def calculate_dependency_metrics(files: dict[str, str]) -> str:
        """Calculate dependency-related metrics for a project."""
        try:
            graph = {}
            all_imports = set()
            all_imported_by = {}

            for file_path, code in files.items():
                imports_match = re.findall(r"(?:from\s+|import\s+)([a-zA-Z_][a-zA-Z0-9_.]*)", code)
                graph[file_path] = {
                    "imports": imports_match,
                    "imported_by": [],
                }
                for imp in imports_match:
                    all_imports.add(imp)
                    if imp not in all_imported_by:
                        all_imported_by[imp] = []
                    all_imported_by[imp].append(file_path)

            max_dependents = 0
            most_depended = None
            for imp, dependents in all_imported_by.items():
                if len(dependents) > max_dependents:
                    max_dependents = len(dependents)
                    most_depended = imp

            import_counts = [len(data["imports"]) for data in graph.values()]
            avg_imports = sum(import_counts) / len(import_counts) if import_counts else 0

            return json.dumps(
                {
                    "total_files": len(graph),
                    "unique_external_imports": len(all_imports),
                    "average_imports_per_file": round(avg_imports, 2),
                    "most_used_import": most_depended,
                    "most_used_import_count": max_dependents,
                    "max_imports_in_file": max(import_counts) if import_counts else 0,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            logger.error(f"Dependency metrics calculation failed: {e}")
            return f'{{"error": "{e}"}}'

    def save_dependencies_to_neo4j(dependencies_json: str, source: str = "agent") -> str:
        """Save dependency graph to Neo4j knowledge graph."""
        try:
            import os

            from neo4j import GraphDatabase

            uri = os.getenv("NEO4J_URI") or os.getenv("AURA_INSTANCEID", "")
            username = os.getenv("NEO4J_USERNAME", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            database = os.getenv("NEO4J_DATABASE", "neo4j")

            if not uri:
                return '{"error": "NEO4J_URI or AURA_INSTANCEID not configured"}'

            data = json.loads(dependencies_json)
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])

            driver = GraphDatabase.driver(uri, auth=(username, password))
            with driver.session(database=database) as session:
                for node in nodes:
                    session.run(
                        """
                        MERGE (f:File {file_path: $file_path})
                        SET f.id = $id,
                            f.import_count = $import_count,
                            f.source = $source,
                            f.updated_at = datetime()
                        """,
                        id=node.get("id", ""),
                        file_path=node.get("file_path", ""),
                        import_count=node.get("import_count", 0),
                        source=source,
                    )

                for edge in edges:
                    session.run(
                        """
                        MATCH (f1:File {file_path: $from_path})
                        MATCH (f2:File {file_path: $to_path})
                        MERGE (f1)-[r:DEPENDS_ON {type: $type}]->(f2)
                        SET r.source = $source,
                            r.updated_at = datetime()
                        """,
                        from_path=edge.get("from", ""),
                        to_path=edge.get("to", ""),
                        type=edge.get("type", "imports"),
                        source=source,
                    )

            driver.close()
            return (
                f'{{"status": "success", "nodes_saved": {len(nodes)}, "edges_saved": {len(edges)}}}'
            )
        except ImportError:
            logger.error("neo4j driver not installed")
            return '{"error": "neo4j driver not installed"}'
        except Exception as e:
            logger.error(f"Neo4j save failed: {e}")
            return f'{{"error": "{e}"}}'

    registry.register(
        name="find_imports",
        func=find_imports,
        description="Extract all imports from source code (standard lib, third-party, local, relative)",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Source code to analyze"},
            },
            "required": ["code"],
        },
    )

    registry.register(
        name="analyze_file_dependencies",
        func=analyze_file_dependencies,
        description="Analyze dependencies for a single file (imports, classes, functions, public API)",
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
        name="build_dependency_graph",
        func=build_dependency_graph_from_files,
        description="Build a dependency graph from multiple files",
        parameters={
            "type": "object",
            "properties": {
                "files": {
                    "type": "object",
                    "description": "Dictionary mapping file paths to code content",
                },
            },
            "required": ["files"],
        },
    )

    registry.register(
        name="detect_circular_dependencies",
        func=detect_circular_dependencies,
        description="Detect circular dependencies in a set of files",
        parameters={
            "type": "object",
            "properties": {
                "files": {
                    "type": "object",
                    "description": "Dictionary mapping file paths to code content",
                },
            },
            "required": ["files"],
        },
    )

    registry.register(
        name="calculate_dependency_metrics",
        func=calculate_dependency_metrics,
        description="Calculate dependency-related metrics for a project",
        parameters={
            "type": "object",
            "properties": {
                "files": {
                    "type": "object",
                    "description": "Dictionary mapping file paths to code content",
                },
            },
            "required": ["files"],
        },
    )

    registry.register(
        name="save_dependencies_to_neo4j",
        func=save_dependencies_to_neo4j,
        description="Save dependency graph to Neo4j knowledge graph",
        parameters={
            "type": "object",
            "properties": {
                "dependencies_json": {"type": "string", "description": "JSON with nodes and edges"},
                "source": {"type": "string", "description": "Source identifier"},
            },
            "required": ["dependencies_json"],
        },
    )

    return registry
