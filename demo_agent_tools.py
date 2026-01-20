#!/usr/bin/env python3
"""
Quick demo of agent tools without LLM API calls.

This script demonstrates the new tools directly without calling the LLM,
showing how the tools work and what they return.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def demo_semantic_tools():
    """Demo semantic analysis tools."""
    print("\n" + "=" * 60)
    print("SEMANTIC ANALYZER TOOLS DEMO")
    print("=" * 60)

    from src.tools.analysis import get_analysis_tools

    registry = get_analysis_tools()

    text = """
    Apple Inc. announced a breakthrough in AI technology today.
    The new system uses advanced machine learning algorithms for natural language processing.
    This innovative development promises to transform industries significantly.
    Google and Microsoft are also investing heavily in similar technologies.
    The extrusion process involves heat treatment and forging of aluminum materials.
    """

    print("\nInput text:")
    print(text)
    print("-" * 60)

    extract_tool = registry.get("extract_entities")
    sentiment_tool = registry.get("analyze_sentiment")

    print("\n1. Extracting entities...")
    entities_result = extract_tool.func(text)
    print(entities_result)

    print("\n2. Analyzing sentiment...")
    sentiment_result = sentiment_tool.func(text)
    print(sentiment_result)


def demo_pattern_tools():
    """Demo pattern recognition tools."""
    print("\n" + "=" * 60)
    print("PATTERN RECOGNIZER TOOLS DEMO")
    print("=" * 60)

    from src.tools.pattern_tools import get_pattern_tools

    registry = get_pattern_tools()

    code = """
class Singleton:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.data = []
        self.users = []
        self.config = {}

class DataProcessor:
    def __init__(self, name: str):
        self.name = name
        self.cache = {}
    
    def process(self, items):
        result = []
        for item in items:
            result.append({"item": item, "processed": True})
        return result
    
    def save_to_file(self, filename: str, data: str):
        with open(filename, 'w') as f:
            f.write(json.dumps(data))
        return True

def calculate(x):
    result = x * 100 + 50
    return result
"""

    print("\nInput code:")
    print(code[:200] + "..." if len(code) > 200 else code)
    print("-" * 60)

    analyze_tool = registry.get("analyze_patterns")
    anti_tool = registry.get("detect_anti_patterns")
    metrics_tool = registry.get("calculate_code_metrics")

    print("\n1. Detecting patterns...")
    patterns_result = analyze_tool.func(code, "demo.py")
    print(f"Found {json.loads(patterns_result)['pattern_count']} patterns")

    print("\n2. Detecting anti-patterns...")
    anti_result = anti_tool.func(code, "demo.py")
    anti_data = json.loads(anti_result)
    print(f"Found {anti_data['anti_pattern_count']} anti-patterns")
    if anti_data["severity_counts"]:
        print(f"  Severity: {anti_data['severity_counts']}")

    print("\n3. Calculating code metrics...")
    metrics_result = metrics_tool.func(code)
    metrics = json.loads(metrics_result)
    print(f"  Lines: {metrics['code_lines']}")
    print(f"  Complexity: {metrics['cyclomatic_complexity']}")


def demo_dependency_tools():
    """Demo dependency analysis tools."""
    print("\n" + "=" * 60)
    print("DEPENDENCY ANALYZER TOOLS DEMO")
    print("=" * 60)

    from src.tools.dependency_tools import get_dependency_tools

    registry = get_dependency_tools()

    code = """
import os
import json
from datetime import datetime
from typing import List, Dict, Optional

from src.module_a import ClassA
from src.module_b import ClassB

import requests
from flask import Flask
"""

    print("\nInput code:")
    print(code)
    print("-" * 60)

    find_tool = registry.get("find_imports")
    analyze_tool = registry.get("analyze_file_dependencies")

    print("\n1. Finding imports...")
    imports_result = find_tool.func(code)
    imports_data = json.loads(imports_result)
    print(f"Total imports: {imports_data['import_count']}")
    print(f"  Standard lib: {imports_data['imports']['standard_lib']}")
    print(f"  Third party: {imports_data['imports']['third_party']}")
    print(f"  Local: {imports_data['imports']['local']}")

    print("\n2. Analyzing file dependencies...")
    deps_result = analyze_tool.func(code, "demo.py")
    deps_data = json.loads(deps_result)
    print(f"File: {deps_data['file_path']}")
    print(f"Classes: {deps_data['dependencies']['classes']}")
    print(f"Functions: {deps_data['dependencies']['functions']}")
    print(f"Public API: {deps_data['dependencies']['public_api']}")


def demo_dependency_graph():
    """Demo dependency graph building."""
    print("\n" + "=" * 60)
    print("DEPENDENCY GRAPH DEMO")
    print("=" * 60)

    from src.tools.dependency_tools import get_dependency_tools

    registry = get_dependency_tools()

    files = {
        "main.py": "from utils import helper\nfrom models import User",
        "utils.py": "import os\nimport json",
        "models.py": "from database import Base\nfrom datetime import datetime",
        "database.py": "import sqlite3\nfrom utils import helper",
        "views.py": "from flask import Flask\nfrom models import User\nfrom views_ui import render",
        "views_ui.py": "",  # Orphan - no imports
    }

    graph_tool = registry.get("build_dependency_graph")
    circular_tool = registry.get("detect_circular_dependencies")
    metrics_tool = registry.get("calculate_dependency_metrics")

    print("\n1. Building dependency graph...")
    graph_result = graph_tool.func(files)
    graph_data = json.loads(graph_result)
    print(f"Nodes: {graph_data['node_count']}")
    print(f"Edges: {graph_data['edge_count']}")

    print("\n2. Detecting circular dependencies...")
    circular_result = circular_tool.func(files)
    circular_data = json.loads(circular_result)
    has_circular = circular_data.get("has_circular_dependencies", False)
    print(f"Has circular dependencies: {has_circular}")
    if circular_data.get("circular_chains"):
        print(f"Circular chains found: {len(circular_data['circular_chains'])}")

    print("\n3. Calculating dependency metrics...")
    metrics_result = metrics_tool.func(files)
    metrics_data = json.loads(metrics_result)
    print(f"Total files: {metrics_data['total_files']}")
    print(f"External imports: {metrics_data['unique_external_imports']}")
    print(f"Avg imports/file: {metrics_data['average_imports_per_file']}")


def main():
    print("\n" + "#" * 60)
    print("# AGENT TOOLS DEMONSTRATION")
    print("#" * 60)

    demo_semantic_tools()
    demo_pattern_tools()
    demo_dependency_tools()
    demo_dependency_graph()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETED")
    print("=" * 60)
    print("\nTo run agents with LLM, use:")
    print("  python test_agent_prompt.py --agent semantic_analyzer")
    print("  python test_agent_prompt.py --agent pattern_recognizer --code")
    print("  python test_agent_prompt.py --agent dependency_analyzer --code")
    print()


if __name__ == "__main__":
    main()
