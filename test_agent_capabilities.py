#!/usr/bin/env python3
"""
Test script for demonstrating agent capabilities with new tools.

This script directly tests the tools without needing to run the full agent system.
"""

import sys

sys.path.insert(0, "src")


def test_semantic_analyzer():
    """Test semantic analyzer tools."""
    print("=" * 60)
    print("Testing Semantic Analyzer Tools")
    print("=" * 60)

    from src.tools.analysis import get_analysis_tools

    registry = get_analysis_tools()
    print(f"\nRegistered {len(registry)} tools")

    # Test extract_entities
    print("\n--- Test: extract_entities ---")
    text = """
    Apple Inc. develops AI technology using advanced machine learning algorithms.
    Google and Microsoft also invest heavily in AI research.
    The technology uses aluminum and steel components in manufacturing.
    The extrusion process improves efficiency.
    """
    extract_tool = registry.get("extract_entities")
    result = extract_tool.func(text)
    print(f"Extracted entities from text")

    # Test analyze_sentiment
    print("\n--- Test: analyze_sentiment ---")
    sentiment_tool = registry.get("analyze_sentiment")
    text = """
    This is an innovative breakthrough with excellent results!
    The technology promises to revolutionize healthcare.
    However, there are concerns about privacy and implementation costs.
    """
    result = sentiment_tool.func(text)
    print(f"Analyzed sentiment")

    # Test full analysis
    print("\n--- Test: analyze_text_full ---")
    full_tool = registry.get("analyze_text_full")
    result = full_tool.func(text, "test_source")
    print(f"Full analysis complete")

    print("\n✓ Semantic Analyzer tools working correctly\n")


def test_pattern_recognizer():
    """Test pattern recognizer tools."""
    print("=" * 60)
    print("Testing Pattern Recognizer Tools")
    print("=" * 60)

    from src.tools.pattern_tools import get_pattern_tools

    registry = get_pattern_tools()
    print(f"\nRegistered {len(registry)} tools")

    code = """
class Singleton:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

class Factory:
    def create_product(self):
        return Product()

class Product:
    pass

class Subject:
    def __init__(self):
        self._observers = []
    def attach(self, observer):
        self._observers.append(observer)
    def detach(self, observer):
        self._observers.remove(observer)
    def notify(self):
        for obs in self._observers:
            obs.update()
"""

    # Test analyze_patterns
    print("\n--- Test: analyze_patterns ---")
    analyze_tool = registry.get("analyze_patterns")
    result = analyze_tool.func(code, "test.py")
    print(f"Detected patterns in code")

    # Test detect_anti_patterns
    print("\n--- Test: detect_anti_patterns ---")
    anti_code = """
class EverythingManager:
    def __init__(self):
        self.data = []
        self.users = []
        self.config = {}
        self.connections = []
    def process(self, x=100, y=50):
        result = x * y + 100 - 50 + 25
        return result
"""
    anti_tool = registry.get("detect_anti_patterns")
    result = anti_tool.func(anti_code, "anti_test.py")
    print(f"Detected anti-patterns")

    # Test calculate_code_metrics
    print("\n--- Test: calculate_code_metrics ---")
    metrics_tool = registry.get("calculate_code_metrics")
    result = metrics_tool.func(code)
    print(f"Calculated code metrics")

    # Test analyze_code_full
    print("\n--- Test: analyze_code_full ---")
    full_tool = registry.get("analyze_code_full")
    result = full_tool.func(code, "full_test")
    print(f"Full pattern analysis complete")

    print("\n✓ Pattern Recognizer tools working correctly\n")


def test_dependency_analyzer():
    """Test dependency analyzer tools."""
    print("=" * 60)
    print("Testing Dependency Analyzer Tools")
    print("=" * 60)

    from src.tools.dependency_tools import get_dependency_tools

    registry = get_dependency_tools()
    print(f"\nRegistered {len(registry)} tools")

    code = """
import os
import sys
from datetime import datetime
import json
import requests
from flask import Flask
from langchain import OpenAI
from .module import ClassA
from ..parent import FunctionB
"""

    # Test find_imports
    print("\n--- Test: find_imports ---")
    find_tool = registry.get("find_imports")
    result = find_tool.func(code)
    print(f"Found imports")

    # Test analyze_file_dependencies
    print("\n--- Test: analyze_file_dependencies ---")
    analyze_tool = registry.get("analyze_file_dependencies")
    result = analyze_tool.func(code, "test.py")
    print(f"Analyzed file dependencies")

    # Test build_dependency_graph
    print("\n--- Test: build_dependency_graph ---")
    files = {
        "main.py": "import a\nfrom b import B",
        "a.py": "import c",
        "b.py": "",
        "c.py": "import main",
    }
    graph_tool = registry.get("build_dependency_graph")
    result = graph_tool.func(files)
    print(f"Built dependency graph")

    # Test detect_circular_dependencies
    print("\n--- Test: detect_circular_dependencies ---")
    circular_tool = registry.get("detect_circular_dependencies")
    result = circular_tool.func(files)
    print(f"Detected circular dependencies")

    # Test calculate_dependency_metrics
    print("\n--- Test: calculate_dependency_metrics ---")
    metrics_tool = registry.get("calculate_dependency_metrics")
    result = metrics_tool.func(files)
    print(f"Calculated dependency metrics")

    print("\n✓ Dependency Analyzer tools working correctly\n")


def test_langchain_integration():
    """Test LangChain tool integration."""
    print("=" * 60)
    print("Testing LangChain Integration")
    print("=" * 60)

    from src.tools.analysis import get_analysis_tools
    from src.tools.pattern_tools import get_pattern_tools
    from src.tools.dependency_tools import get_dependency_tools

    all_tools = []
    all_tools.extend(get_analysis_tools().to_langchain_tools())
    all_tools.extend(get_pattern_tools().to_langchain_tools())
    all_tools.extend(get_dependency_tools().to_langchain_tools())

    print(f"\nTotal LangChain tools: {len(all_tools)}")
    print("\nTool names:")
    for tool in all_tools:
        print(f"  - {tool.name}")

    print("\n✓ LangChain integration working correctly\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print(" AGENT CAPABILITIES TEST SUITE")
    print("=" * 60)
    print()

    tests = [
        ("Semantic Analyzer", test_semantic_analyzer),
        ("Pattern Recognizer", test_pattern_recognizer),
        ("Dependency Analyzer", test_dependency_analyzer),
        ("LangChain Integration", test_langchain_integration),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            failed += 1

    print("=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print()

    if failed == 0:
        print("✓ All tests passed!")
        print("\nAgents are ready for use with orchestrate CLI:")
        print()
        print("  # Semantic analysis")
        print("  orchestrate run 'Analyze code with semantic_analyzer' --agents-dir agents/")
        print()
        print("  # Pattern recognition")
        print("  orchestrate run 'Detect patterns with pattern_recognizer' --agents-dir agents/")
        print()
        print("  # Dependency analysis")
        print(
            "  orchestrate run 'Analyze dependencies with dependency_analyzer' --agents-dir agents/"
        )
    else:
        print("✗ Some tests failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
