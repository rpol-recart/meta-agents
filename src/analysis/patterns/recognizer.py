"""
Pattern Recognizer for Project Analysis Tool

This module provides the main PatternRecognizer class.
"""

import logging
from typing import Any

from .detectors import AntiPatternDetector, CodeMetricsCalculator, PatternDetector
from .models import Pattern, PatternDefinition
from .registry import get_anti_pattern_definitions, get_pattern_definitions

logger = logging.getLogger(__name__)


class PatternRecognizer:
    """
    Pattern recognizer for detecting design patterns and anti-patterns.

    This class provides:
    - Design pattern detection (Singleton, Factory, Observer, etc.)
    - Architectural pattern recognition
    - Anti-pattern identification
    - Cross-module pattern analysis
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the pattern recognizer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._pattern_definitions = get_pattern_definitions()
        self._anti_pattern_definitions = get_anti_pattern_definitions()

    def analyze_code(self, code: str, file_path: str = "unknown") -> dict[str, Any]:
        """
        Perform comprehensive pattern analysis on code.

        Args:
            code: Source code to analyze
            file_path: Path to the source file

        Returns:
            Dictionary containing detected patterns and anti-patterns
        """
        patterns = self.detect_patterns(code, file_path)
        anti_patterns = self.detect_anti_patterns(code, file_path)
        metrics = self.calculate_code_metrics(code)

        return {
            "patterns": [p.to_dict() for p in patterns],
            "anti_patterns": [a.to_dict() for a in anti_patterns],
            "metrics": metrics,
            "pattern_count": len(patterns),
            "anti_pattern_count": len(anti_patterns),
        }

    def detect_patterns(self, code: str, file_path: str = "unknown") -> list[Pattern]:
        """
        Detect design patterns in code.

        Args:
            code: Source code to analyze
            file_path: Path to the source file

        Returns:
            List of detected patterns
        """
        return PatternDetector.detect_patterns(code, self._pattern_definitions, file_path)

    def detect_anti_patterns(self, code: str, file_path: str = "unknown") -> list:
        """
        Detect anti-patterns in code.

        Args:
            code: Source code to analyze
            file_path: Path to the source file

        Returns:
            List of detected anti-patterns
        """
        return AntiPatternDetector.detect_anti_patterns(
            code, self._anti_pattern_definitions, file_path
        )

    def calculate_code_metrics(self, code: str) -> dict[str, Any]:
        """Calculate code quality metrics."""
        return CodeMetricsCalculator.calculate(code)

    def register_pattern(self, pattern: PatternDefinition) -> None:
        """Register a new pattern definition."""
        self._pattern_definitions[pattern.name] = pattern
        logger.info(f"Registered pattern: {pattern.name}")

    def get_registered_patterns(self) -> list[PatternDefinition]:
        """Get all registered pattern definitions."""
        return list(self._pattern_definitions.values())

    def analyze_cross_module_patterns(self, files: dict[str, str]) -> dict[str, Any]:
        """
        Analyze patterns across multiple files.

        Args:
            files: Dictionary mapping file paths to code content

        Returns:
            Dictionary containing cross-module pattern analysis
        """
        all_patterns = []
        file_patterns = {}

        for file_path, code in files.items():
            patterns = self.detect_patterns(code, file_path)
            file_patterns[file_path] = [p.to_dict() for p in patterns]
            all_patterns.extend(patterns)

        pattern_groups: dict[str, list[Pattern]] = {}
        for pattern in all_patterns:
            if pattern.name not in pattern_groups:
                pattern_groups[pattern.name] = []
            pattern_groups[pattern.name].append(pattern)

        cross_module_patterns = []
        for pattern_name, instances in pattern_groups.items():
            if len(instances) > 1:
                cross_module_patterns.append(
                    {
                        "pattern": pattern_name,
                        "occurrences": len(instances),
                        "files": list(set(p.file_path for p in instances)),
                        "confidence": sum(p.confidence for p in instances) / len(instances),
                    }
                )

        return {
            "file_patterns": file_patterns,
            "cross_module_patterns": cross_module_patterns,
            "total_patterns": len(all_patterns),
            "unique_patterns": len(pattern_groups),
        }
