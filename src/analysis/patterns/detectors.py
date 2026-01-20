"""
Pattern Detectors for Project Analysis Tool

This module provides detection logic for patterns and anti-patterns.
"""

import re
from typing import Any

from .models import AntiPattern, Pattern, PatternDefinition, Severity


class PatternDetector:
    """Detector for design patterns."""

    @staticmethod
    def detect_patterns(
        code: str,
        pattern_definitions: dict[str, PatternDefinition],
        file_path: str = "unknown",
    ) -> list[Pattern]:
        """
        Detect design patterns in code.

        Args:
            code: Source code to analyze
            pattern_definitions: Dictionary of pattern definitions
            file_path: Path to the source file

        Returns:
            List of detected patterns
        """
        patterns = []
        lines = code.split("\n")

        for pattern_name, definition in pattern_definitions.items():
            matches = PatternDetector._find_pattern_matches(code, definition, lines)

            for match in matches:
                pattern = Pattern(
                    name=definition.name,
                    pattern_type=definition.pattern_type,
                    description=definition.description,
                    file_path=file_path,
                    start_line=match[0],
                    end_line=match[1],
                    confidence=match[2],
                    properties={"match_content": match[3]},
                )
                patterns.append(pattern)

        return patterns

    @staticmethod
    def _find_pattern_matches(
        code: str, definition: PatternDefinition, lines: list[str]
    ) -> list[tuple[int, int, float, str]]:
        """Find pattern matches in code."""
        matches = []

        for indicator in definition.indicators:
            for i, line in enumerate(lines):
                if re.search(indicator, line, re.IGNORECASE):
                    context_start = max(0, i - 2)
                    context_end = min(len(lines), i + 3)
                    context = "\n".join(lines[context_start:context_end])

                    confidence = PatternDetector._calculate_pattern_confidence(definition, lines, i)

                    matches.append((i + 1, i + 1, confidence, context.strip()))

        return matches

    @staticmethod
    def _calculate_pattern_confidence(
        definition: PatternDefinition, lines: list[str], match_index: int
    ) -> float:
        """Calculate confidence score for a pattern match."""
        base_confidence = 0.7

        structural_bonus = 0.0
        for req in definition.structural_requirements:
            for line in lines:
                if req.lower() in line.lower():
                    structural_bonus += 0.1
                    break

        proximity_bonus = 0.0
        nearby_lines = lines[max(0, match_index - 5) : min(len(lines), match_index + 5)]
        for indicator in definition.indicators[:3]:
            for line in nearby_lines:
                if re.search(indicator, line, re.IGNORECASE):
                    proximity_bonus += 0.05
                    break

        confidence = min(1.0, base_confidence + structural_bonus + proximity_bonus)

        return confidence


class AntiPatternDetector:
    """Detector for anti-patterns."""

    @staticmethod
    def detect_anti_patterns(
        code: str, anti_pattern_definitions: dict[str, dict[str, Any]], file_path: str = "unknown"
    ) -> list[AntiPattern]:
        """
        Detect anti-patterns in code.

        Args:
            code: Source code to analyze
            anti_pattern_definitions: Dictionary of anti-pattern definitions
            file_path: Path to the source file

        Returns:
            List of detected anti-patterns
        """
        anti_patterns = []

        lines = code.split("\n")

        for anti_name, definition in anti_pattern_definitions.items():
            if anti_name == "long_method":
                detected = AntiPatternDetector._detect_long_method(lines, file_path, definition)
                anti_patterns.extend(detected)
            elif anti_name == "feature_envy":
                detected = AntiPatternDetector._detect_feature_envy(code, file_path, definition)
                anti_patterns.extend(detected)
            else:
                matches = AntiPatternDetector._find_anti_pattern_matches(code, definition, lines)
                for match in matches:
                    anti_pattern = AntiPattern(
                        name=definition["name"],
                        description=definition["description"],
                        file_path=file_path,
                        start_line=match[0],
                        end_line=match[1],
                        severity=definition["severity"],
                        impact=definition["impact"],
                        remediation=definition["remediation"],
                        confidence=match[2],
                    )
                    anti_patterns.append(anti_pattern)

        return anti_patterns

    @staticmethod
    def _find_anti_pattern_matches(
        code: str, definition: dict[str, Any], lines: list[str]
    ) -> list[tuple[int, int, float]]:
        """Find anti-pattern matches in code."""
        matches = []

        for indicator in definition.get("indicators", []):
            for i, line in enumerate(lines):
                if re.search(indicator, line):
                    confidence = 0.8
                    matches.append((i + 1, i + 1, confidence))

        return matches

    @staticmethod
    def _detect_long_method(
        lines: list[str], file_path: str, definition: dict[str, Any]
    ) -> list[AntiPattern]:
        """Detect long methods in code."""
        anti_patterns = []
        max_lines = 50

        in_method = False
        method_start = 0
        indent_level = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            if in_method:
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and stripped:
                    method_length = i - method_start
                    if method_length > max_lines:
                        anti_pattern = AntiPattern(
                            name=definition["name"],
                            description=definition["description"],
                            file_path=file_path,
                            start_line=method_start + 1,
                            end_line=i,
                            severity=Severity.MEDIUM,
                            impact=f"Method spans {method_length} lines",
                            remediation=definition["remediation"],
                            confidence=min(1.0, method_length / 100),
                        )
                        anti_patterns.append(anti_pattern)
                    in_method = False
            else:
                if any(kw in stripped for kw in ["def ", "async def "]):
                    in_method = True
                    method_start = i
                    indent_level = len(line) - len(line.lstrip())

        return anti_patterns

    @staticmethod
    def _detect_feature_envy(
        code: str, file_path: str, definition: dict[str, Any]
    ) -> list[AntiPattern]:
        """Detect feature envy anti-pattern."""
        anti_patterns = []

        lines = code.split("\n")
        method_pattern = r"(?:def|async def)\s+(\w+)"

        for i, line in enumerate(lines):
            match = re.search(method_pattern, line)
            if match:
                method_name = match.group(1)

                if i + 1 < len(lines):
                    method_body = []
                    j = i + 1
                    while j < len(lines):
                        if (
                            lines[j].strip()
                            and not lines[j].startswith(" ")
                            and not lines[j].startswith("\t")
                        ):
                            break
                        method_body.append(lines[j])
                        j += 1

                    other_class_refs = len(re.findall(r"self\.\w+\.\w+", "\n".join(method_body)))
                    self_refs = len(re.findall(r"self\.\w+(?!\.\w+)", "\n".join(method_body)))

                    if other_class_refs > self_refs * 2 and other_class_refs > 3:
                        anti_pattern = AntiPattern(
                            name=definition["name"],
                            description=definition["description"],
                            file_path=file_path,
                            start_line=i + 1,
                            end_line=j,
                            severity=Severity.MEDIUM,
                            impact=f"Method {method_name} uses more data from other classes",
                            remediation=definition["remediation"],
                            confidence=0.7,
                        )
                        anti_patterns.append(anti_pattern)

        return anti_patterns


class CodeMetricsCalculator:
    """Calculator for code quality metrics."""

    @staticmethod
    def calculate(code: str) -> dict[str, Any]:
        """Calculate code quality metrics."""
        lines = code.split("\n")
        non_empty_lines = [ln for ln in lines if ln.strip()]

        total_lines = len(lines)
        code_lines = len(non_empty_lines)
        comment_lines = len([ln for ln in lines if ln.strip().startswith("#")])
        blank_lines = total_lines - code_lines

        cyclomatic_complexity = 1
        for line in lines:
            if any(
                kw in line for kw in ["if ", "elif ", "while ", "for ", "and ", "or ", "?", "case "]
            ):
                cyclomatic_complexity += 1

        indent_levels = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_levels.append(indent)

        avg_indent = sum(indent_levels) / len(indent_levels) if indent_levels else 0

        docstring_pattern = r'"""[\s\S]*?"""'
        docstrings = len(re.findall(docstring_pattern, code))

        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "blank_lines": blank_lines,
            "comment_ratio": comment_lines / code_lines if code_lines else 0,
            "cyclomatic_complexity": cyclomatic_complexity,
            "average_indent_level": avg_indent,
            "docstring_count": docstrings,
        }
