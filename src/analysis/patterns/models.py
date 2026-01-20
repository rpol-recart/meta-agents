"""
Pattern Recognition Models for Project Analysis Tool

This module provides data classes and enums for pattern recognition.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PatternType(Enum):
    """Enumeration of pattern types."""

    CREATIONAL = "creational"
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    ARCHITECTURAL = "architectural"
    DOMAIN_SPECIFIC = "domain_specific"
    ANTI_PATTERN = "anti_pattern"


class Severity(Enum):
    """Enumeration of severity levels for anti-patterns."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Pattern:
    """Represents a detected pattern in code."""

    name: str
    pattern_type: PatternType
    description: str
    file_path: str
    start_line: int
    end_line: int
    confidence: float
    properties: dict[str, Any] = field(default_factory=dict)
    related_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.pattern_type.value,
            "description": self.description,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "confidence": self.confidence,
            "properties": self.properties,
            "related_patterns": self.related_patterns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pattern":
        return cls(
            name=data["name"],
            pattern_type=PatternType(data["type"]),
            description=data["description"],
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            confidence=data.get("confidence", 1.0),
            properties=data.get("properties", {}),
            related_patterns=data.get("related_patterns", []),
        )


@dataclass
class AntiPattern:
    """Represents a detected anti-pattern in code."""

    name: str
    description: str
    file_path: str
    start_line: int
    end_line: int
    severity: Severity
    impact: str
    remediation: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "severity": self.severity.value,
            "impact": self.impact,
            "remediation": self.remediation,
            "confidence": self.confidence,
        }


@dataclass
class PatternMatch:
    """Represents a match of a pattern definition in code."""

    pattern_name: str
    file_path: str
    start_line: int
    end_line: int
    matched_content: str
    context: str
    confidence: float


class PatternDefinition:
    """Definition of a pattern for detection."""

    def __init__(
        self,
        name: str,
        pattern_type: PatternType,
        description: str,
        indicators: list[str],
        structural_requirements: list[str] | None = None,
        anti_pattern_indicators: list[str] | None = None,
    ):
        self.name = name
        self.pattern_type = pattern_type
        self.description = description
        self.indicators = indicators
        self.structural_requirements = structural_requirements or []
        self.anti_pattern_indicators = anti_pattern_indicators or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.pattern_type.value,
            "description": self.description,
            "indicators": self.indicators,
            "structural_requirements": self.structural_requirements,
            "anti_pattern_indicators": self.anti_pattern_indicators,
        }
