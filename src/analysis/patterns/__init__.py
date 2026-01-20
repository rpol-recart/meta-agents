"""
Pattern Recognition Module for Project Analysis Tool

This module provides capabilities for detecting software design patterns,
architectural patterns, and anti-patterns in code repositories.
"""

from .models import (
    AntiPattern,
    Pattern,
    PatternDefinition,
    PatternMatch,
    PatternType,
    Severity,
)
from .recognizer import PatternRecognizer

__all__ = [
    "PatternType",
    "Severity",
    "Pattern",
    "AntiPattern",
    "PatternMatch",
    "PatternDefinition",
    "PatternRecognizer",
]
