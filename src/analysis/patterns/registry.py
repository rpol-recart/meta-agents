"""
Pattern Registry for Project Analysis Tool

This module contains all pattern and anti-pattern definitions.
"""

from typing import Any

from .models import PatternDefinition, PatternType, Severity


def get_pattern_definitions() -> dict[str, PatternDefinition]:
    """Get all pattern definitions."""
    return {
        **_get_creational_patterns(),
        **_get_structural_patterns(),
        **_get_behavioral_patterns(),
        **_get_architectural_patterns(),
    }


def _get_creational_patterns() -> dict[str, PatternDefinition]:
    """Get creational design patterns."""
    return {
        "singleton": PatternDefinition(
            name="Singleton",
            pattern_type=PatternType.CREATIONAL,
            description="Ensure a class has only one instance and provide a global point of access to it",
            indicators=[
                r"self\._instance",
                r"@classmethod\s+def\s+get_instance",
                r"if\s+instance\s+is\s+None",
                r"static\s+instance",
            ],
            structural_requirements=["private constructor", "static instance variable"],
        ),
        "factory_method": PatternDefinition(
            name="Factory Method",
            pattern_type=PatternType.CREATIONAL,
            description="Define an interface for creating an object, but let subclasses decide which class to instantiate",
            indicators=[
                r"def\s+create\w+",
                r"def\s+make\w+",
                r"def\s+new\w+",
            ],
            structural_requirements=["abstract creator class", "concrete products"],
        ),
        "abstract_factory": PatternDefinition(
            name="Abstract Factory",
            pattern_type=PatternType.CREATIONAL,
            description="Provide an interface for creating families of related or dependent objects",
            indicators=[
                r"interface\s+\w+Factory",
                r"class\s+\w+Factory",
                r"create\w+\(\)",
            ],
            structural_requirements=["abstract products", "concrete factories"],
        ),
        "builder": PatternDefinition(
            name="Builder",
            pattern_type=PatternType.CREATIONAL,
            description="Separate the construction of a complex object from its representation",
            indicators=[
                r"class\s+\w+Builder",
                r"def\s+build\w+",
                r"\.build\(\)",
                r"fluent\s+interface",
            ],
            structural_requirements=["director", "builder interface", "concrete builders"],
        ),
    }


def _get_structural_patterns() -> dict[str, PatternDefinition]:
    """Get structural design patterns."""
    return {
        "adapter": PatternDefinition(
            name="Adapter",
            pattern_type=PatternType.STRUCTURAL,
            description="Convert the interface of a class into another interface clients expect",
            indicators=[
                r"inherits\s+\w+Interface",
                r"implements\s+\w+Interface",
                r"def\s+adapt\w+",
            ],
            structural_requirements=["target interface", "adaptee", "adapter"],
        ),
        "decorator": PatternDefinition(
            name="Decorator",
            pattern_type=PatternType.STRUCTURAL,
            description="Attach additional responsibilities to an object dynamically",
            indicators=[
                r"def\s+__init__\(self,\s+\w+\)",
                r"self\.\w+\s*=\s*\w+",
                r"super\(\)\.__init__\(\w+\)",
            ],
            structural_requirements=["component interface", "concrete component", "decorator"],
        ),
        "facade": PatternDefinition(
            name="Facade",
            pattern_type=PatternType.STRUCTURAL,
            description="Provide a unified interface to a set of interfaces in a subsystem",
            indicators=[
                r"class\s+\w+Facade",
                r"def\s+\w+\(\w+\)",
                r"subsystem",
            ],
            structural_requirements=["facade class", "subsystem classes"],
        ),
        "proxy": PatternDefinition(
            name="Proxy",
            pattern_type=PatternType.STRUCTURAL,
            description="Provide a surrogate or placeholder for another object",
            indicators=[
                r"class\s+\w+Proxy",
                r"self\._real\w+",
                r"lazy\s+initialization",
            ],
            structural_requirements=["subject interface", "real subject", "proxy"],
        ),
    }


def _get_behavioral_patterns() -> dict[str, PatternDefinition]:
    """Get behavioral design patterns."""
    return {
        "observer": PatternDefinition(
            name="Observer",
            pattern_type=PatternType.BEHAVIORAL,
            description="Define a one-to-many dependency between objects so that when one object changes state",
            indicators=[
                r"def\s+attach\(",
                r"def\s+detach\(",
                r"def\s+notify\(",
                r"self\._observers",
                r"addEventListener",
            ],
            structural_requirements=["subject", "observer interface", "concrete observers"],
        ),
        "strategy": PatternDefinition(
            name="Strategy",
            pattern_type=PatternType.BEHAVIORAL,
            description="Define a family of algorithms, encapsulate each one, and make them interchangeable",
            indicators=[
                r"class\s+\w+Strategy",
                r"self\._strategy",
                r"def\s+execute\(",
            ],
            structural_requirements=["strategy interface", "concrete strategies", "context"],
        ),
        "command": PatternDefinition(
            name="Command",
            pattern_type=PatternType.BEHAVIORAL,
            description="Encapsulate a request as an object, thereby letting you parameterize different requests",
            indicators=[
                r"class\s+\w+Command",
                r"def\s+execute\(",
                r"def\s+undo\(",
            ],
            structural_requirements=["command interface", "concrete commands", "invoker"],
        ),
        "state": PatternDefinition(
            name="State",
            pattern_type=PatternType.BEHAVIORAL,
            description="Allow an object to alter its behavior when its internal state changes",
            indicators=[
                r"class\s+\w+State",
                r"self\._state",
                r"state\s*=\s*\w+State",
            ],
            structural_requirements=["state interface", "concrete states", "context"],
        ),
        "iterator": PatternDefinition(
            name="Iterator",
            pattern_type=PatternType.BEHAVIORAL,
            description="Provide a way to access the elements of an aggregate object sequentially",
            indicators=[
                r"def\s+__iter__\(",
                r"def\s+__next__\(",
                r"yield",
            ],
            structural_requirements=["aggregate interface", "iterator interface"],
        ),
    }


def _get_architectural_patterns() -> dict[str, PatternDefinition]:
    """Get architectural patterns."""
    return {
        "mvc": PatternDefinition(
            name="MVC",
            pattern_type=PatternType.ARCHITECTURAL,
            description="Model-View-Controller pattern separating data, presentation, and control logic",
            indicators=[
                r"class\s+\w+Controller",
                r"class\s+\w+Model",
                r"class\s+\w+View",
            ],
            structural_requirements=["model", "view", "controller"],
        ),
        "microservices": PatternDefinition(
            name="Microservices",
            pattern_type=PatternType.ARCHITECTURAL,
            description="Structure an application as a collection of loosely coupled services",
            indicators=[
                r"@app\.route",
                r"FastAPI|Flask|django",
                r"service\s*=\s*",
                r"API|endpoint",
            ],
            structural_requirements=[
                "independent services",
                "API communication",
                "service discovery",
            ],
        ),
        "repository": PatternDefinition(
            name="Repository",
            pattern_type=PatternType.ARCHITECTURAL,
            description="Mediate between the domain and data mapping layers",
            indicators=[
                r"class\s+\w+Repository",
                r"def\s+get_by_id\(",
                r"def\s+save\(",
                r"def\s+delete\(",
            ],
            structural_requirements=["repository interface", "entity", "data source"],
        ),
    }


def get_anti_pattern_definitions() -> dict[str, dict[str, Any]]:
    """Get all anti-pattern definitions."""
    return {
        "god_object": {
            "name": "God Object",
            "description": "A class that knows too much or does too much",
            "indicators": [
                r"class\s+\w+.*(?:Manager|Controller|Handler)",
                r"def\s+\w+\s*\([^)]{100,}",
            ],
            "severity": Severity.HIGH,
            "impact": "High coupling, difficult maintenance, low reusability",
            "remediation": "Split into smaller, focused classes",
        },
        "spaghetti_code": {
            "name": "Spaghetti Code",
            "description": "Code with tangled control flow and complex nested conditionals",
            "indicators": [
                r"if.*else.*if.*else.*if.*else",
                r"for.*for.*for",
                r"while.*while.*while",
                r"goto",
            ],
            "severity": Severity.CRITICAL,
            "impact": "Extremely difficult to understand and maintain",
            "remediation": "Refactor using functions, classes, and clear structure",
        },
        "copy_paste": {
            "name": "Copy-Paste Programming",
            "description": "Duplicated code blocks with minor modifications",
            "indicators": [
                r"(\w{50,}).*\n.*\1",
            ],
            "severity": Severity.MEDIUM,
            "impact": "Code bloat, inconsistency, increased bug potential",
            "remediation": "Extract common functionality into reusable functions or classes",
        },
        "magic_numbers": {
            "name": "Magic Numbers",
            "description": "Using hard-coded numeric literals without explanation",
            "indicators": [
                r"= [0-9]{2,}(?!\s*(?:or|and|==|!=))",
                r"= -[0-9]+",
            ],
            "severity": Severity.LOW,
            "impact": "Poor code readability and maintainability",
            "remediation": "Replace with named constants",
        },
        "long_method": {
            "name": "Long Method",
            "description": "A method that is too long and does too much",
            "indicators": [],
            "severity": Severity.MEDIUM,
            "impact": "Difficult to understand, test, and maintain",
            "remediation": "Extract smaller methods with single responsibilities",
        },
        "feature_envy": {
            "name": "Feature Envy",
            "description": "A method that uses more data from other classes than its own",
            "indicators": [],
            "severity": Severity.MEDIUM,
            "impact": "Poor encapsulation, tight coupling",
            "remediation": "Move method to the class whose data it uses",
        },
    }
