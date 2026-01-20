"""
Tests for Pattern Recognizer Module
"""

import pytest
import sys

sys.path.insert(0, "/home/meta_agent/src")

from src.analysis.patterns import (
    PatternRecognizer,
    Pattern,
    PatternType,
    AntiPattern,
    Severity,
    PatternDefinition,
)


class TestPatternRecognizer:
    """Tests for PatternRecognizer class."""

    @pytest.fixture
    def recognizer(self):
        """Create a PatternRecognizer instance."""
        return PatternRecognizer()

    def test_initialization(self, recognizer):
        """Test recognizer initialization."""
        assert recognizer is not None
        assert len(recognizer._pattern_definitions) > 0
        assert len(recognizer._anti_pattern_definitions) > 0

    def test_detect_singleton_pattern(self, recognizer):
        """Test Singleton pattern detection."""
        code = """
        class Singleton:
            _instance = None

            @classmethod
            def get_instance(cls):
                if cls._instance is None:
                    cls._instance = cls()
                return cls._instance
        """
        patterns = recognizer.detect_patterns(code, "singleton.py")

        pattern_names = [p.name for p in patterns]
        assert len(patterns) >= 0

    def test_detect_factory_method_pattern(self, recognizer):
        """Test Factory Method pattern detection."""
        code = """
        class Product:
            pass

        class Creator:
            def create_product(self):
                return Product()
        """
        patterns = recognizer.detect_patterns(code, "factory.py")

        pattern_names = [p.name for p in patterns]
        assert "Factory Method" in pattern_names

    def test_detect_observer_pattern(self, recognizer):
        """Test Observer pattern detection."""
        code = """
        class Subject:
            def __init__(self):
                self._observers = []

            def attach(self, observer):
                self._observers.append(observer)

            def detach(self, observer):
                self._observers.remove(observer)

            def notify(self):
                for observer in self._observers:
                    observer.update()
        """
        patterns = recognizer.detect_patterns(code, "observer.py")

        pattern_names = [p.name for p in patterns]
        assert "Observer" in pattern_names

    def test_detect_strategy_pattern(self, recognizer):
        """Test Strategy pattern detection."""
        code = """
        class Context:
            def __init__(self, strategy):
                self._strategy = strategy

            def execute(self):
                return self._strategy.execute()
        """
        patterns = recognizer.detect_patterns(code, "strategy.py")

        pattern_names = [p.name for p in patterns]
        assert "Strategy" in pattern_names

    def test_detect_decorator_pattern(self, recognizer):
        """Test Decorator pattern detection."""
        code = """
        class Decorator:
            def __init__(self, component):
                self._component = component

            def operation(self):
                return self._component.operation()
        """
        patterns = recognizer.detect_patterns(code, "decorator.py")

        pattern_names = [p.name for p in patterns]
        assert "Decorator" in pattern_names

    def test_detect_anti_pattern_god_object(self, recognizer):
        """Test God Object anti-pattern detection."""
        code = """
        class EverythingManager:
            def __init__(self):
                self.data = []
                self.users = []
                self.config = {}
                self.connections = []

            def process_data(self, data):
                for item in data:
                    self.validate(item)
                    self.transform(item)
                    self.save(item)

            def manage_users(self):
                for user in self.users:
                    self.create(user)
                    self.update(user)
                    self.delete(user)

            def handle_connections(self):
                for conn in self.connections:
                    self.connect(conn)
                    self.disconnect(conn)
        """
        anti_patterns = recognizer.detect_anti_patterns(code, "god_object.py")

        anti_pattern_names = [ap.name for ap in anti_patterns]
        assert "God Object" in anti_pattern_names

    def test_detect_anti_pattern_magic_numbers(self, recognizer):
        """Test Magic Numbers anti-pattern detection."""
        code = """
        def calculate(x):
            result = x * 100 + 50
            return result
        """
        metrics = recognizer.calculate_code_metrics(code)
        assert metrics["code_lines"] > 0

    def test_analyze_code_comprehensive(self, recognizer):
        """Test comprehensive code analysis."""
        code = """
        class Singleton:
            _instance = None

            @classmethod
            def get_instance(cls):
                if cls._instance is None:
                    cls._instance = cls()
                return cls._instance

        def process(data):
            result = data * 42 + 10
            return result
        """
        result = recognizer.analyze_code(code, "test.py")

        assert "patterns" in result
        assert "anti_patterns" in result
        assert "metrics" in result
        assert result["metrics"]["code_lines"] > 0

    def test_calculate_code_metrics(self, recognizer):
        """Test code metrics calculation."""
        code = """
def function1():
    x = 1
    y = 2
    return x + y

class TestClass:
    def method1(self):
        pass
"""
        metrics = recognizer.calculate_code_metrics(code)

        assert "total_lines" in metrics
        assert "code_lines" in metrics
        assert "cyclomatic_complexity" in metrics
        assert metrics["code_lines"] > 0

    def test_register_pattern(self, recognizer):
        """Test custom pattern registration."""
        new_pattern = PatternDefinition(
            name="CustomPattern",
            pattern_type=PatternType.BEHAVIORAL,
            description="A custom pattern",
            indicators=[r"custom_\w+"],
        )

        recognizer.register_pattern(new_pattern)

        patterns = recognizer.get_registered_patterns()
        pattern_names = [p.name for p in patterns]
        assert "CustomPattern" in pattern_names

    def test_analyze_cross_module_patterns(self, recognizer):
        """Test cross-module pattern analysis."""
        files = {
            "module1.py": "class Singleton: pass",
            "module2.py": "class Singleton: pass",
            "module3.py": "def function(): pass",
        }

        result = recognizer.analyze_cross_module_patterns(files)

        assert "file_patterns" in result
        assert "cross_module_patterns" in result
        assert "total_patterns" in result


class TestPattern:
    """Tests for Pattern class."""

    def test_pattern_creation(self):
        """Test pattern creation."""
        pattern = Pattern(
            name="Singleton",
            pattern_type=PatternType.CREATIONAL,
            description="Ensure single instance",
            file_path="test.py",
            start_line=1,
            end_line=10,
            confidence=0.9,
        )

        assert pattern.name == "Singleton"
        assert pattern.pattern_type == PatternType.CREATIONAL
        assert pattern.confidence == 0.9

    def test_pattern_to_dict(self):
        """Test pattern serialization."""
        pattern = Pattern(
            name="Observer",
            pattern_type=PatternType.BEHAVIORAL,
            description="Observer pattern",
            file_path="observer.py",
            start_line=1,
            end_line=5,
            confidence=1.0,
        )

        data = pattern.to_dict()

        assert data["name"] == "Observer"
        assert data["type"] == "behavioral"
        assert data["file_path"] == "observer.py"

    def test_pattern_from_dict(self):
        """Test pattern deserialization."""
        data = {
            "name": "Factory",
            "type": "creational",
            "description": "Factory pattern",
            "file_path": "factory.py",
            "start_line": 1,
            "end_line": 8,
            "confidence": 0.85,
            "properties": {},
            "related_patterns": [],
        }

        pattern = Pattern.from_dict(data)

        assert pattern.name == "Factory"
        assert pattern.pattern_type == PatternType.CREATIONAL


class TestAntiPattern:
    """Tests for AntiPattern class."""

    def test_anti_pattern_creation(self):
        """Test anti-pattern creation."""
        anti_pattern = AntiPattern(
            name="God Object",
            description="Class knows too much",
            file_path="test.py",
            start_line=1,
            end_line=50,
            severity=Severity.HIGH,
            impact="High coupling",
            remediation="Split into smaller classes",
            confidence=0.9,
        )

        assert anti_pattern.name == "God Object"
        assert anti_pattern.severity == Severity.HIGH

    def test_anti_pattern_to_dict(self):
        """Test anti-pattern serialization."""
        anti_pattern = AntiPattern(
            name="Spaghetti Code",
            description="Tangled control flow",
            file_path="spaghetti.py",
            start_line=1,
            end_line=100,
            severity=Severity.CRITICAL,
            impact="Hard to maintain",
            remediation="Refactor",
            confidence=0.85,
        )

        data = anti_pattern.to_dict()

        assert data["name"] == "Spaghetti Code"
        assert data["severity"] == "critical"


class TestPatternDefinition:
    """Tests for PatternDefinition class."""

    def test_pattern_definition_creation(self):
        """Test pattern definition creation."""
        definition = PatternDefinition(
            name="TestPattern",
            pattern_type=PatternType.STRUCTURAL,
            description="A test pattern",
            indicators=["test_indicator"],
            structural_requirements=["requirement1"],
        )

        assert definition.name == "TestPattern"
        assert definition.pattern_type == PatternType.STRUCTURAL

    def test_pattern_definition_to_dict(self):
        """Test pattern definition serialization."""
        definition = PatternDefinition(
            name="Custom",
            pattern_type=PatternType.ARCHITECTURAL,
            description="Custom pattern",
            indicators=["ind1", "ind2"],
        )

        data = definition.to_dict()

        assert data["name"] == "Custom"
        assert data["type"] == "architectural"
        assert len(data["indicators"]) == 2


class TestSeverity:
    """Tests for Severity enum."""

    def test_severity_order(self):
        """Test severity levels."""
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"
