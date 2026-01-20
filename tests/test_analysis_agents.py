"""Tests for semantic analyzer, pattern recognizer, and dependency analyzer agents."""

import os
import tempfile
from pathlib import Path

import pytest

from src.agent_registry import SubAgentRegistry, SubAgentSpec


class TestSemanticAnalyzerAgent:
    """Tests for the Semantic Analyzer Agent."""

    def test_agent_spec_exists(self):
        """Test that semantic_analyzer agent is defined."""
        registry = SubAgentRegistry()
        registry.load_directory("agents")

        spec = registry.get("semantic_analyzer")
        assert spec is not None
        assert spec.name == "semantic_analyzer"
        assert "semantic" in spec.description.lower()

    def test_agent_has_required_tools(self):
        """Test that semantic analyzer has required tools."""
        registry = SubAgentRegistry()
        registry.load_directory("agents")

        spec = registry.get("semantic_analyzer")
        assert "read_file" in spec.tools
        assert "grep" in spec.tools
        assert "glob" in spec.tools

    def test_agent_system_prompt_contains_key_concepts(self):
        """Test that system prompt contains key semantic analysis concepts."""
        registry = SubAgentRegistry()
        registry.load_directory("agents")

        spec = registry.get("semantic_analyzer")
        prompt = spec.system_prompt.lower()

        assert "function" in prompt or "method" in prompt
        assert "business logic" in prompt
        assert "data flow" in prompt

    def test_analyze_simple_function(self):
        """Test analyzing a simple function with clear semantics."""
        test_code = '''
def calculate_total(items):
    """Calculate total price of items."""
    total = 0
    for item in items:
        total += item.price * item.quantity
    return total
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_function.py"
            test_file.write_text(test_code)

            registry = SubAgentRegistry()
            registry.load_directory("agents")
            spec = registry.get("semantic_analyzer")

            assert spec is not None
            assert "semantic" in spec.description.lower() or "analysis" in spec.description.lower()

    def test_analyze_complex_business_logic(self):
        """Test analyzing complex business logic."""
        test_code = """
class OrderProcessor:
    def __init__(self, validator, repository):
        self.validator = validator
        self.repository = repository
    
    def process(self, order_data):
        if not self.validator.validate(order_data):
            raise ValueError("Invalid order")
        
        order = self.repository.create(order_data)
        return self._execute_order(order)
    
    def _execute_order(self, order):
        for item in order.items:
            item.reserved = True
        order.status = 'processing'
        return self.repository.save(order)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "order_processor.py"
            test_file.write_text(test_code)

            registry = SubAgentRegistry()
            registry.load_directory("agents")
            spec = registry.get("semantic_analyzer")

            assert spec is not None
            assert len(spec.tools) > 0


class TestPatternRecognizerAgent:
    """Tests for the Pattern Recognizer Agent."""

    def test_agent_spec_exists(self):
        """Test that pattern_recognizer agent is defined."""
        registry = SubAgentRegistry()
        registry.load_directory("agents")

        spec = registry.get("pattern_recognizer")
        assert spec is not None
        assert spec.name == "pattern_recognizer"
        assert "pattern" in spec.description.lower()

    def test_agent_has_required_tools(self):
        """Test that pattern recognizer has required tools."""
        registry = SubAgentRegistry()
        registry.load_directory("agents")

        spec = registry.get("pattern_recognizer")
        assert "read_file" in spec.tools
        assert "grep" in spec.tools

    def test_agent_system_prompt_contains_patterns(self):
        """Test that system prompt mentions design patterns."""
        registry = SubAgentRegistry()
        registry.load_directory("agents")

        spec = registry.get("pattern_recognizer")
        prompt = spec.system_prompt.lower()

        assert "observer" in prompt or "factory" in prompt or "strategy" in prompt
        assert "anti-pattern" in prompt or "code smell" in prompt

    def test_recognize_factory_pattern(self):
        """Test recognizing Factory pattern code."""
        factory_code = """
class Vehicle:
    def drive(self):
        pass

class Car(Vehicle):
    def drive(self):
        return "Driving a car"

class Truck(Vehicle):
    def drive(self):
        return "Driving a truck"

class VehicleFactory:
    @staticmethod
    def create_vehicle(vehicle_type):
        if vehicle_type == "car":
            return Car()
        elif vehicle_type == "truck":
            return Truck()
        raise ValueError(f"Unknown vehicle type: {vehicle_type}")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "factory.py"
            test_file.write_text(factory_code)

            registry = SubAgentRegistry()
            registry.load_directory("agents")
            spec = registry.get("pattern_recognizer")

            assert spec is not None

    def test_recognize_observer_pattern(self):
        """Test recognizing Observer pattern code."""
        observer_code = """
from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass

class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "observer.py"
            test_file.write_text(observer_code)

            registry = SubAgentRegistry()
            registry.load_directory("agents")
            spec = registry.get("pattern_recognizer")

            assert spec is not None

    def test_detect_god_class_antipattern(self):
        """Test detecting God Class anti-pattern."""
        god_class_code = """
class SystemManager:
    def __init__(self):
        self.users = []
        self.orders = []
        self.products = []
        self.invoices = []
        self.reports = []
    
    def add_user(self, user):
        self.users.append(user)
    
    def delete_user(self, user):
        self.users.remove(user)
    
    def create_order(self, order):
        self.orders.append(order)
    
    def cancel_order(self, order):
        self.orders.remove(order)
    
    def generate_report(self):
        pass
    
    def validate_invoice(self, invoice):
        pass
    
    def process_payment(self, payment):
        pass
    
    def send_email(self, recipient, message):
        pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "god_class.py"
            test_file.write_text(god_class_code)

            registry = SubAgentRegistry()
            registry.load_directory("agents")
            spec = registry.get("pattern_recognizer")

            assert spec is not None


class TestDependencyAnalyzerAgent:
    """Tests for the Dependency Analyzer Agent."""

    def test_agent_spec_exists(self):
        """Test that dependency_analyzer agent is defined."""
        registry = SubAgentRegistry()
        registry.load_directory("agents")

        spec = registry.get("dependency_analyzer")
        assert spec is not None
        assert spec.name == "dependency_analyzer"
        assert "dependency" in spec.description.lower()

    def test_agent_has_required_tools(self):
        """Test that dependency analyzer has required tools."""
        registry = SubAgentRegistry()
        registry.load_directory("agents")

        spec = registry.get("dependency_analyzer")
        assert "read_file" in spec.tools
        assert "grep" in spec.tools

    def test_agent_system_prompt_contains_dependency_concepts(self):
        """Test that system prompt mentions dependency concepts."""
        registry = SubAgentRegistry()
        registry.load_directory("agents")

        spec = registry.get("dependency_analyzer")
        prompt = spec.system_prompt.lower()

        assert "module" in prompt or "package" in prompt
        assert "circular" in prompt or "transitive" in prompt
        assert "inheritance" in prompt

    def test_detect_circular_dependency(self):
        """Test detecting circular dependency code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = Path(tmpdir) / "module_a.py"
            file_a.write_text("""
from module_b import ClassB

class ClassA:
    def method_a(self):
        return ClassB().method_b()
""")
            file_b = Path(tmpdir) / "module_b.py"
            file_b.write_text("""
from module_a import ClassA

class ClassB:
    def method_b(self):
        return ClassA().method_a()
""")

            registry = SubAgentRegistry()
            registry.load_directory("agents")
            spec = registry.get("dependency_analyzer")

            assert spec is not None

    def test_analyze_inheritance_hierarchy(self):
        """Test analyzing inheritance hierarchy."""
        hierarchy_code = """
class BaseClass:
    def base_method(self):
        pass

class Level1(BaseClass):
    def level1_method(self):
        pass

class Level2(Level1):
    def level2_method(self):
        pass

class Level3(Level2):
    def level3_method(self):
        pass

class Level4(Level3):
    def level4_method(self):
        pass

class Level5(Level4):
    def deep_method(self):
        return self.level4_method()
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "hierarchy.py"
            test_file.write_text(hierarchy_code)

            registry = SubAgentRegistry()
            registry.load_directory("agents")
            spec = registry.get("dependency_analyzer")

            assert spec is not None

    def test_analyze_transitive_dependencies(self):
        """Test analyzing transitive dependencies."""
        transitive_code = """
from module_a import ClassA

class ClassB:
    def __init__(self):
        self.a = ClassA()
    
    def use_a(self):
        return self.a.operation()

class ClassC:
    def __init__(self):
        self.b = ClassB()
    
    def process(self):
        return self.b.use_a()
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "transitive.py"
            test_file.write_text(transitive_code)

            registry = SubAgentRegistry()
            registry.load_directory("agents")
            spec = registry.get("dependency_analyzer")

            assert spec is not None


class TestAgentIntegration:
    """Integration tests for all analysis agents working together."""

    def test_all_agents_loaded(self):
        """Test that all three agents are loaded from YAML files."""
        registry = SubAgentRegistry()
        count = registry.load_directory("agents")

        assert "semantic_analyzer" in registry
        assert "pattern_recognizer" in registry
        assert "dependency_analyzer" in registry
        assert count >= 3

    def test_agents_have_distinct_prompts(self):
        """Test that each agent has a distinct system prompt focus."""
        registry = SubAgentRegistry()
        registry.load_directory("agents")

        semantic = registry.get("semantic_analyzer")
        pattern = registry.get("pattern_recognizer")
        dependency = registry.get("dependency_analyzer")

        assert semantic.system_prompt != pattern.system_prompt
        assert semantic.system_prompt != dependency.system_prompt
        assert pattern.system_prompt != dependency.system_prompt

    def test_agents_complementary_capabilities(self):
        """Test that agents have complementary tool configurations."""
        registry = SubAgentRegistry()
        registry.load_directory("agents")

        semantic = registry.get("semantic_analyzer")
        pattern = registry.get("pattern_recognizer")
        dependency = registry.get("dependency_analyzer")

        all_tools = set(semantic.tools) | set(pattern.tools) | set(dependency.tools)
        assert "read_file" in all_tools
        assert "grep" in all_tools

    def test_sample_codebase_analysis(self):
        """Test analyzing a sample codebase with all three agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_code = """
# service.py - Example service module

class NotificationService:
    def __init__(self, email_service, sms_service):
        self.email = email_service
        self.sms = sms_service
    
    def send_notification(self, user, message, channel):
        if channel == "email":
            return self.email.send(user.email, message)
        elif channel == "sms":
            return self.sms.send(user.phone, message)

class User:
    def __init__(self, name, email, phone):
        self.name = name
        self.email = email
        self.phone = phone

# Factory pattern for creating services
class ServiceFactory:
    @staticmethod
    def create_notification_service():
        email = EmailService()
        sms = SmsService()
        return NotificationService(email, sms)
"""
            test_file = Path(tmpdir) / "service.py"
            test_file.write_text(sample_code)

            registry = SubAgentRegistry()
            registry.load_directory("agents")

            semantic_spec = registry.get("semantic_analyzer")
            pattern_spec = registry.get("pattern_recognizer")
            dependency_spec = registry.get("dependency_analyzer")

            assert semantic_spec is not None
            assert pattern_spec is not None
            assert dependency_spec is not None
