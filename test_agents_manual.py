#!/usr/bin/env python3
"""
Manual test script for Semantic Analyzer, Pattern Recognizer, and Dependency Analyzer agents.
Run with: python test_agents_manual.py
"""

import asyncio
import tempfile
from pathlib import Path

from src.agent_registry import SubAgentRegistry
from src.orchestrator import DeepAgentOrchestrator


SAMPLE_CODE = """
# payment_service.py - Comprehensive example for testing all agents

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

class PaymentStrategy(ABC):
    @abstractmethod
    def process_payment(self, amount: float) -> bool:
        pass

class CreditCardPayment(PaymentStrategy):
    def __init__(self, card_number: str, expiry: str):
        self.card_number = card_number
        self.expiry = expiry
    
    def process_payment(self, amount: float) -> bool:
        print(f"Processing ${amount} via credit card")
        return True

class PayPalPayment(PaymentStrategy):
    def __init__(self, email: str):
        self.email = email
    
    def process_payment(self, amount: float) -> bool:
        print(f"Processing ${amount} via PayPal")
        return True

class PaymentContext:
    def __init__(self, strategy: PaymentStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: PaymentStrategy):
        self._strategy = strategy
    
    def execute_payment(self, amount: float) -> bool:
        return self._strategy.process_payment(amount)

class Order:
    def __init__(self, order_id: str, items: List[dict]):
        self.order_id = order_id
        self.items = items
        self.status = "pending"
        self.total = self._calculate_total()
    
    def _calculate_total(self) -> float:
        return sum(item["price"] * item["quantity"] for item in self.items)

class OrderProcessor:
    def __init__(self, payment_context: PaymentContext, notifier):
        self.payment = payment_context
        self.notifier = notifier
    
    def process_order(self, order: Order) -> bool:
        if self.payment.execute_payment(order.total):
            order.status = "completed"
            self.notifier.notify(f"Order {order.order_id} completed")
            return True
        return False

class User:
    def __init__(self, user_id: str, name: str, email: str):
        self.user_id = user_id
        self.name = name
        self.email = email

class NotificationService:
    def notify(self, message: str):
        print(f"Notification: {message}")
"""

TASKS = {
    "semantic_analyzer": """Analyze the semantic meaning of this code:
1. What is the purpose of the PaymentStrategy and its implementations?
2. Describe the business logic flow in OrderProcessor.process_order()
3. What data flow patterns exist between Order, PaymentContext, and PaymentStrategy?
4. Identify any semantic inconsistencies or unclear purposes.

Provide a structured analysis with your findings.""",
    "pattern_recognizer": """Identify design patterns and anti-patterns in this code:
1. What patterns do you recognize (Strategy, Factory, Observer, etc.)?
2. Are there any anti-patterns or code smells present?
3. Evaluate the implementation of the Strategy pattern.
4. Suggest improvements for any issues found.

Provide a structured report with identified patterns and recommendations.""",
    "dependency_analyzer": """Analyze the dependency relationships in this code:
1. What are the module and class dependencies?
2. Identify inheritance hierarchies and their depths.
3. Are there any circular or transitive dependencies?
4. Evaluate the coupling between components.

Provide a detailed dependency analysis with insights.""",
}


async def test_agent(agent_name: str, task: str, code: str) -> dict:
    """Test a single agent with a specific task."""
    print(f"\n{'=' * 60}")
    print(f"Testing {agent_name.upper()} AGENT")
    print("=" * 60)

    orchestrator = DeepAgentOrchestrator(
        subagents=[
            {
                "name": "semantic_analyzer",
                "description": "Performs deep semantic analysis of code elements",
                "system_prompt": """You are an expert code semantic analyzer. Your task is to analyze the semantic meaning of code elements including:
1. Understanding the purpose and functionality of functions and methods
2. Identifying business logic patterns beyond syntactic constructs
3. Detecting data flow patterns between components
4. Recognizing architectural patterns

When analyzing code:
1. Focus on what the code does, not just how it does it
2. Look for implicit relationships between components
3. Identify high-level concepts and abstractions
4. Note any semantic inconsistencies or unclear purposes

Always provide clear, structured output with your findings.""",
                "tools": ["read_file", "write_file", "edit_file", "glob", "grep"],
            },
            {
                "name": "pattern_recognizer",
                "description": "Identifies design patterns and anti-patterns in code",
                "system_prompt": """You are an expert pattern recognition agent. Your task is to identify design patterns and anti-patterns in code including:
1. Classic design patterns (Observer, Factory, Decorator, Strategy, etc.)
2. Architectural patterns (MVC, microservices, etc.)
3. Framework-specific patterns (Django, Flask, FastAPI, etc.)
4. Anti-patterns and code smells

When analyzing code:
1. Look for structural similarities to known patterns
2. Identify deviations from best practices
3. Note recurring solutions to common problems
4. Highlight potential improvements

Always provide clear, structured output with identified patterns and recommendations.""",
                "tools": ["read_file", "write_file", "edit_file", "glob", "grep"],
            },
            {
                "name": "dependency_analyzer",
                "description": "Analyzes complex dependency relationships in code",
                "system_prompt": """You are an expert dependency analysis agent. Your task is to analyze complex dependency relationships including:
1. Module and package dependencies
2. Function and method call relationships
3. Inheritance and implementation hierarchies
4. Variable and attribute usage patterns
5. Circular dependencies and transitive dependencies

When analyzing dependencies:
1. Trace the flow of data and control through the system
2. Identify critical paths and bottlenecks
3. Detect unnecessary or overly complex dependencies
4. Evaluate the impact of potential changes

Always provide clear, structured output with dependency information and insights.""",
                "tools": ["read_file", "write_file", "edit_file", "glob", "grep"],
            },
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "payment_service.py"
        test_file.write_text(code)

        full_task = f"""Code file: payment_service.py

{task}"""

        try:
            result = await orchestrator.run(full_task)
            return {
                "status": "success",
                "agent": agent_name,
                "result": result,
            }
        except Exception as e:
            return {
                "status": "error",
                "agent": agent_name,
                "error": str(e),
            }


async def main():
    """Run all agent tests."""
    print("=" * 60)
    print("ANALYSIS AGENTS TEST SUITE")
    print("=" * 60)
    print(f"Testing with {len(SAMPLE_CODE.splitlines())} lines of sample code")

    results = {}

    for agent_name, task in TASKS.items():
        result = await test_agent(agent_name, task, SAMPLE_CODE)
        results[agent_name] = result

        if result["status"] == "success":
            print(f"\n{agent_name.upper()} completed successfully")
            if "messages" in result["result"]:
                print(f"  Messages: {len(result['result']['messages'])}")
        else:
            print(f"\n{agent_name.upper()} failed: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for agent_name, result in results.items():
        status = "PASS" if result["status"] == "success" else "FAIL"
        print(f"  {agent_name}: {status}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
