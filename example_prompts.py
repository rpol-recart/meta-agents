"""
Example prompts for testing agent capabilities with new tools.

Usage:
    # Run with orchestrate CLI
    orchestrate run "<prompt>" --agents-dir agents/

    # Or use the API
    curl -X POST http://localhost:8000/api/run \
        -H "Content-Type: application/json" \
        -d '{"agent": "semantic_analyzer", "task": "<prompt>"}'
"""

PROMpts = {
    "semantic_analyzer": [
        # Basic entity extraction
        """
        Analyze the following Python code and extract named entities:
        - Technologies and frameworks (AI, ML, React, etc.)
        - Companies and organizations
        - Materials and technologies
        - Processes and methodologies

        Code:
        ```python
        import torch
        import transformers
        from langchain import OpenAI
        from flask import Flask

        class DocumentProcessor:
            def __init__(self, model_name="gpt-4"):
                self.model = OpenAI(model=model_name)
                self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base")
                
            def process(self, text):
                # Extract entities using NLP
                entities = self.model.extract_entities(text)
                return entities
        ```

        Return the extracted entities in JSON format.
        """,
        # Sentiment and strategic analysis
        """
        Analyze the sentiment and strategic implications of this technology description:

        "Apple Inc. announced a breakthrough in AI technology using advanced machine learning algorithms.
         The new system promises to revolutionize healthcare by improving diagnostic accuracy by 40%.
         However, the implementation faces challenges with data privacy and regulatory compliance.
         This development could significantly impact the competitive landscape of medical AI."

        Provide:
        1. Sentiment analysis (positive/negative/neutral)
        2. Strategic implications
        3. Detected intents (announcement, research, deployment, etc.)
        """,
    ],
    "pattern_recognizer": [
        # Pattern detection
        """
        Detect design patterns in the following code:

        ```python
        class Singleton:
            _instance = None
            
            @classmethod
            def get_instance(cls):
                if cls._instance is None:
                    cls._instance = cls()
                return cls._instance

        class Product:
            def __init__(self, name, price):
                self.name = name
                self.price = price

        class Creator:
            def factory_method(self):
                return Product("Default Product", 9.99)

        class Subject:
            def __init__(self):
                self._observers = []
                
            def attach(self, observer):
                self._observers.append(observer)
                
            def notify(self):
                for observer in self._observers:
                    observer.update()
        ```

        Return all detected patterns with confidence scores.
        """,
        # Anti-pattern detection
        """
        Analyze the following code for anti-patterns and code smells:

        ```python
        class EverythingManager:
            def __init__(self):
                self.data = []
                self.users = []
                self.config = {}
                self.connections = []
                self.cache = []
                self.logger = []
                
            def process_all_data(self, x=100, y=50, z=25):
                result = x * y * z + 1000 - 500 + 200
                for item in self.data:
                    if item.status == 1:
                        self.validate(item)
                        self.transform(item)
                        self.save(item)
                        self.log(item)
                        self.cache.append(item)
                    elif item.status == 2:
                        self.validate(item)
                        self.update(item)
                        self.delete(item)
                    else:
                        self.ignore(item)
                        
            def validate(self, item): pass
            def transform(self, item): pass
            def save(self, item): pass
            def log(self, item): pass
            def update(self, item): pass
            def delete(self, item): pass
            def ignore(self, item): pass
        ```

        Identify all anti-patterns with severity levels.
        """,
        # Full analysis with metrics
        """
        Perform a complete pattern analysis of this code including:
        1. All design patterns
        2. Anti-patterns and code smells
        3. Code quality metrics (complexity, lines, etc.)

        ```python
        from abc import ABC, abstractmethod

        class Strategy(ABC):
            @abstractmethod
            def execute(self, data):
                pass

        class FastStrategy(Strategy):
            def execute(self, data):
                return [x * 2 for x in data]

        class SlowStrategy(Strategy):
            def execute(self, data):
                return [x ** 2 for x in data]

        class Context:
            def __init__(self, strategy: Strategy):
                self._strategy = strategy
                
            def set_strategy(self, strategy: Strategy):
                self._strategy = strategy
                
            def do_work(self, data):
                return self._strategy.execute(data)
        ```
        """,
    ],
    "dependency_analyzer": [
        # Import analysis
        """
        Extract all imports from this codebase and categorize them:

        ```python
        # Standard library
        import os
        import sys
        from datetime import datetime
        import json
        import re

        # Third party
        import requests
        from flask import Flask, jsonify
        from langchain import OpenAI, Chain
        import numpy as np
        import pandas as pd

        # Local
        from .utils import helper_function
        from .models import User, Product, Order
        from ..services import PaymentService
        ```

        Return categorized imports with counts.
        """,
        # Dependency graph analysis
        """
        Analyze dependencies between these files and detect any circular dependencies:

        File: main.py
        ```
        from module_a import ClassA
        from utils import helper
        ```

        File: module_a.py
        ```
        from main import main_function
        from module_b import ClassB
        ```

        File: module_b.py
        ```
        from module_a import ClassA
        from utils import process_data
        ```

        File: utils.py
        ```
        from module_b import validate
        ```

        Build the dependency graph and report findings.
        """,
        # Full dependency analysis
        """
        Perform a comprehensive dependency analysis on this project structure:

        ```python
        # main.py
        import os
        from flask import Flask
        from routes.api import api_bp
        from services.auth import AuthService
        from models.user import User

        # routes/api.py
        from flask import Blueprint
        from services.auth import require_auth
        from models.user import User

        # services/auth.py
        from models.user import User
        from utils.jwt_handler import create_token
        import requests

        # models/user.py
        from datetime import datetime
        import uuid
        ```

        Provide:
        1. Dependency graph (nodes and edges)
        2. Circular dependency detection
        3. Dependency metrics (total files, imports, most used modules)
        """,
    ],
    "cross_agent": [
        # Multi-agent analysis
        """
        Perform a comprehensive analysis of this code using multiple approaches:

        ```python
        class DataProcessor:
            _instance = None
            
            @classmethod
            def get_instance(cls):
                if cls._instance is None:
                    cls._instance = cls()
                return cls._instance
                
            def __init__(self):
                self.data = []
                self.model = None
                self.cache = {}
                
            def process(self, data):
                import pandas as pd
                import numpy as np
                from sklearn.preprocessing import StandardScaler
                
                df = pd.DataFrame(data)
                scaler = StandardScaler()
                return scaler.fit_transform(df)
        ```

        Analyze:
        1. Design patterns (pattern_recognizer)
        2. Dependencies and imports (dependency_analyzer)
        3. Entities and sentiment (semantic_analyzer)
        """,
    ],
}


def get_prompt(agent_name: str, index: int = 0) -> str:
    """Get a prompt for testing."""
    prompts = PROMPts.get(agent_name, [])
    if index < len(prompts):
        return prompts[index]
    return prompts[0] if prompts else "Analyze this code."


if __name__ == "__main__":
    import sys

    agent = sys.argv[1] if len(sys.argv) > 1 else "semantic_analyzer"
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    prompt = get_prompt(agent, index)

    print(f"=== Prompt for {agent} (index {index}) ===\n")
    print(prompt)
    print("\n=== End of Prompt ===")
