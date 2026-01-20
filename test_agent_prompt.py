#!/usr/bin/env python3
"""
Test script for demonstrating new agent capabilities.

Usage:
    python test_agent_prompt.py --agent semantic_analyzer --prompt "Analyze this code..."
    python test_agent_prompt.py --agent pattern_recognizer --prompt "Find patterns in this code..."
    python test_agent_prompt.py --agent dependency_analyzer --prompt "Analyze dependencies in this code..."
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.orchestration.coordinator import AgentOrchestrator
from src.orchestrator import DeepAgentOrchestrator
from src.tools import get_analysis_tools, get_pattern_tools, get_dependency_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_CODE = """
import os
import json
from datetime import datetime
from typing import List, Dict, Optional

class Singleton:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.data = []
        self.users = []
        self.config = {}

class DataProcessor:
    def __init__(self, name: str):
        self.name = name
        self.cache = {}
    
    def process(self, items: List[str]) -> List[Dict]:
        result = []
        for item in items:
            result.append({"item": item, "processed": True})
        return result
    
    def save_to_file(self, filename: str, data: str):
        with open(filename, 'w') as f:
            f.write(json.dumps(data))
        return True

class UserManager:
    def __init__(self):
        self.processor = DataProcessor("user")
    
    def get_users(self) -> List[str]:
        return ["alice", "bob", "charlie"]
"""

SAMPLE_TEXT = """
Apple Inc. announced a breakthrough in AI technology today.
The new system uses advanced machine learning algorithms for natural language processing.
This innovative development promises to transform industries significantly.
Google and Microsoft are also investing heavily in similar technologies.
The extrusion process involves heat treatment and forging of aluminum materials.
"""

AGENT_PROMPTS = {
    "semantic_analyzer": """You are a semantic analyzer. Analyze the following code and text.

Focus on:
1. Extracting named entities (technologies, companies, materials, processes)
2. Analyzing sentiment (positive, negative, neutral)
3. Identifying strategic implications

Provide your findings in a structured format.

=== CODE AND TEXT ===
{input}

=== ANALYSIS ===
""",
    "pattern_recognizer": """You are a pattern recognition agent. Analyze the following code.

Focus on:
1. Detecting design patterns (Singleton, Factory, Observer, etc.)
2. Identifying anti-patterns (God Object, Spaghetti Code, Magic Numbers)
3. Calculating code metrics (complexity, lines, etc.)

Provide your findings in a structured format.

=== CODE ===
{input}

=== ANALYSIS ===
""",
    "dependency_analyzer": """You are a dependency analysis agent. Analyze the following code.

Focus on:
1. Finding all imports (standard lib, third-party, local)
2. Analyzing file dependencies
3. Building dependency graph concepts
4. Detecting potential circular dependencies

Provide your findings in a structured format.

=== CODE ===
{input}

=== ANALYSIS ===
""",
}


async def run_agent_test(agent_name: str, input_text: str, verbose: bool = False):
    """Run an agent with a specific prompt."""
    logger.info(f"Starting test for agent: {agent_name}")

    orchestrator = DeepAgentOrchestrator(
        model_name="anthropic:claude-sonnet-4-20250514",
    )

    coordinator = AgentOrchestrator(orchestrator)

    yaml_files = {
        "semantic_analyzer": "/home/meta_agent/agents/semantic_analyzer.yaml",
        "pattern_recognizer": "/home/meta_agent/agents/pattern_recognizer.yaml",
        "dependency_analyzer": "/home/meta_agent/agents/dependency_analyzer.yaml",
    }

    if agent_name not in yaml_files:
        logger.error(f"Unknown agent: {agent_name}")
        return None

    yaml_path = yaml_files[agent_name]
    if not Path(yaml_path).exists():
        logger.error(f"Agent YAML not found: {yaml_path}")
        return None

    from src.agent_registry import SubAgentSpec

    spec = SubAgentSpec.from_dict(
        {
            "name": agent_name,
            "description": f"Test {agent_name} agent",
            "system_prompt": AGENT_PROMPTS[agent_name],
            "tools": [],
        }
    )
    coordinator.registry.register(spec)
    coordinator.orchestrator.add_subagent(
        name=agent_name,
        description=spec.description,
        system_prompt=spec.system_prompt,
        tools=spec.tools,
    )

    prompt = AGENT_PROMPTS[agent_name].format(input=input_text)

    logger.info(f"Running {agent_name} agent...")
    logger.info(f"Input length: {len(input_text)} characters")

    result = await coordinator.orchestrator.run(
        task=prompt,
        thread_id=f"test_{agent_name}",
    )

    if verbose:
        logger.info(f"\n=== Result from {agent_name} ===")
        logger.info(f"Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test new agent capabilities with custom prompts")
    parser.add_argument(
        "--agent",
        choices=["semantic_analyzer", "pattern_recognizer", "dependency_analyzer"],
        default="semantic_analyzer",
        help="Which agent to test (default: semantic_analyzer)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt to send to the agent",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input text/code to analyze (uses sample if not provided)",
    )
    parser.add_argument(
        "--code",
        action="store_true",
        help="Use sample code instead of sample text",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save result to file",
    )

    args = parser.parse_args()

    if args.code:
        input_text = SAMPLE_CODE
    elif args.input:
        input_text = args.input
    elif args.prompt:
        input_text = args.prompt
    else:
        input_text = SAMPLE_TEXT

    print(f"\n{'=' * 60}")
    print(f"Testing Agent: {args.agent}")
    print(f"{'=' * 60}")
    print(f"Input:\n{input_text[:200]}..." if len(input_text) > 200 else f"Input:\n{input_text}")
    print(f"{'=' * 60}\n")

    result = asyncio.run(run_agent_test(args.agent, input_text, args.verbose))

    if result:
        if isinstance(result, dict):
            print("\n=== Agent Response ===")
            if "messages" in result:
                for msg in result["messages"][-3:]:
                    if hasattr(msg, "content"):
                        print(msg.content)
            elif "result" in result:
                print(result["result"])
            else:
                for key, value in result.items():
                    print(f"\n{key}: {value}")

        if args.output:
            import json

            Path(args.output).write_text(json.dumps(result, indent=2))
            print(f"\nResult saved to: {args.output}")

    print(f"\n{'=' * 60}")
    print("Test completed!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
