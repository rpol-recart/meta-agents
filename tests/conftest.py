"""Configuration for test fixtures."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def sample_agent_spec():
    """Provide a sample agent spec."""
    return {
        "name": "sample_agent",
        "description": "A sample agent for testing",
        "system_prompt": "You are a sample agent.",
        "tools": ["tool1", "tool2"],
    }


@pytest.fixture
def sample_task():
    """Provide a sample task."""
    return "Test task for the orchestrator"
