"""Tests for the agent registry module."""

import os
import tempfile
from pathlib import Path

from src.agent_registry import SubAgentRegistry, SubAgentSpec


class TestSubAgentSpec:
    """Tests for SubAgentSpec class."""

    def test_from_dict(self):
        """Test creating SubAgentSpec from dictionary."""
        data = {
            "name": "test_agent",
            "description": "A test agent",
            "system_prompt": "You are a test agent.",
            "tools": ["tool1", "tool2"],
            "model": "gpt-4",
            "tags": ["test", "example"],
        }

        spec = SubAgentSpec.from_dict(data)

        assert spec.name == "test_agent"
        assert spec.description == "A test agent"
        assert spec.system_prompt == "You are a test agent."
        assert spec.tools == ["tool1", "tool2"]
        assert spec.model == "gpt-4"
        assert spec.tags == ["test", "example"]

    def test_to_dict(self):
        """Test converting SubAgentSpec to dictionary."""
        spec = SubAgentSpec(
            name="test_agent",
            description="A test agent",
            system_prompt="You are a test agent.",
            tools=["tool1"],
            model="claude-3",
        )

        data = spec.to_dict()

        assert data["name"] == "test_agent"
        assert data["description"] == "A test agent"
        assert data["system_prompt"] == "You are a test agent."
        assert data["tools"] == ["tool1"]
        assert data["model"] == "claude-3"

    def test_default_values(self):
        """Test default values for optional fields."""
        spec = SubAgentSpec(
            name="minimal_agent",
            description="A minimal agent",
            system_prompt="You are minimal.",
        )

        assert spec.tools == []
        assert spec.model is None
        assert spec.tags == []


class TestSubAgentRegistry:
    """Tests for SubAgentRegistry class."""

    def test_register(self):
        """Test registering a sub-agent."""
        registry = SubAgentRegistry()

        spec = SubAgentSpec(
            name="test_agent",
            description="Test agent",
            system_prompt="You are a test agent.",
        )

        result = registry.register(spec)

        assert result.name == "test_agent"
        assert "test_agent" in registry
        assert len(registry) == 1

    def test_unregister(self):
        """Test unregistering a sub-agent."""
        registry = SubAgentRegistry()

        spec = SubAgentSpec(
            name="test_agent",
            description="Test agent",
            system_prompt="You are a test agent.",
        )
        registry.register(spec)

        removed = registry.unregister("test_agent")

        assert removed is True
        assert "test_agent" not in registry
        assert len(registry) == 0

    def test_unregister_not_found(self):
        """Test unregistering a non-existent agent."""
        registry = SubAgentRegistry()

        removed = registry.unregister("nonexistent")

        assert removed is False

    def test_get(self):
        """Test getting a sub-agent by name."""
        registry = SubAgentRegistry()

        spec = SubAgentSpec(
            name="test_agent",
            description="Test agent",
            system_prompt="You are a test agent.",
        )
        registry.register(spec)

        result = registry.get("test_agent")

        assert result is not None
        assert result.name == "test_agent"

    def test_get_not_found(self):
        """Test getting a non-existent sub-agent."""
        registry = SubAgentRegistry()

        result = registry.get("nonexistent")

        assert result is None

    def test_list(self):
        """Test listing all sub-agents."""
        registry = SubAgentRegistry()

        registry.register(
            SubAgentSpec(
                name="agent1", description="Agent 1", system_prompt="Prompt 1."
            )
        )
        registry.register(
            SubAgentSpec(
                name="agent2", description="Agent 2", system_prompt="Prompt 2."
            )
        )

        agents = registry.list()

        assert len(agents) == 2
        assert agents[0].name in ["agent1", "agent2"]
        assert agents[1].name in ["agent1", "agent2"]

    def test_list_names(self):
        """Test listing all sub-agent names."""
        registry = SubAgentRegistry()

        registry.register(
            SubAgentSpec(
                name="agent1", description="Agent 1", system_prompt="Prompt 1."
            )
        )
        registry.register(
            SubAgentSpec(
                name="agent2", description="Agent 2", system_prompt="Prompt 2."
            )
        )

        names = registry.list_names()

        assert len(names) == 2
        assert "agent1" in names
        assert "agent2" in names

    def test_save_and_load_file(self):
        """Test saving and loading a YAML file."""
        registry = SubAgentRegistry()

        spec = SubAgentSpec(
            name="save_test",
            description="Save test agent",
            system_prompt="You are a save test agent.",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_agent.yaml")
            registry.save_spec(spec, path)

            assert os.path.exists(path)

            loaded = registry.load_file(path)

            assert loaded.name == "save_test"
            assert loaded.description == "Save test agent"

    def test_load_directory(self):
        """Test loading all YAML files from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_dir = Path(tmpdir) / "agents"
            agent_dir.mkdir()

            yaml1 = agent_dir / "agent1.yaml"
            yaml1.write_text("""
name: agent1
description: Agent 1
system_prompt: You are agent 1.
""")

            yaml2 = agent_dir / "agent2.yaml"
            yaml2.write_text("""
name: agent2
description: Agent 2
system_prompt: You are agent 2.
""")

            registry = SubAgentRegistry()
            count = registry.load_directory(str(agent_dir))

            assert count == 2
            assert "agent1" in registry
            assert "agent2" in registry

    def test_clear(self):
        """Test clearing all sub-agents."""
        registry = SubAgentRegistry()

        registry.register(
            SubAgentSpec(
                name="agent1", description="Agent 1", system_prompt="Prompt 1."
            )
        )
        registry.register(
            SubAgentSpec(
                name="agent2", description="Agent 2", system_prompt="Prompt 2."
            )
        )

        registry.clear()

        assert len(registry) == 0
        assert "agent1" not in registry
        assert "agent2" not in registry

    def test_iterator(self):
        """Test iterating over registry."""
        registry = SubAgentRegistry()

        registry.register(
            SubAgentSpec(
                name="agent1", description="Agent 1", system_prompt="Prompt 1."
            )
        )
        registry.register(
            SubAgentSpec(
                name="agent2", description="Agent 2", system_prompt="Prompt 2."
            )
        )

        names = list(registry)

        assert len(names) == 2
        assert "agent1" in names
        assert "agent2" in names
