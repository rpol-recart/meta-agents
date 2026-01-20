"""Tests for the CLI module."""

import pytest
from click.testing import CliRunner

from src.cli import cli


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_version(self, runner):
        """Test version command."""
        result = runner.invoke(cli, ["version"])

        assert result.exit_code == 0
        assert "DeepAgent Orchestrator" in result.output or "v0." in result.output

    def test_list_agents_nonexistent_dir(self, runner):
        """Test list_agents with nonexistent directory."""
        result = runner.invoke(cli, ["list-agents", "/nonexistent"])

        assert result.exit_code != 0
        assert (
            "does not exist" in result.output.lower()
            or "not found" in result.output.lower()
        )

    def test_create_agent_output(self, runner):
        """Test create_agent outputs YAML."""
        result = runner.invoke(
            cli, ["create-agent", "test_agent", "A test agent", "You are a test agent."]
        )

        assert result.exit_code == 0
        assert "test_agent" in result.output
        assert "A test agent" in result.output

    def test_run_no_task(self, runner):
        """Test run with no task."""
        result = runner.invoke(cli, ["run"])

        assert result.exit_code != 0
        assert "task" in result.output.lower()
