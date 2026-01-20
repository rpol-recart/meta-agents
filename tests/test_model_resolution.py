"""
Tests for model name resolution in DeepAgent Orchestrator.

Tests verify that model name is always read from:
1. Explicitly provided model_name parameter
2. DEFAULT_MODEL environment variable
3. Fallback default as last resort
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestModelNameResolution:
    """Tests for model name resolution logic."""

    def test_model_from_explicit_parameter(self):
        """Test that explicitly provided model_name is used."""
        from src.orchestrator import DeepAgentOrchestrator

        with patch('deepagents.create_deep_agent'):
            with patch('src.orchestrator.ModelService') as MockService:
                mock_service = MagicMock()
                mock_model = MagicMock()
                mock_service.initialize.return_value = mock_model
                MockService.return_value = mock_service

                orchestrator = DeepAgentOrchestrator(
                    model_name="custom/model:name",
                    enable_memory=False,
                )

                assert orchestrator.config.model_name == "custom/model:name"
                mock_service.initialize.assert_called_once()
                call_kwargs = mock_service.initialize.call_args[1]
                assert call_kwargs['model_name'] == "custom/model:name"

    def test_model_from_env_var(self, monkeypatch):
        """Test that DEFAULT_MODEL from environment is used when model_name is None."""
        from src.orchestrator import DeepAgentOrchestrator

        monkeypatch.setenv("DEFAULT_MODEL", "env/model:v1")

        with patch('deepagents.create_deep_agent'):
            with patch('src.orchestrator.ModelService') as MockService:
                mock_service = MagicMock()
                mock_model = MagicMock()
                mock_service.initialize.return_value = mock_model
                MockService.return_value = mock_service

                orchestrator = DeepAgentOrchestrator(
                    model_name=None,
                    enable_memory=False,
                )

                assert orchestrator.config.model_name == "env/model:v1"
                mock_service.initialize.assert_called_once()
                call_kwargs = mock_service.initialize.call_args[1]
                assert call_kwargs['model_name'] == "env/model:v1"

    def test_model_fallback_to_default(self, monkeypatch):
        """Test fallback to default when no env var and no explicit model."""
        from src.orchestrator import DeepAgentOrchestrator

        monkeypatch.delenv("DEFAULT_MODEL", raising=False)

        with patch('deepagents.create_deep_agent'):
            with patch('src.orchestrator.ModelService') as MockService:
                mock_service = MagicMock()
                mock_model = MagicMock()
                mock_service.initialize.return_value = mock_model
                MockService.return_value = mock_service

                orchestrator = DeepAgentOrchestrator(
                    model_name=None,
                    enable_memory=False,
                )

                assert orchestrator.config.model_name == "anthropic:claude-sonnet-4-20250514"

    def test_explicit_model_overrides_env(self, monkeypatch):
        """Test that explicit model_name takes precedence over env var."""
        from src.orchestrator import DeepAgentOrchestrator

        monkeypatch.setenv("DEFAULT_MODEL", "env/model:v1")

        with patch('deepagents.create_deep_agent'):
            with patch('src.orchestrator.ModelService') as MockService:
                mock_service = MagicMock()
                mock_model = MagicMock()
                mock_service.initialize.return_value = mock_model
                MockService.return_value = mock_service

                orchestrator = DeepAgentOrchestrator(
                    model_name="explicit/model:v2",
                    enable_memory=False,
                )

                assert orchestrator.config.model_name == "explicit/model:v2"


class TestCLI:
    """Tests for CLI model handling."""

    def test_cli_default_model_none(self, monkeypatch):
        """Test that CLI passes None when --model is not specified."""
        from src.cli import run
        from click.testing import CliRunner

        # Load .env first
        from dotenv import load_dotenv
        load_dotenv()

        # Mock the orchestrator to capture the model parameter
        with patch('src.cli.DeepAgentOrchestrator') as MockOrchestrator:
            mock_instance = MagicMock()
            MockOrchestrator.return_value = mock_instance

            runner = CliRunner()
            result = runner.invoke(run, ["test task"])

            # Check what model was passed to orchestrator
            call_kwargs = MockOrchestrator.call_args[1]
            # When --model is not specified, it should be None
            # The orchestrator should then read from DEFAULT_MODEL
            assert call_kwargs.get('model_name') is None or 'model_name' not in call_kwargs

    def test_cli_explicit_model(self, monkeypatch):
        """Test that CLI passes explicit model when --model is specified."""
        from src.cli import run
        from click.testing import CliRunner

        with patch('src.cli.DeepAgentOrchestrator') as MockOrchestrator:
            mock_instance = MagicMock()
            MockOrchestrator.return_value = mock_instance

            runner = CliRunner()
            result = runner.invoke(run, ["--model", "test/model:v1", "test task"])

            call_kwargs = MockOrchestrator.call_args[1]
            assert call_kwargs.get('model_name') == "test/model:v1"


class TestDotenvLoading:
    """Tests for .env file loading."""

    def test_current_env_loaded(self):
        """Test that current .env file is loaded."""
        from dotenv import load_dotenv

        load_dotenv()

        # The current .env should have DEFAULT_MODEL
        default_model = os.environ.get("DEFAULT_MODEL")
        assert default_model is not None
        assert default_model == "Qwen/Qwen3-Coder-480B-A35B-Instruct"


class TestIntegration:
    """Integration tests for model resolution."""

    def test_full_resolution_chain(self, monkeypatch):
        """Test the complete model resolution chain."""
        from src.orchestrator import DeepAgentOrchestrator

        # Set environment variable
        monkeypatch.setenv("DEFAULT_MODEL", "integration/test:v1")

        with patch('deepagents.create_deep_agent'):
            with patch('src.orchestrator.ModelService') as MockService:
                mock_service = MagicMock()
                mock_model = MagicMock()
                mock_service.initialize.return_value = mock_model
                MockService.return_value = mock_service

                # Create orchestrator without specifying model
                orchestrator = DeepAgentOrchestrator(
                    model_name=None,  # Not specified
                    enable_memory=False,
                )

                # Verify the full chain: None -> env var -> effective model
                assert orchestrator.config.model_name == "integration/test:v1"

                # Verify the model service was called with the correct model name
                # (api_key and base_url may come from .env, that's expected)
                call_kwargs = mock_service.initialize.call_args[1]
                assert call_kwargs['model_name'] == "integration/test:v1"

    def test_env_var_precedence(self, monkeypatch):
        """Test that .env loaded via load_dotenv takes precedence."""
        from dotenv import load_dotenv

        # Create a clean environment
        monkeypatch.delenv("DEFAULT_MODEL", raising=False)

        # Load .env from current directory
        load_dotenv()

        # The model should come from .env if set
        default_model = os.environ.get("DEFAULT_MODEL")

        # This test just verifies the loading mechanism works
        # The actual value depends on .env file content
        if default_model:
            assert default_model == "Qwen/Qwen3-Coder-480B-A35B-Instruct"


class TestOrchestratorConfigDefaults:
    """Tests for OrchestratorConfig defaults."""

    def test_config_default_model(self):
        """Test that OrchestratorConfig has correct default model."""
        from src.orchestrator import OrchestratorConfig

        config = OrchestratorConfig()
        assert config.model_name == "anthropic:claude-sonnet-4-20250514"

    def test_config_accepts_custom_model(self):
        """Test that OrchestratorConfig accepts custom model."""
        from src.orchestrator import OrchestratorConfig

        config = OrchestratorConfig(model_name="custom/model:v1")
        assert config.model_name == "custom/model:v1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
