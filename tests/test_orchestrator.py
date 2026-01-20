
from src.config import Settings, apply_env_overrides, load_settings
from src.orchestrator import OrchestratorConfig


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig class."""

    def test_defaults(self):
        """Test default values."""
        config = OrchestratorConfig()

        assert config.model_name == "anthropic:claude-sonnet-4-20250514"
        assert config.system_prompt is None
        assert config.subagents == []
        assert config.interrupt_on is None
        assert config.custom_tools == []
        assert config.middleware == []
        assert config.enable_memory is False  # Disabled to avoid store errors
        assert config.memory_namespace == "memories"
        assert config.workspace_dir == "/tmp/agent-workspace"

    def test_custom_values(self):
        """Test custom values."""
        subagents = [
            {"name": "test_agent", "description": "Test", "system_prompt": "Test."}
        ]
        config = OrchestratorConfig(
            model_name="openai:gpt-4o",
            system_prompt="You are custom.",
            subagents=subagents,
            enable_memory=False,
        )

        assert config.model_name == "openai:gpt-4o"
        assert config.system_prompt == "You are custom."
        assert len(config.subagents) == 1
        assert config.enable_memory is False

    def test_from_settings_default(self):
        """Test creating config from default Settings."""
        settings = Settings()
        config = OrchestratorConfig.from_settings(settings)

        # Should use Settings.model.name (not OrchestratorConfig default)
        assert config.model_name == settings.model.name
        assert config.workspace_dir == settings.backend.root_dir
        assert config.memory_namespace == settings.backend.store_namespace
        assert config.enable_memory == (
            settings.hitl.enabled if settings.hitl else False
        )

    def test_from_settings_with_env_override(self):
        """Test that environment variables override Settings."""
        import os

        # Save original env
        original_model = os.environ.get("DEFAULT_MODEL")
        original_base_url = os.environ.get("OPENAI_BASE_URL")

        try:
            # Set env vars
            os.environ["DEFAULT_MODEL"] = "anthropic:claude-3-5-sonnet"
            os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"

            settings = Settings()
            settings = apply_env_overrides(settings)
            config = OrchestratorConfig.from_settings(settings)

            # Should use env override
            assert config.model_name == "anthropic:claude-3-5-sonnet"
            assert config.openai_base_url == "http://localhost:11434/v1"
        finally:
            # Restore env
            if original_model:
                os.environ["DEFAULT_MODEL"] = original_model
            else:
                os.environ.pop("DEFAULT_MODEL", None)
            if original_base_url:
                os.environ["OPENAI_BASE_URL"] = original_base_url
            else:
                os.environ.pop("OPENAI_BASE_URL", None)

    def test_from_settings_creates_workspace_directory(self):
        """Test that from_settings creates the workspace directory."""
        import tempfile
        from pathlib import Path

        settings = Settings()
        custom_workspace = tempfile.mktemp(prefix="test_workspace_")
        settings.backend.root_dir = custom_workspace

        try:
            config = OrchestratorConfig.from_settings(settings)
            assert Path(custom_workspace).exists()
            assert config.workspace_dir == custom_workspace
        finally:
            # Cleanup
            import shutil

            if Path(custom_workspace).exists():
                shutil.rmtree(custom_workspace)

    def test_config_consistency(self):
        """Test that Settings and OrchestratorConfig are consistent."""
        settings = load_settings()
        settings = apply_env_overrides(settings)
        config = OrchestratorConfig.from_settings(settings)

        # Both should have the same workspace directory
        assert config.workspace_dir == settings.backend.root_dir

        # Both should have consistent memory namespace
        assert config.memory_namespace == settings.backend.store_namespace

        # Model names should match between settings and config
        # (after env overrides)
        if settings.model.name:
            assert config.model_name == settings.model.name
