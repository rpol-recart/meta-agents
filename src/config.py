"""
Configuration management for the orchestrator.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for the LLM model."""

    provider: str = "anthropic"
    name: str = "openai/gpt-oss-120b"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
    max_tokens: int = 4096


class BackendConfig(BaseModel):
    """Configuration for the backend."""

    type: str = "composite"
    root_dir: str = "/tmp/agent-workspace"
    store_namespace: str = "memories"


class SubAgentsConfig(BaseModel):
    """Configuration for sub-agents."""

    directory: str = "agents/"
    auto_load: bool = True
    hot_reload: bool = True


class HITLToolConfig(BaseModel):
    """Configuration for a HITL-protected tool."""

    allowed_decisions: list[str] = Field(
        default_factory=lambda: ["approve", "edit", "reject"]
    )


class HITLConfig(BaseModel):
    """Configuration for human-in-the-loop."""

    enabled: bool = True
    tools: dict[str, HITLToolConfig] = Field(default_factory=dict)


class APIConfig(BaseModel):
    """Configuration for the API server."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Settings(BaseModel):
    """Main settings configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    subagents: SubAgentsConfig = Field(default_factory=SubAgentsConfig)
    hitl: HITLConfig = Field(default_factory=HITLConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_settings(path: str | None = None) -> Settings:
    """
    Load settings from a YAML file.

    Args:
        path: Path to settings YAML file. If None, uses default settings.

    Returns:
        Settings instance
    """
    if path is None:
        config_dir = Path(__file__).parent.parent.parent / "config"
        config_path = config_dir / "settings.yaml"
    else:
        config_path = Path(path)

    if not config_path.exists():
        # Create default workspace directory
        _ensure_workspace_directory()
        return Settings()

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        _ensure_workspace_directory()
        return Settings()

    settings = Settings(**data)

    # Ensure workspace directory exists
    _ensure_workspace_directory(settings.backend.root_dir)

    return settings


def _ensure_workspace_directory(path: str = "/tmp/agent-workspace") -> None:
    """
    Ensure the workspace directory exists.

    Args:
        path: Path to workspace directory
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_settings(settings: Settings, path: str):
    """
    settings to a YAML file.

    Args:
        settings: Settings instance to save
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(settings.model_dump(), f, default_flow_style=False, sort_keys=False)


def resolve_env_vars(value: Any) -> Any:
    """
    Resolve environment variables in a value.

    Args:
        value: Value that may contain ${VAR_NAME}

    Returns:
        Value with environment variables resolved
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var, value)
    return value


def apply_env_overrides(settings: Settings) -> Settings:
    """
    Apply environment variable overrides to settings.

    Environment variables:
    - ANTHROPIC_API_KEY
    - OPENAI_API_KEY
    - OPENAI_BASE_URL
    - TAVILY_API_KEY
    - DEFAULT_MODEL
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        settings.model.api_key = os.environ["ANTHROPIC_API_KEY"]

    if os.environ.get("OPENAI_API_KEY"):
        settings.model.api_key = os.environ["OPENAI_API_KEY"]

    if os.environ.get("OPENAI_BASE_URL"):
        settings.model.base_url = os.environ["OPENAI_BASE_URL"]

    if os.environ.get("TAVILY_API_KEY"):
        pass  # Could be used by TavilySearchTool

    if os.environ.get("DEFAULT_MODEL"):
        settings.model.name = os.environ["DEFAULT_MODEL"]

    return settings
