"""
Sub-agent Registry - Manage sub-agents with hot-reload capability.

Features:
- Load YAML definitions from directory
- Hot-reload on file changes
- Thread-safe registry access
"""

import builtins
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


@dataclass
class SubAgentSpec:
    """Specification for a sub-agent."""

    name: str
    description: str
    system_prompt: str
    tools: list[str] = None
    model: str | None = None
    tags: list[str] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.tags is None:
            self.tags = []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubAgentSpec":
        """Create SubAgentSpec from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt", ""),
            tools=data.get("tools", []),
            model=data.get("model"),
            tags=data.get("tags", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "model": self.model,
            "tags": self.tags,
        }


class SubAgentFileHandler(FileSystemEventHandler):
    """Handle file system events for agent YAML files."""

    def __init__(self, registry: "SubAgentRegistry"):
        self.registry = registry
        self.debounce_timer = None

    def on_modified(self, event):
        if event.src_path.endswith((".yaml", ".yml")) and not event.is_directory:
            self._debounced_reload(event.src_path)

    def on_created(self, event):
        if event.src_path.endswith((".yaml", ".yml")) and not event.is_directory:
            self._debounced_reload(event.src_path)

    def on_deleted(self, event):
        if event.src_path.endswith((".yaml", ".yml")) and not event.is_directory:
            self._handle_delete(event.src_path)

    def _debounced_reload(self, path: str):
        if self.debounce_timer:
            self.debounce_timer.cancel()
        self.debounce_timer = threading.Timer(0.5, self._do_reload, [path])
        self.debounce_timer.start()

    def _do_reload(self, path: str):
        try:
            self.registry.load_file(path)
            logger.info(f"Hot-reloaded sub-agent from {path}")
        except Exception as e:
            logger.error(f"Failed to reload {path}: {e}")

    def _handle_delete(self, path: str):
        try:
            filename = Path(path).stem
            self.registry.remove_by_filename(filename)
            logger.info(f"Removed sub-agent due to file deletion: {path}")
        except Exception as e:
            logger.error(f"Failed to handle deletion: {e}")


class SubAgentRegistry:
    """
    Registry for managing sub-agents with hot-reload capability.

    This registry:
    - Loads YAML definitions from a directory
    - Provides hot-reload on file changes
    - Maintains thread-safe access to sub-agent specs
    - Automatically updates the orchestrator when agents change
    """

    def __init__(self, orchestrator=None):
        """
        Initialize the sub-agent registry.

        Args:
            orchestrator: Optional DeepAgentOrchestrator instance to sync with
        """
        self.orchestrator = orchestrator
        self._subagents: dict[str, SubAgentSpec] = {}
        self._lock = threading.RLock()
        self._watcher: Observer | None = None
        self._watch_paths: list[str] = []

    def load_directory(self, path: str, recursive: bool = True) -> int:
        """
        Load all YAML files from a directory.

        Args:
            path: Directory path to load from
            recursive: Whether to search recursively

        Returns:
            Number of agents loaded
        """
        agent_path = Path(path)
        if not agent_path.exists():
            logger.warning(f"Agent directory not found: {path}")
            return 0

        count = 0
        pattern = "**/*.yaml" if recursive else "*.yaml"

        for yaml_file in agent_path.glob(pattern):
            try:
                self.load_file(str(yaml_file))
                count += 1
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

        logger.info(f"Loaded {count} sub-agents from {path}")
        return count

    def load_file(self, path: str) -> SubAgentSpec:
        """
        Load a single YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            Loaded SubAgentSpec
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty or invalid YAML file: {path}")

        spec = SubAgentSpec.from_dict(data)
        self.register(spec)

        return spec

    def register(self, spec: SubAgentSpec) -> SubAgentSpec:
        """
        Register a sub-agent specification.

        Args:
            spec: SubAgentSpec to register

        Returns:
            The registered spec
        """
        with self._lock:
            self._subagents[spec.name] = spec

        if self.orchestrator:
            self.orchestrator.add_subagent(
                name=spec.name,
                description=spec.description,
                system_prompt=spec.system_prompt,
                tools=spec.tools,
                model=spec.model,
            )

        logger.debug(f"Registered sub-agent: {spec.name}")
        return spec

    def unregister(self, name: str) -> bool:
        """
        Unregister a sub-agent by name.

        Args:
            name: Name of the sub-agent to unregister

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name not in self._subagents:
                return False
            spec = self._subagents.pop(name)

        if self.orchestrator:
            self.orchestrator.remove_subagent(name)

        logger.debug(f"Unregistered sub-agent: {name}")
        return True

    def remove_by_filename(self, filename: str) -> bool:
        """
        Remove a sub-agent by filename (without extension).

        Args:
            filename: Filename to match

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            for name, spec in self._subagents.items():
                if spec.name == filename:
                    return self.unregister(name)
        return False

    def get(self, name: str) -> SubAgentSpec | None:
        """
        Get a sub-agent specification by name.

        Args:
            name: Name of the sub-agent

        Returns:
            SubAgentSpec or None if not found
        """
        with self._lock:
            return self._subagents.get(name)

    def list(self) -> list[SubAgentSpec]:
        """
        List all registered sub-agents.

        Returns:
            List of SubAgentSpec objects
        """
        with self._lock:
            return list(self._subagents.values())

    def list_names(self) -> builtins.list[str]:
        """
        List all registered sub-agent names.

        Returns:
            List of names
        """
        with self._lock:
            return list(self._subagents.keys())

    def save_spec(self, spec: SubAgentSpec, path: str):
        """
        Save a sub-agent spec to a YAML file.

        Args:
            spec: SubAgentSpec to save
            path: Output file path
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(spec.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.debug(f"Saved sub-agent spec to {path}")

    def save_directory(self, spec: SubAgentSpec, directory: str) -> str:
        """
        Save a sub-agent spec to a YAML file in a directory.

        Args:
            spec: SubAgentSpec to save
            directory: Output directory

        Returns:
            Path to the saved file
        """
        os.makedirs(directory, exist_ok=True)
        filename = f"{spec.name}.yaml"
        path = os.path.join(directory, filename)
        self.save_spec(spec, path)
        return path

    def enable_hot_reload(self, paths: builtins.list[str]):
        """
        Enable file watching for hot-reload.

        Args:
            paths: List of directories to watch
        """
        if self._watcher is not None:
            return

        self._watch_paths = paths
        self._watcher = Observer()
        handler = SubAgentFileHandler(self)

        for path in paths:
            if os.path.exists(path):
                self._watcher.schedule(handler, path, recursive=True)
                logger.info(f"Watching directory for changes: {path}")

        self._watcher.start()
        logger.info("Hot-reload enabled")

    def disable_hot_reload(self):
        """Disable file watching."""
        if self._watcher:
            self._watcher.stop()
            self._watcher.join()
            self._watcher = None
            logger.info("Hot-reload disabled")

    def reload(self):
        """Reload all agents from watched directories."""
        for path in self._watch_paths:
            self.load_directory(path)

    def clear(self):
        """Clear all registered sub-agents."""
        with self._lock:
            self._subagents.clear()
        logger.info("Cleared all sub-agents from registry")

    def sync_with_orchestrator(self, orchestrator):
        """
        Sync registry with an orchestrator.

        Args:
            orchestrator: DeepAgentOrchestrator instance
        """
        self.orchestrator = orchestrator

        for spec in self.list():
            orchestrator.add_subagent(
                name=spec.name,
                description=spec.description,
                system_prompt=spec.system_prompt,
                tools=spec.tools,
                model=spec.model,
            )

        logger.info(f"Synced {len(self.list())} sub-agents with orchestrator")

    def __len__(self) -> int:
        """Return the number of registered sub-agents."""
        with self._lock:
            return len(self._subagents)

    def __contains__(self, name: str) -> bool:
        """Check if a sub-agent is registered."""
        with self._lock:
            return name in self._subagents

    def __iter__(self):
        """Iterate over sub-agent names."""
        return iter(self.list_names())
