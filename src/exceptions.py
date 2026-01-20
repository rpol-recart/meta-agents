"""
Custom exceptions for DeepAgent Orchestrator.

This module defines a hierarchy of exceptions for consistent error handling
across the application.
"""

from typing import Any, Dict


class OrchestratorError(Exception):
    """Base exception for all orchestrator errors."""

    def __init__(
        self,
        message: str,
        details: Dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

        if cause:
            self.__cause__ = cause

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ModelInitializationError(OrchestratorError):
    """Raised when model initialization fails."""

    def __init__(
        self,
        message: str = "Failed to initialize model",
        model_name: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if model_name:
            details["model_name"] = model_name
        super().__init__(message, details=details, **kwargs)


class AgentCreationError(OrchestratorError):
    """Raised when agent creation fails."""

    def __init__(
        self,
        message: str = "Failed to create agent",
        agent_name: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if agent_name:
            details["agent_name"] = agent_name
        super().__init__(message, details=details, **kwargs)


class TaskExecutionError(OrchestratorError):
    """Raised when task execution fails."""

    def __init__(
        self,
        message: str = "Task execution failed",
        task: str | None = None,
        thread_id: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if task:
            details["task"] = task[:100]  # Truncate long tasks
        if thread_id:
            details["thread_id"] = thread_id
        super().__init__(message, details=details, **kwargs)


class ConfigurationError(OrchestratorError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str = "Invalid configuration",
        config_key: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details=details, **kwargs)


class SubAgentError(OrchestratorError):
    """Raised when sub-agent operations fail."""

    def __init__(
        self,
        message: str = "Sub-agent operation failed",
        subagent_name: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if subagent_name:
            details["subagent_name"] = subagent_name
        super().__init__(message, details=details, **kwargs)


class BackendError(OrchestratorError):
    """Raised when backend operations fail."""

    def __init__(
        self,
        message: str = "Backend operation failed",
        backend_type: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if backend_type:
            details["backend_type"] = backend_type
        super().__init__(message, details=details, **kwargs)


class ToolExecutionError(OrchestratorError):
    """Raised when tool execution fails."""

    def __init__(
        self,
        message: str = "Tool execution failed",
        tool_name: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(message, details=details, **kwargs)


class FileOperationError(OrchestratorError):
    """Raised when file operations fail."""

    def __init__(
        self,
        message: str = "File operation failed",
        file_path: str | None = None,
        operation: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if file_path:
            details["file_path"] = file_path
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details, **kwargs)


class ValidationError(OrchestratorError):
    """Raised when validation fails."""

    def __init__(
        self,
        message: str = "Validation failed",
        field_name: str | None = None,
        value: Any | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if field_name:
            details["field_name"] = field_name
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, details=details, **kwargs)


class OrchestratorCreationError(OrchestratorError):
    """Raised when orchestrator creation fails."""

    def __init__(
        self,
        message: str = "Failed to create orchestrator",
        model_name: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if model_name:
            details["model_name"] = model_name
        super().__init__(message, details=details, **kwargs)


def wrap_exception(
    exception: Exception,
    new_class: type = OrchestratorError,
    message: str | None = None,
) -> OrchestratorError:
    """
    Wrap an exception in an OrchestratorError.

    Args:
        exception: The original exception
        new_class: The exception class to use for wrapping
        message: Optional custom message

    Returns:
        Wrapped exception
    """
    if isinstance(exception, OrchestratorError):
        return exception

    return new_class(message=message or str(exception), cause=exception)
