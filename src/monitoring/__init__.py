"""
Monitoring module for DeepAgent Orchestrator.

This module provides observability and monitoring capabilities including:
- Metrics collection
- Health check endpoints
- Distributed tracing support
- Centralized logging
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result."""

    status: HealthStatus
    component: str
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "component": self.component,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Metrics:
    """Metrics snapshot."""

    task_count: int = 0
    task_success_count: int = 0
    task_failure_count: int = 0
    average_task_duration_ms: float = 0.0
    active_threads: int = 0
    memory_usage_mb: float = 0.0
    uptime_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_count": self.task_count,
            "task_success_count": self.task_success_count,
            "task_failure_count": self.task_failure_count,
            "average_task_duration_ms": self.average_task_duration_ms,
            "active_threads": self.active_threads,
            "memory_usage_mb": self.memory_usage_mb,
            "uptime_seconds": self.uptime_seconds,
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.task_count == 0:
            return 0.0
        return self.task_success_count / self.task_count


class MetricsCollector:
    """Collector for orchestrator metrics."""

    def __init__(self):
        self._task_count = 0
        self._task_success_count = 0
        self._task_failure_count = 0
        self._total_task_duration_ms = 0.0
        self._active_threads: set[str] = set()
        self._start_time = datetime.utcnow()
        self._durations: list[float] = []

    def record_task_start(self, thread_id: str) -> None:
        """Record task start."""
        self._task_count += 1
        self._active_threads.add(thread_id)

    def record_task_complete(self, thread_id: str, duration_ms: float) -> None:
        """Record task completion."""
        self._task_success_count += 1
        self._active_threads.discard(thread_id)
        self._total_task_duration_ms += duration_ms
        self._durations.append(duration_ms)

        if len(self._durations) > 100:
            self._durations.pop(0)

    def record_task_failure(self, thread_id: str, duration_ms: float) -> None:
        """Record task failure."""
        self._task_failure_count += 1
        self._active_threads.discard(thread_id)
        self._total_task_duration_ms += duration_ms
        self._durations.append(duration_ms)

        if len(self._durations) > 100:
            self._durations.pop(0)

    def get_metrics(self) -> Metrics:
        """Get current metrics."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()

        avg_duration = 0.0
        if self._durations:
            avg_duration = sum(self._durations) / len(self._durations)

        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            memory_mb = 0.0

        return Metrics(
            task_count=self._task_count,
            task_success_count=self._task_success_count,
            task_failure_count=self._task_failure_count,
            average_task_duration_ms=avg_duration,
            active_threads=len(self._active_threads),
            memory_usage_mb=memory_mb,
            uptime_seconds=uptime,
        )


class HealthChecker:
    """Health check manager for the orchestrator."""

    def __init__(self):
        self._components: Dict[str, callable] = {}
        self._last_health_check: Optional[HealthCheck] = None

    def register_component(self, name: str, check_func: callable) -> None:
        """
        Register a health check component.

        Args:
            name: Component name
            check_func: Async function that returns HealthCheck
        """
        self._components[name] = check_func
        logger.debug(f"Registered health check component: {name}")

    async def check_all(self) -> list[HealthCheck]:
        """
        Run all health checks.

        Returns:
            List of health check results
        """
        results = []

        for name, check_func in self._components.items():
            try:
                if check_func.__code__.co_flags & 0x80:
                    result = await check_func()
                else:
                    result = check_func()
                if isinstance(result, HealthCheck):
                    results.append(result)
                else:
                    results.append(
                        HealthCheck(
                            status=HealthStatus.UNKNOWN,
                            component=name,
                            message="Invalid health check result",
                        )
                    )
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results.append(
                    HealthCheck(
                        status=HealthStatus.UNHEALTHY,
                        component=name,
                        message=str(e),
                    )
                )

        self._last_health_check = results[0] if results else None
        return results

    def get_overall_status(self, checks: list[HealthCheck]) -> HealthStatus:
        """Determine overall health status from check results."""
        if not checks:
            return HealthStatus.UNKNOWN

        statuses = [check.status for check in checks]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN


class ObservabilityManager:
    """
    Central manager for observability features.

    This class integrates metrics collection, health checking, and logging
    for comprehensive system monitoring.

    Example:
        >>> obs = ObservabilityManager()
        >>> obs.start()
        >>> # ... perform operations ...
        >>> metrics = obs.get_metrics()
        >>> health = await obs.check_health()
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self._enabled = False

    def enable(self) -> None:
        """Enable observability features."""
        self._enabled = True
        self._setup_logging()

    def disable(self) -> None:
        """Disable observability features."""
        self._enabled = False

    def _setup_logging(self) -> None:
        """Configure enhanced logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    def record_task_start(self, thread_id: str) -> None:
        """Record task start event."""
        if self._enabled:
            self.metrics_collector.record_task_start(thread_id)

    def record_task_complete(self, thread_id: str, duration_ms: float) -> None:
        """Record task completion event."""
        if self._enabled:
            self.metrics_collector.record_task_complete(thread_id, duration_ms)

    def record_task_failure(self, thread_id: str, duration_ms: float) -> None:
        """Record task failure event."""
        if self._enabled:
            self.metrics_collector.record_task_failure(thread_id, duration_ms)

    def get_metrics(self) -> Metrics:
        """Get current metrics."""
        return self.metrics_collector.get_metrics()

    async def check_health(self) -> list[HealthCheck]:
        """Run all health checks."""
        return await self.health_checker.check_all()

    def get_health_status(self) -> HealthStatus:
        """Get overall health status."""
        import asyncio
        checks = asyncio.run(self.check_health())
        return self.health_checker.get_overall_status(checks)


def get_memory_usage_mb() -> float:
    """Get current memory usage in megabytes."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def get_cpu_percent() -> float:
    """Get current CPU usage percentage."""
    try:
        import psutil
        return psutil.cpu_percent()
    except ImportError:
        return 0.0


def get_uptime_seconds() -> float:
    """Get system uptime in seconds."""
    import os

    try:
        with open("/proc/uptime", "r") as f:
            return float(f.read().split()[0])
    except (FileNotFoundError, IOError):
        return 0.0
