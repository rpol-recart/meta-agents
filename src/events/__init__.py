"""
Events module for DeepAgent Orchestrator.

This module provides an event-driven communication system for loose coupling
between components and extensibility.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Dict, List
from collections.abc import Coroutine
from datetime import datetime


logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event types for the orchestrator."""

    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    AGENT_CREATED = "agent_created"
    SUBAGENT_ADDED = "subagent_added"
    SUBAGENT_REMOVED = "subagent_removed"
    HITL_INTERRUPT = "hitl_interrupt"
    STATE_CLEARED = "state_cleared"
    ERROR_OCCURRED = "error_occurred"
    MODEL_INITIALIZED = "model_initialized"
    CONFIG_CHANGED = "config_changed"


@dataclass
class Event:
    """Base event class for all events."""

    type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = "orchestrator"

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "source": self.source,
        }


EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """
    Event bus for publish-subscribe communication.

    This class provides an event-driven communication mechanism that enables
    loose coupling between components and extensibility.

    Example:
        >>> event_bus = EventBus()
        >>> async def handler(event: Event):
        ...     print(f"Received: {event.type}")
        >>> event_bus.subscribe(EventType.TASK_COMPLETED, handler)
        >>> await event_bus.publish(Event(type=EventType.TASK_COMPLETED))
    """

    def __init__(self):
        self._subscribers: Dict[EventType, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: The event type to subscribe to
            handler: Async handler function to call when event is published
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to {event_type.value}")

    def subscribe_all(self, handler: EventHandler) -> None:
        """
        Subscribe to all event types.

        Args:
            handler: Async handler function to call for all events
        """
        self._global_handlers.append(handler)
        logger.debug("Subscribed to all events")

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> bool:
        """
        Unsubscribe from an event type.

        Args:
            event_type: The event type to unsubscribe from
            handler: The handler to remove

        Returns:
            True if handler was found and removed
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed from {event_type.value}")
                return True
            except ValueError:
                pass
        return False

    def unsubscribe_all(self, handler: EventHandler) -> bool:
        """
        Unsubscribe from all event types.

        Args:
            handler: The handler to remove from all subscriptions

        Returns:
            True if handler was found and removed
        """
        removed = False
        for event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                removed = True
            except ValueError:
                pass
        if removed:
            logger.debug("Unsubscribed from all events")
        return removed

    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: The event to publish
        """
        handlers: List[EventHandler] = []

        if event.type in self._subscribers:
            handlers.extend(self._subscribers[event.type])

        handlers.extend(self._global_handlers)

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.type.value}: {e}")

    def get_subscribers(self, event_type: EventType) -> List[EventHandler]:
        """Get all subscribers for an event type."""
        return self._subscribers.get(event_type, []).copy()

    def get_all_subscribers(self) -> Dict[EventType, List[EventHandler]]:
        """Get all subscribers."""
        return {k: v.copy() for k, v in self._subscribers.items()}


class EventEmitter:
    """Mixin class for emitting events."""

    def __init__(self, *args, **kwargs):
        self._event_bus: EventBus | None = None
        super().__init__(*args, **kwargs)

    def set_event_bus(self, event_bus: EventBus) -> None:
        """
        Set the event bus for this emitter.

        Args:
            event_bus: The event bus to use
        """
        self._event_bus = event_bus

    def emit(self, event: Event) -> Coroutine[Any, Any, None]:
        """
        Emit an event through the event bus.

        Args:
            event: The event to emit

        Returns:
            Awaitable that completes when event is published
        """
        if self._event_bus:
            return self._event_bus.publish(event)
        return self._noop()

    async def _noop(self) -> None:
        """No-op when no event bus is configured."""
        pass

    def emit_sync(self, event: Event) -> None:
        """
        Emit an event synchronously (for non-async contexts).

        Args:
            event: The event to emit
        """
        if self._event_bus:
            import asyncio
            asyncio.create_task(self._event_bus.publish(event))


def create_event(
    event_type: EventType,
    payload: Dict[str, Any] | None = None,
    source: str = "orchestrator",
) -> Event:
    """
    Factory function to create an event.

    Args:
        event_type: The type of event
        payload: Optional event payload
        source: The source of the event

    Returns:
        Event instance
    """
    return Event(
        type=event_type,
        payload=payload or {},
        source=source,
    )


def setup_default_event_handlers(event_bus: EventBus) -> None:
    """
    Set up default event handlers for logging and monitoring.

    Args:
        event_bus: The event bus to configure
    """
    async def log_event(event: Event):
        logger.info(f"Event: {event.type.value} - {event.payload}")

    async def log_error(event: Event):
        logger.error(f"Error event: {event.type.value} - {event.payload}")

    event_bus.subscribe_all(log_event)

    event_bus.subscribe(EventType.TASK_FAILED, log_error)
    event_bus.subscribe(EventType.ERROR_OCCURRED, log_error)
