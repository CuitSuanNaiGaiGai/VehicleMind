from __future__ import annotations

from collections import defaultdict
from collections import deque
from copy import deepcopy
from threading import RLock
from typing import Callable

from modules.vehicle_ai.events.events import (
    EventType,
    VehicleEvent,
)


# ============================================================
# Callback
# ============================================================


EventCallback = Callable[
    [VehicleEvent],
    None,
]


# ============================================================
# Event Bus
# ============================================================


class EventBus:
    """
    Lightweight synchronous event bus.

    Current purpose:

        Semantic Event
             ↓
          EventBus
             ↓
        ┌────┼───────────┐
        ▼    ▼           ▼
      Logger Agent   Safety Layer

    It is deliberately synchronous for the first version.

    Async execution will only be introduced later if it is
    actually needed.
    """

    def __init__(
        self,
        max_history: int = 200,
    ):
        self._lock = RLock()

        self._subscribers: dict[
            EventType,
            list[EventCallback],
        ] = defaultdict(list)

        self._global_subscribers: list[EventCallback] = []

        self._history: deque[VehicleEvent] = deque(maxlen=max_history)

        self._pending: deque[VehicleEvent] = deque(maxlen=max_history)

    # ========================================================
    # Subscribe
    # ========================================================

    def subscribe(
        self,
        event_type: EventType,
        callback: EventCallback,
    ) -> None:
        """
        Subscribe to one event type.
        """

        with self._lock:
            self._subscribers[event_type].append(callback)

    def subscribe_all(
        self,
        callback: EventCallback,
    ) -> None:
        """
        Receive every published event.
        """

        with self._lock:
            self._global_subscribers.append(callback)

    # ========================================================
    # Publish
    # ========================================================

    def publish(
        self,
        event: VehicleEvent,
    ) -> None:
        """
        Publish one semantic event.
        """

        with self._lock:
            self._history.append(event)

            self._pending.append(event)

            callbacks = list(
                self._subscribers.get(
                    event.type,
                    [],
                )
            )

            global_callbacks = list(self._global_subscribers)

        # ----------------------------------------------------
        # Execute callbacks outside the lock.
        #
        # A callback may later publish another event, so
        # holding the lock while running user code would be
        # undesirable.
        # ----------------------------------------------------

        for callback in global_callbacks:
            callback(event)

        for callback in callbacks:
            callback(event)

    def publish_many(
        self,
        events: list[VehicleEvent],
    ) -> None:

        for event in events:
            self.publish(event)

    # ========================================================
    # Pending events
    # ========================================================

    def pending_events(
        self,
    ) -> list[VehicleEvent]:
        """
        Inspect pending events without removing them.
        """

        with self._lock:
            return deepcopy(list(self._pending))

    def consume_pending(
        self,
    ) -> list[VehicleEvent]:
        """
        Return and clear pending events.

        Future VehicleAgent can consume events using this
        interface.
        """

        with self._lock:
            events = list(self._pending)

            self._pending.clear()

            return deepcopy(events)

    # ========================================================
    # History
    # ========================================================

    def recent_events(
        self,
        limit: int = 20,
    ) -> list[VehicleEvent]:

        if limit <= 0:
            return []

        with self._lock:
            history = list(self._history)

            return deepcopy(history[-limit:])

    # ========================================================
    # Clear
    # ========================================================

    def clear_pending(
        self,
    ) -> None:

        with self._lock:
            self._pending.clear()
