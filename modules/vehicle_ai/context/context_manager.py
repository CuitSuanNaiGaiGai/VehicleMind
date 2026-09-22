from __future__ import annotations

import time

from collections import deque
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Any

from modules.vehicle_ai.context.contract import validate_domain_updates
from modules.vehicle_ai.context.models import (
    DriverContext,
    RoadContext,
    VehicleContext,
    VehicleStatus,
)


# ============================================================
# Context domain
# ============================================================


class ContextDomain(StrEnum):
    DRIVER = "driver"
    ROAD = "road"
    VEHICLE = "vehicle"


# ============================================================
# Context change
# ============================================================


@dataclass(frozen=True)
class ContextChange:
    """
    One semantic context change.

    Example:

        domain     = driver
        field      = state
        old_value  = NORMAL
        new_value  = DROWSY
    """

    domain: ContextDomain

    field: str

    old_value: Any
    new_value: Any

    timestamp: float

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "domain": self.domain,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp,
        }

    def __str__(
        self,
    ) -> str:

        return f"{self.domain}.{self.field}: {self.old_value} -> {self.new_value}"


# ============================================================
# Context Manager
# ============================================================


class ContextManager:
    """
    Central runtime context store for VehicleMind.

    Responsibilities:

        1. Hold the latest VehicleContext
        2. Update individual context domains
        3. Detect semantic state changes
        4. Track recent changes
        5. Provide safe context snapshots
        6. Report context freshness

    This class intentionally does NOT:

        - call an LLM
        - select prompt context
        - execute vehicle tools
        - make safety decisions

    Those belong to later layers.
    """

    # Fields managed by the runtime rather than partial-update callers.
    _IGNORED_FIELDS = {"updated_at", "source"}

    def __init__(
        self,
        initial_context: VehicleContext | None = None,
        max_change_history: int = 200,
    ):
        self._lock = RLock()

        if initial_context is None:
            self._context = VehicleContext()

        else:
            self._context = deepcopy(initial_context)

        self._changes: deque[ContextChange] = deque(maxlen=max_change_history)

    # ========================================================
    # Snapshot
    # ========================================================

    def get_context(
        self,
    ) -> VehicleContext:
        """
        Return a defensive copy of the complete context.

        External code should not directly mutate the internal
        context stored by ContextManager.
        """

        with self._lock:
            return deepcopy(self._context)

    def get_agent_context(
        self,
    ) -> dict[str, Any]:
        """
        Return the semantic context representation intended
        for future agent use.
        """

        with self._lock:
            return deepcopy(self._context.to_agent_context())

    # ========================================================
    # Internal update
    # ========================================================

    def _update_domain(
        self,
        domain: ContextDomain,
        updates: dict[str, Any],
    ) -> list[ContextChange]:
        """
        Apply partial updates to one context domain.

        Returns all semantic field changes produced by this
        update.
        """

        with self._lock:
            validate_domain_updates(domain.value, updates)
            target = getattr(self._context, domain.value)
            replacement = deepcopy(target)
            changes: list[ContextChange] = []
            now = time.time()
            for field_name, new_value in updates.items():
                old_value = getattr(target, field_name)
                if old_value == new_value:
                    continue
                setattr(replacement, field_name, new_value)
                changes.append(
                    ContextChange(
                        domain=domain,
                        field=field_name,
                        old_value=old_value,
                        new_value=new_value,
                        timestamp=now,
                    )
                )
            if updates:
                replacement.updated_at = now
                setattr(self._context, domain.value, replacement)
                self._changes.extend(changes)
            return changes

    # ========================================================
    # Driver update
    # ========================================================

    def update_driver(
        self,
        **updates: Any,
    ) -> list[ContextChange]:
        """
        Partially update DriverContext.

        Example:

            manager.update_driver(
                state=DriverState.DROWSY,
                risk=RiskLevel.HIGH,
            )
        """

        return self._update_domain(
            ContextDomain.DRIVER,
            updates,
        )

    # ========================================================
    # Road update
    # ========================================================

    def update_road(
        self,
        **updates: Any,
    ) -> list[ContextChange]:
        """
        Partially update RoadContext.
        """

        return self._update_domain(
            ContextDomain.ROAD,
            updates,
        )

    # ========================================================
    # Vehicle update
    # ========================================================

    def update_vehicle(
        self,
        **updates: Any,
    ) -> list[ContextChange]:
        """
        Partially update VehicleStatus.
        """

        return self._update_domain(
            ContextDomain.VEHICLE,
            updates,
        )

    # ========================================================
    # Full domain replacement
    # ========================================================

    def set_driver_context(
        self,
        context: DriverContext,
    ) -> list[ContextChange]:
        """
        Replace DriverContext while still generating
        field-level change information.
        """

        data = asdict(context)

        updates = {
            key: value
            for key, value in data.items()
            if key not in (self._IGNORED_FIELDS)
        }

        return self.update_driver(**updates)

    def set_road_context(
        self,
        context: RoadContext,
    ) -> list[ContextChange]:

        data = asdict(context)

        updates = {
            key: value
            for key, value in data.items()
            if key not in (self._IGNORED_FIELDS)
        }

        return self.update_road(**updates)

    def set_vehicle_status(
        self,
        context: VehicleStatus,
    ) -> list[ContextChange]:

        data = asdict(context)

        updates = {
            key: value
            for key, value in data.items()
            if key not in (self._IGNORED_FIELDS)
        }

        return self.update_vehicle(**updates)

    # ========================================================
    # Change history
    # ========================================================

    def recent_changes(
        self,
        limit: int = 20,
    ) -> list[ContextChange]:
        """
        Return recent semantic changes without removing them.
        """

        if limit <= 0:
            return []

        with self._lock:
            history = list(self._changes)

            return deepcopy(history[-limit:])

    def consume_changes(
        self,
    ) -> list[ContextChange]:
        """
        Return and clear pending changes.

        This will later become useful when EventBus consumes
        context transitions.
        """

        with self._lock:
            changes = list(self._changes)

            self._changes.clear()

            return deepcopy(changes)

    def clear_changes(
        self,
    ) -> None:

        with self._lock:
            self._changes.clear()

    # ========================================================
    # Freshness
    # ========================================================

    def freshness(
        self,
    ) -> dict[str, dict[str, Any]]:
        """
        Report age and freshness for each context domain.
        """

        with self._lock:
            now = time.time()

            driver_age = self._context.driver.age_seconds(now)

            road_age = self._context.road.age_seconds(now)

            vehicle_age = self._context.vehicle.age_seconds(now)

            return {
                "driver": {
                    "age_seconds": driver_age,
                    "fresh": self._context.driver.is_fresh(now=now),
                },
                "road": {
                    "age_seconds": road_age,
                    "fresh": self._context.road.is_fresh(now=now),
                },
                "vehicle": {
                    "age_seconds": vehicle_age,
                    "fresh": self._context.vehicle.is_fresh(now=now),
                },
            }

    # ========================================================
    # Convenience
    # ========================================================

    def summary(
        self,
    ) -> str:

        with self._lock:
            return self._context.summary()
