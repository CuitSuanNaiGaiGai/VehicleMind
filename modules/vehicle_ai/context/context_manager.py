from __future__ import annotations

import time

from collections import deque
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Any
from collections.abc import Callable

from modules.observation import ObservationMetadata
from modules.vehicle_ai.context.contract import (
    CONTEXT_SCHEMA_VERSION,
    CONTEXT_FIELD_CONTRACTS,
    validate_domain_updates,
)
from modules.vehicle_ai.context.models import (
    DriverContext,
    RoadContext,
    VehicleContext,
    VehicleStatus,
)
from modules.vehicle_ai.context.quality import (
    ObservationQualityTracker,
    QualityStatus,
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
        quality_clock: Callable[[], float] = time.monotonic,
    ):
        self._lock = RLock()

        if initial_context is None:
            self._context = VehicleContext()

        else:
            if (
                type(initial_context.schema_version) is not int
                or initial_context.schema_version != CONTEXT_SCHEMA_VERSION
            ):
                raise ValueError("initial context schema_version is unsupported")
            for domain, rules in CONTEXT_FIELD_CONTRACTS.items():
                value = getattr(initial_context, domain)
                validate_domain_updates(
                    domain,
                    {field_name: getattr(value, field_name) for field_name in rules},
                )
            self._context = deepcopy(initial_context)

        self._changes: deque[ContextChange] = deque(maxlen=max_change_history)
        self._quality = ObservationQualityTracker(clock=quality_clock)

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
        observation: ObservationMetadata | None = None,
    ) -> list[ContextChange]:
        """
        Apply partial updates to one context domain.

        Returns all semantic field changes produced by this
        update.
        """

        with self._lock:
            if observation is not None and not observation.valid:
                raise ValueError("semantic update requires a valid observation")
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
                self._quality.record(
                    domain.value,
                    valid=True,
                    metadata=observation,
                    fields=tuple(updates),
                )
            return changes

    def mark_invalid_observation(
        self, domain: str, metadata: ObservationMetadata
    ) -> None:
        if metadata.valid:
            raise ValueError("invalid observation marker requires valid=False")
        with self._lock:
            self._quality.record(domain, valid=False, metadata=metadata)

    def observation_quality(
        self, domain: str, *, now: float | None = None
    ) -> dict[str, Any]:
        with self._lock:
            return self._quality.report(domain, now=now)

    def field_quality(
        self, domain: str, field: str, *, now: float | None = None
    ) -> QualityStatus:
        if domain not in CONTEXT_FIELD_CONTRACTS:
            raise ValueError(f"unknown context domain: {domain}")
        if field not in CONTEXT_FIELD_CONTRACTS[domain]:
            raise ValueError(f"unknown context field: {domain}.{field}")
        with self._lock:
            value = getattr(getattr(self._context, domain), field)
            return self._quality.field_status(domain, field, value, now=now)

    # ========================================================
    # Driver update
    # ========================================================

    def update_driver(
        self,
        observation: ObservationMetadata | None = None,
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
            observation,
        )

    # ========================================================
    # Road update
    # ========================================================

    def update_road(
        self,
        observation: ObservationMetadata | None = None,
        **updates: Any,
    ) -> list[ContextChange]:
        """
        Partially update RoadContext.
        """

        return self._update_domain(
            ContextDomain.ROAD,
            updates,
            observation,
        )

    # ========================================================
    # Vehicle update
    # ========================================================

    def update_vehicle(
        self,
        observation: ObservationMetadata | None = None,
        **updates: Any,
    ) -> list[ContextChange]:
        """
        Partially update VehicleStatus.
        """

        return self._update_domain(
            ContextDomain.VEHICLE,
            updates,
            observation,
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
            now = time.monotonic()
            result = {}
            for domain in ("driver", "road", "vehicle"):
                quality = self._quality.report(domain, now=now)
                result[domain] = {
                    "age_seconds": quality["age_seconds"],
                    "fresh": quality["status"] is QualityStatus.KNOWN,
                    "status": quality["status"],
                }
            return result

    # ========================================================
    # Convenience
    # ========================================================

    def summary(
        self,
    ) -> str:

        with self._lock:
            return self._context.summary()
