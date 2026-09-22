from enum import StrEnum


class DriverPresence(StrEnum):
    """High-level driver presence state."""

    UNKNOWN = "UNKNOWN"
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"


class DriverState(StrEnum):
    """High-level driver state."""

    UNKNOWN = "UNKNOWN"
    WARMING_UP = "WARMING_UP"
    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    DROWSY = "DROWSY"


class RiskLevel(StrEnum):
    """Driver-related risk level."""

    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class GearState(StrEnum):
    """Simplified vehicle gear state."""

    P = "P"
    R = "R"
    N = "N"
    D = "D"
    UNKNOWN = "UNKNOWN"


class NavigationState(StrEnum):
    """Navigation state used by the vehicle agent."""

    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    ACTIVE = "ACTIVE"
    ARRIVED = "ARRIVED"
