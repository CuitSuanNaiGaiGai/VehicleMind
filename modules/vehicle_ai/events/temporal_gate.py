from __future__ import annotations


class TemporalGate:
    """Emit only for a sustained condition outside its cooldown window."""

    def __init__(
        self, *, hold_ms: int, cooldown_ms: int, max_gap_ms: int | None = None
    ) -> None:
        if type(hold_ms) is not int or hold_ms < 0:
            raise ValueError("hold_ms must be a nonnegative integer")
        if type(cooldown_ms) is not int or cooldown_ms <= 0:
            raise ValueError("cooldown_ms must be a positive integer")
        if max_gap_ms is not None and (type(max_gap_ms) is not int or max_gap_ms <= 0):
            raise ValueError("max_gap_ms must be a positive integer")
        self.hold_ms = hold_ms
        self.cooldown_ms = cooldown_ms
        self.max_gap_ms = max_gap_ms
        self._active_since: int | None = None
        self._last_emitted: int | None = None
        self._last_observed: int | None = None

    def observe(self, active: bool, *, at_ms: int) -> bool:
        if type(active) is not bool or type(at_ms) is not int or at_ms < 0:
            raise ValueError("active must be boolean and at_ms nonnegative integer")
        if self._last_observed is not None and at_ms < self._last_observed:
            raise ValueError("observation time must not go backwards")
        if (
            self.max_gap_ms is not None
            and self._last_observed is not None
            and at_ms - self._last_observed > self.max_gap_ms
        ):
            self._active_since = None
        self._last_observed = at_ms
        if not active:
            self._active_since = None
            return False
        if self._active_since is None:
            self._active_since = at_ms
        if at_ms - self._active_since < self.hold_ms:
            return False
        if (
            self._last_emitted is not None
            and at_ms - self._last_emitted < self.cooldown_ms
        ):
            return False
        self._last_emitted = at_ms
        return True

    def reset(self) -> None:
        self._active_since = None


class StableValueGate:
    """Report a value transition only after the new value remains stable."""

    def __init__(
        self, *, initial: str, hold_ms: int, max_gap_ms: int | None = None
    ) -> None:
        if type(hold_ms) is not int or hold_ms < 0:
            raise ValueError("hold_ms must be a nonnegative integer")
        if max_gap_ms is not None and (type(max_gap_ms) is not int or max_gap_ms <= 0):
            raise ValueError("max_gap_ms must be a positive integer")
        self._stable = initial
        self._candidate: str | None = None
        self._since: int | None = None
        self._last_observed: int | None = None
        self._hold_ms = hold_ms
        self._max_gap_ms = max_gap_ms

    def observe(self, value: str, *, at_ms: int) -> tuple[str, str] | None:
        if type(at_ms) is not int or at_ms < 0:
            raise ValueError("at_ms must be a nonnegative integer")
        if self._last_observed is not None and at_ms < self._last_observed:
            raise ValueError("observation time must not go backwards")
        if (
            self._max_gap_ms is not None
            and self._last_observed is not None
            and at_ms - self._last_observed > self._max_gap_ms
        ):
            self.reset()
        self._last_observed = at_ms
        if value == self._stable:
            self._candidate = None
            self._since = None
            return None
        if value != self._candidate:
            self._candidate = value
            self._since = at_ms
        assert self._since is not None
        if at_ms - self._since < self._hold_ms:
            return None
        previous = self._stable
        self._stable = value
        self._candidate = None
        self._since = None
        return previous, value

    def reset(self) -> None:
        self._candidate = None
        self._since = None
