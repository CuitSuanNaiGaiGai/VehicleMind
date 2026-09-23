from __future__ import annotations

from collections import deque
from enum import StrEnum
from threading import Condition
from typing import Generic, TypeVar


T = TypeVar("T")


class OverflowPolicy(StrEnum):
    DROP_OLDEST = "drop_oldest"
    BLOCK = "block"


class ChannelClosed(RuntimeError):
    """Raised when a closed, drained channel cannot provide or accept items."""


class BoundedChannel(Generic[T]):
    """A thread-safe bounded channel with explicit overflow semantics."""

    def __init__(self, capacity: int, policy: OverflowPolicy) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not isinstance(policy, OverflowPolicy):
            raise TypeError("policy must be an OverflowPolicy")
        self.capacity = capacity
        self.policy = policy
        self._items: deque[T] = deque()
        self._condition = Condition()
        self._closed = False
        self._dropped = 0

    @property
    def depth(self) -> int:
        with self._condition:
            return len(self._items)

    @property
    def dropped(self) -> int:
        with self._condition:
            return self._dropped

    def put(self, item: T) -> None:
        with self._condition:
            if self._closed:
                raise ChannelClosed("channel is closed")
            if self.policy is OverflowPolicy.BLOCK:
                while len(self._items) >= self.capacity and not self._closed:
                    self._condition.wait()
                if self._closed:
                    raise ChannelClosed("channel is closed")
            elif len(self._items) >= self.capacity:
                self._items.popleft()
                self._dropped += 1
            self._items.append(item)
            self._condition.notify_all()

    def get(self, timeout: float | None = None) -> T:
        with self._condition:
            ready = self._condition.wait_for(
                lambda: bool(self._items) or self._closed, timeout=timeout
            )
            if not ready:
                raise TimeoutError("channel receive timed out")
            if not self._items:
                raise ChannelClosed("channel is closed and drained")
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
