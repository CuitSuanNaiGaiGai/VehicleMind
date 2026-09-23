from __future__ import annotations

import threading

import pytest

from modules.vehicle_ai.pipeline.channel import (
    BoundedChannel,
    ChannelClosed,
    OverflowPolicy,
)


def test_drop_oldest_keeps_latest_frames_and_bounded_depth() -> None:
    channel: BoundedChannel[int] = BoundedChannel(2, OverflowPolicy.DROP_OLDEST)

    channel.put(1)
    channel.put(2)
    channel.put(3)

    assert channel.depth == 2
    assert channel.dropped == 1
    assert channel.get() == 2
    assert channel.get() == 3


def test_blocking_channel_preserves_completed_snapshots() -> None:
    channel: BoundedChannel[int] = BoundedChannel(1, OverflowPolicy.BLOCK)
    channel.put(1)
    entered = threading.Event()
    finished = threading.Event()

    def producer() -> None:
        entered.set()
        channel.put(2)
        finished.set()

    thread = threading.Thread(target=producer)
    thread.start()
    assert entered.wait(1)
    assert not finished.wait(0.02)
    assert channel.get() == 1
    assert finished.wait(1)
    assert channel.get() == 2
    thread.join(1)
    assert channel.dropped == 0


def test_close_wakes_waiting_consumer() -> None:
    channel: BoundedChannel[int] = BoundedChannel(1, OverflowPolicy.BLOCK)
    observed: list[str] = []
    ready = threading.Event()

    def consumer() -> None:
        ready.set()
        with pytest.raises(ChannelClosed):
            channel.get()
        observed.append("closed")

    thread = threading.Thread(target=consumer)
    thread.start()
    assert ready.wait(1)
    channel.close()
    thread.join(1)

    assert observed == ["closed"]
    with pytest.raises(ChannelClosed):
        channel.put(1)
