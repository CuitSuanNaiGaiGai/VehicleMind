from __future__ import annotations

from modules.vehicle_ai.memory.event_store import TripEvent, TripEventStore


def test_store_deduplicates_event_ids_and_isolates_trips(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    event = TripEvent(
        event_id="e-1",
        trip_id="trip-a",
        event_type="HIGH_RISK_DETECTED",
        occurred_at=10.0,
        source="event_bus",
        payload={"risk": "HIGH"},
    )

    assert store.append(event) is True
    assert store.append(event) is False
    assert store.count(trip_id="trip-a") == 1
    assert store.count(trip_id="trip-b") == 0


def test_store_filters_time_orders_events_and_survives_reopen(tmp_path) -> None:
    path = tmp_path / "events.sqlite3"
    store = TripEventStore(path)
    for event_id, timestamp in (("e-2", 20.0), ("e-1", 10.0), ("e-3", 30.0)):
        store.append(
            TripEvent(event_id, "trip-a", "REMINDER", timestamp, "agent", {})
        )

    reopened = TripEventStore(path)
    events = reopened.query(
        trip_id="trip-a", event_type="REMINDER", since=10.0, until=20.0
    )

    assert [event.event_id for event in events] == ["e-1", "e-2"]


def test_store_retention_and_trip_clear(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    for event_id, timestamp in (("old", 10.0), ("new", 100.0)):
        store.append(TripEvent(event_id, "trip-a", "RISK", timestamp, "event", {}))
    assert store.prune(before=50.0) == 1
    assert store.count(trip_id="trip-a") == 1
    assert store.clear_trip("trip-a") == 1
    assert store.count(trip_id="trip-a") == 0


def test_store_rejects_invalid_time_range_and_unbounded_limit(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    try:
        store.query(trip_id="trip-a", since=5.0, until=1.0)
    except ValueError as error:
        assert "since" in str(error)
    else:
        raise AssertionError("invalid time range should be rejected")

    try:
        store.query(trip_id="trip-a", limit=0)
    except ValueError as error:
        assert "limit" in str(error)
    else:
        raise AssertionError("non-positive limit should be rejected")

    for invalid_limit in (1.5, True):
        try:
            store.query(trip_id="trip-a", limit=invalid_limit)
        except ValueError as error:
            assert "limit" in str(error)
        else:
            raise AssertionError("non-integer limit should be rejected")

    try:
        store.query(trip_id="trip-a", since=float("nan"))
    except ValueError as error:
        assert "since" in str(error)
    else:
        raise AssertionError("non-finite query time should be rejected")


def test_count_supports_event_type_and_time_range(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    store.append(TripEvent("r1", "trip-a", "REMINDER", 10, "agent", {}))
    store.append(TripEvent("r2", "trip-a", "REMINDER", 20, "agent", {}))
    store.append(TripEvent("x1", "trip-a", "RISK", 15, "event", {}))

    assert store.count(
        trip_id="trip-a", event_type="REMINDER", since=15, until=20
    ) == 1


def test_bounded_query_returns_the_most_recent_events_in_time_order(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    for index in range(100):
        store.append(
            TripEvent(str(index), "trip-a", "USER_REQUEST", index, "user", {})
        )

    events = store.query(trip_id="trip-a", limit=20)

    assert [event.occurred_at for event in events] == list(range(80, 100))
