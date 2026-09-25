"""SQLite event store for bounded, trip-scoped historical evidence."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


MAX_QUERY_LIMIT = 50


def _validate_time_range(since: float | None, until: float | None) -> None:
    for name, value in (("since", since), ("until", until)):
        if value is not None and (
            type(value) not in {int, float} or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite Unix timestamp")
    if since is not None and until is not None and since > until:
        raise ValueError("since must not be later than until")


@dataclass(frozen=True)
class TripEvent:
    event_id: str
    trip_id: str
    event_type: str
    occurred_at: float
    source: str
    payload: dict

    def __post_init__(self) -> None:
        for name in ("event_id", "trip_id", "event_type", "source"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if not math.isfinite(self.occurred_at):
            raise ValueError("occurred_at must be finite")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a JSON object")


class TripEventStore:
    """Small standard-library SQLite store with per-operation connections."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS trip_events (
                    event_id TEXT PRIMARY KEY,
                    trip_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at REAL NOT NULL,
                    source TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_trip_events_trip_time "
                "ON trip_events(trip_id, occurred_at, event_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_trip_events_trip_type_time "
                "ON trip_events(trip_id, event_type, occurred_at, event_id)"
            )

    def append(self, event: TripEvent) -> bool:
        """Insert once by event ID; return False for an already-seen event."""
        payload = json.dumps(
            event.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO trip_events
                   (event_id, trip_id, event_type, occurred_at, source, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.trip_id,
                    event.event_type,
                    event.occurred_at,
                    event.source,
                    payload,
                ),
            )
            return cursor.rowcount == 1

    def query(
        self,
        *,
        trip_id: str,
        event_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 20,
    ) -> list[TripEvent]:
        if not trip_id.strip():
            raise ValueError("trip_id must not be empty")
        _validate_time_range(since, until)
        if type(limit) is not int or not 1 <= limit <= MAX_QUERY_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_QUERY_LIMIT}")

        clauses = ["trip_id = ?"]
        parameters: list[object] = [trip_id]
        if event_type is not None:
            clauses.append("event_type = ?")
            parameters.append(event_type)
        if since is not None:
            clauses.append("occurred_at >= ?")
            parameters.append(since)
        if until is not None:
            clauses.append("occurred_at <= ?")
            parameters.append(until)
        parameters.append(limit)
        sql = (
            "SELECT event_id, trip_id, event_type, occurred_at, source, payload_json "
            "FROM (SELECT event_id, trip_id, event_type, occurred_at, source, payload_json "
            "FROM trip_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY occurred_at DESC, event_id DESC LIMIT ?) "
            "ORDER BY occurred_at ASC, event_id ASC"
        )
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [
            TripEvent(
                event_id=row["event_id"],
                trip_id=row["trip_id"],
                event_type=row["event_type"],
                occurred_at=row["occurred_at"],
                source=row["source"],
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        ]

    def count(
        self,
        *,
        trip_id: str,
        event_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> int:
        if not trip_id.strip():
            raise ValueError("trip_id must not be empty")
        _validate_time_range(since, until)
        clauses = ["trip_id = ?"]
        parameters: list[object] = [trip_id]
        if event_type is not None:
            clauses.append("event_type = ?")
            parameters.append(event_type)
        if since is not None:
            clauses.append("occurred_at >= ?")
            parameters.append(since)
        if until is not None:
            clauses.append("occurred_at <= ?")
            parameters.append(until)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM trip_events WHERE "
                + " AND ".join(clauses),
                parameters,
            ).fetchone()
        return int(row["total"])

    def prune(self, *, before: float) -> int:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM trip_events WHERE occurred_at < ?", (before,)
            )
            return cursor.rowcount

    def clear_trip(self, trip_id: str) -> int:
        if not trip_id.strip():
            raise ValueError("trip_id must not be empty")
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM trip_events WHERE trip_id = ?", (trip_id,)
            )
            return cursor.rowcount
