from __future__ import annotations

from apps.vehicle_ai_demo.trip_memory_display import handle_trip_memory_command
from modules.vehicle_ai.memory import TripEvent, TripEventStore


def test_clear_history_requires_exact_trip_id_confirmation(
    tmp_path, monkeypatch, capsys
) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    store.append(TripEvent("e1", "trip-a", "REMINDER", 1, "agent", {}))
    answers = iter(("wrong", "trip-a"))
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert handle_trip_memory_command("clear-history", store, "trip-a")
    assert store.count(trip_id="trip-a") == 1
    assert "未清空" in capsys.readouterr().out

    assert handle_trip_memory_command("clear-history", store, "trip-a")
    assert store.count(trip_id="trip-a") == 0
    assert "已清空" in capsys.readouterr().out
