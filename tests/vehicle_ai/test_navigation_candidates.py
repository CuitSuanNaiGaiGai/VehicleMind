from __future__ import annotations

from modules.vehicle_ai.context import ContextManager, NavigationState
from modules.vehicle_ai.tools.navigation import NavigationTools


def _catalog() -> tuple[dict, ...]:
    return (
        {
            "poi_id": "east_rest",
            "name": "East Rest Area",
            "aliases": ["东湖服务区"],
            "distance_km": 4.0,
            "eta_minutes": 6,
        },
        {
            "poi_id": "river_rest",
            "name": "Riverside Service Area",
            "aliases": ["河滨服务区"],
            "distance_km": 8.5,
            "eta_minutes": 11,
        },
    )


def test_search_returns_bounded_canonical_candidates_sorted_by_distance() -> None:
    tools = NavigationTools(ContextManager(), poi_catalog=_catalog(), max_candidates=1)

    result = tools.search_nearby_rest_area()

    assert result.success is True
    assert result.data["simulated"] is True
    assert [item["poi_id"] for item in result.data["candidates"]] == ["east_rest"]
    assert result.data["poi_id"] == result.data["candidates"][0]["poi_id"]


def test_search_filters_candidates_by_distance_and_localized_preference() -> None:
    tools = NavigationTools(ContextManager(), poi_catalog=_catalog())

    result = tools.search_nearby_rest_area(max_distance_km=10.0, preferred_area="河滨")

    assert [item["poi_id"] for item in result.data["candidates"]] == ["river_rest"]


def test_search_with_no_feasible_candidate_returns_explicit_no_results() -> None:
    tools = NavigationTools(ContextManager(), poi_catalog=_catalog())

    result = tools.search_nearby_rest_area(max_distance_km=1.0)

    assert result.success is False
    assert result.error == "NO_RESULTS"
    assert result.data["candidates"] == []
    assert result.data["simulated"] is True


def test_explicit_empty_catalog_does_not_fall_back_to_default_locations() -> None:
    tools = NavigationTools(ContextManager(), poi_catalog=())

    result = tools.search_nearby_rest_area()

    assert result.success is False
    assert result.error == "NO_RESULTS"
    assert result.data["candidates"] == []


def test_unavailable_poi_fails_without_changing_vehicle_state() -> None:
    context = ContextManager()
    tools = NavigationTools(
        context, poi_catalog=_catalog(), unavailable_poi_ids=frozenset({"east_rest"})
    )

    result = tools.start_navigation("east_rest")

    assert result.success is False
    assert result.error == "POI_UNAVAILABLE"
    assert result.data["recoverable"] is True
    assert context.get_context().vehicle.navigation_state == NavigationState.IDLE


def test_unknown_poi_id_is_not_resolved_or_started() -> None:
    context = ContextManager()
    tools = NavigationTools(context, poi_catalog=_catalog())

    result = tools.start_navigation("invented_rest")

    assert result.success is False
    assert result.error == "UNKNOWN_POI"
    assert context.get_context().vehicle.navigation_state == NavigationState.IDLE


def test_navigation_tools_reject_invalid_search_filters() -> None:
    tools = NavigationTools(ContextManager(), poi_catalog=_catalog())

    invalid_distance = tools.search_nearby_rest_area(max_distance_km=True)
    invalid_preference = tools.search_nearby_rest_area(preferred_area=12)

    assert invalid_distance.error == "INVALID_SEARCH_FILTER"
    assert invalid_preference.error == "INVALID_SEARCH_FILTER"


def test_navigation_rejects_blank_id_and_can_cancel_active_navigation() -> None:
    context = ContextManager()
    tools = NavigationTools(context, poi_catalog=_catalog())

    assert tools.start_navigation(" ").error == "EMPTY_POI_ID"
    assert tools.start_navigation("east_rest").success is True
    cancelled = tools.cancel_navigation()

    assert cancelled.success is True
    assert context.get_context().vehicle.navigation_state == NavigationState.IDLE
