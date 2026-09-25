from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from modules.vehicle_ai.context import (
    ContextManager,
    NavigationState,
)

from modules.vehicle_ai.tools.base import (
    ToolDefinition,
    ToolResult,
)


# ============================================================
# Mock POI database
# ============================================================


MOCK_REST_AREAS = [
    {
        "poi_id": "rest_area_001",
        "name": "West Lake Rest Area",
        "aliases": ["西湖服务区", "西湖休息区"],
        "distance_km": 6.8,
        "eta_minutes": 8,
    },
    {
        "poi_id": "rest_area_002",
        "name": "Riverside Service Area",
        "aliases": ["河滨服务区"],
        "distance_km": 12.4,
        "eta_minutes": 15,
    },
]


@dataclass(frozen=True)
class NavigationConfig:
    """Injectable simulated POIs and deterministic faults for replay tests."""

    poi_catalog: Sequence[Mapping[str, Any]] | None = None
    unavailable_poi_ids: frozenset[str] = frozenset()
    transient_search_failures: int = 0
    max_candidates: int = 3


# ============================================================
# Navigation Tools
# ============================================================


class NavigationTools:
    def __init__(
        self,
        context_manager: ContextManager,
        *,
        poi_catalog: Sequence[Mapping[str, Any]] | None = None,
        unavailable_poi_ids: frozenset[str] = frozenset(),
        transient_search_failures: int = 0,
        max_candidates: int = 3,
    ):
        self.context_manager = context_manager
        catalog = MOCK_REST_AREAS if poi_catalog is None else poi_catalog
        self.poi_catalog = tuple(deepcopy(dict(item)) for item in catalog)
        self.unavailable_poi_ids = unavailable_poi_ids
        if type(transient_search_failures) is not int or transient_search_failures < 0:
            raise ValueError("transient_search_failures must be a non-negative integer")
        self._transient_search_failures = transient_search_failures
        if type(max_candidates) is not int or max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer")
        self.max_candidates = max_candidates

    # ========================================================
    # POI lookup
    # ========================================================

    def _find_poi(
        self,
        poi_id: str,
    ) -> dict | None:

        for poi in self.poi_catalog:
            if poi["poi_id"] == poi_id:
                resolved = deepcopy(poi)
                aliases = resolved.get("aliases", ())
                resolved["display_name_zh"] = next(
                    (str(alias) for alias in aliases if str(alias).strip()),
                    str(resolved.get("name", "休息地点")),
                )
                return resolved

        return None

    # ========================================================
    # Search nearby rest area
    # ========================================================

    def search_nearby_rest_area(
        self,
        max_distance_km: float | None = None,
        preferred_area: str | None = None,
    ) -> ToolResult:
        """
        Mock POI search.

        IMPORTANT:

        poi_id is the canonical execution identity.

        name is only a human-readable display label.
        """

        if max_distance_km is not None and (
            isinstance(max_distance_km, bool)
            or not isinstance(max_distance_km, int | float)
            or not math.isfinite(max_distance_km)
            or max_distance_km <= 0
        ):
            return ToolResult(
                False,
                "最大距离必须为正数。",
                error="INVALID_SEARCH_FILTER",
                data={"candidates": [], "simulated": True},
            )
        if preferred_area is not None and not isinstance(preferred_area, str):
            return ToolResult(
                False,
                "区域偏好必须为文本。",
                error="INVALID_SEARCH_FILTER",
                data={"candidates": [], "simulated": True},
            )

        if self._transient_search_failures:
            self._transient_search_failures -= 1
            return ToolResult(
                False,
                "模拟地点查询暂时失败；允许进行一次只读重试。",
                error="TRANSIENT_ERROR",
                data={"retryable": True, "simulated": True},
            )

        preference = preferred_area.strip().casefold() if preferred_area else ""
        matches = []
        for poi in self.poi_catalog:
            name = str(poi.get("name", ""))
            aliases = poi.get("aliases", ())
            labels = (name, *(str(alias) for alias in aliases))
            if max_distance_km is not None and poi["distance_km"] > max_distance_km:
                continue
            if preference and not any(
                preference in label.casefold() for label in labels
            ):
                continue
            candidate = deepcopy(poi)
            candidate["display_name_zh"] = next(
                (str(alias) for alias in aliases if str(alias).strip()), name
            )
            candidate["available_at_search"] = True
            candidate["simulated"] = True
            matches.append(candidate)
        candidates = sorted(matches, key=lambda item: item["distance_km"])[
            : self.max_candidates
        ]
        if not candidates:
            return ToolResult(
                False,
                "没有找到符合条件的模拟休息地点。",
                error="NO_RESULTS",
                data={"candidates": [], "simulated": True},
            )
        return ToolResult(
            True,
            "已找到模拟休息地点候选；导航前仍需明确确认。",
            data={
                **deepcopy(candidates[0]),
                "candidates": candidates,
                "candidate_count": len(candidates),
                "simulated": True,
            },
        )

    # ========================================================
    # Start navigation
    # ========================================================

    def start_navigation(
        self,
        poi_id: str,
    ) -> ToolResult:
        """
        Start navigation using a canonical POI identifier.

        Do NOT accept a free-form destination generated by
        the LLM.
        """

        poi_id = poi_id.strip()

        if not poi_id:
            return ToolResult(
                success=False,
                message=("POI identifier cannot be empty."),
                error="EMPTY_POI_ID",
            )

        poi = self._find_poi(poi_id)

        if poi is None:
            return ToolResult(
                success=False,
                message=("The requested POI could not be resolved."),
                error="UNKNOWN_POI",
                data={
                    "poi_id": poi_id,
                },
            )

        if poi_id in self.unavailable_poi_ids:
            return ToolResult(
                False,
                "该模拟地点在确认后不可用；可重新选择其他候选地点。",
                error="POI_UNAVAILABLE",
                data={
                    "poi_id": poi_id,
                    "recoverable": True,
                    "simulated": True,
                },
            )

        self.context_manager.update_vehicle(
            navigation_destination_id=(poi["poi_id"]),
            navigation_destination=(poi["name"]),
            navigation_state=(NavigationState.ACTIVE),
        )

        return ToolResult(
            success=True,
            message=(
                f"模拟导航已开始，目的地：{poi.get('display_name_zh', poi['name'])}。"
            ),
            data={
                "poi_id": poi["poi_id"],
                "destination": poi["name"],
                "destination_display_zh": poi.get("display_name_zh", poi["name"]),
                "distance_km": poi["distance_km"],
                "eta_minutes": poi["eta_minutes"],
                "navigation_state": NavigationState.ACTIVE,
            },
        )

    # ========================================================
    # Cancel navigation
    # ========================================================

    def cancel_navigation(
        self,
    ) -> ToolResult:

        self.context_manager.update_vehicle(
            navigation_destination_id=None,
            navigation_destination=None,
            navigation_state=(NavigationState.IDLE),
        )

        return ToolResult(
            success=True,
            message=("Navigation cancelled."),
            data={
                "navigation_state": NavigationState.IDLE,
            },
        )


# ============================================================
# Registration
# ============================================================


def build_navigation_tools(
    context_manager: ContextManager,
    *,
    poi_catalog: Sequence[Mapping[str, Any]] | None = None,
    unavailable_poi_ids: frozenset[str] = frozenset(),
    transient_search_failures: int = 0,
    max_candidates: int = 3,
    config: NavigationConfig | None = None,
) -> list[ToolDefinition]:

    if config is not None:
        poi_catalog = config.poi_catalog
        unavailable_poi_ids = config.unavailable_poi_ids
        transient_search_failures = config.transient_search_failures
        max_candidates = config.max_candidates

    tools = NavigationTools(
        context_manager,
        poi_catalog=poi_catalog,
        unavailable_poi_ids=unavailable_poi_ids,
        transient_search_failures=transient_search_failures,
        max_candidates=max_candidates,
    )

    return [
        ToolDefinition(
            name=("search_nearby_rest_area"),
            description=(
                "查询周边模拟休息地点，最多返回三个候选；结果含规范 poi_id、中文名称、距离和预计时间。"
                "只能使用本次 candidates 中的精确 poi_id；默认按距离从近到远。"
                "这是模拟目录，不是实时地图或实时可用性数据。"
            ),
            category="navigation",
            parameters={
                "type": "object",
                "properties": {
                    "max_distance_km": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                    },
                    "preferred_area": {
                        "type": "string",
                        "description": "可选区域偏好，例如服务区别名",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=(tools.search_nearby_rest_area),
        ),
        ToolDefinition(
            name="start_navigation",
            description=(
                "对搜索结果中的规范地点启动模拟导航。必须使用搜索候选或有效待确认动作提供的精确 poi_id；"
                "不得翻译、改写或编造 ID。此动作必须经过单次用户确认。"
            ),
            category="navigation",
            parameters={
                "type": "object",
                "properties": {
                    "poi_id": {
                        "type": "string",
                        "description": "搜索候选或待确认动作返回的规范地点 ID。",
                    },
                },
                "required": ["poi_id"],
                "additionalProperties": False,
            },
            handler=(tools.start_navigation),
            requires_confirmation=True,
        ),
        ToolDefinition(
            name="cancel_navigation",
            description=("Cancel the active navigation session."),
            category="navigation",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            handler=(tools.cancel_navigation),
            requires_confirmation=True,
        ),
    ]
