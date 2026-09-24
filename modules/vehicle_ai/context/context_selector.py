from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from collections.abc import Callable

from modules.vehicle_ai.context.models import (
    VehicleContext,
)
from modules.vehicle_ai.context.quality import QualityStatus


# ============================================================
# Context Topic
# ============================================================


class ContextTopic(StrEnum):
    """
    Semantic context domains relevant to one user request.
    """

    DRIVER = "driver"
    ROAD = "road"
    CLIMATE = "climate"
    MEDIA = "media"
    NAVIGATION = "navigation"
    VEHICLE = "vehicle"


# ============================================================
# Selection Result
# ============================================================


@dataclass
class ContextSelection:
    """
    Result produced by ContextSelector.
    """

    topics: list[ContextTopic]

    context: dict[str, Any]

    matched_keywords: dict[str, list[str]]

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "topics": [topic.value for topic in self.topics],
            "context": self.context,
            "matched_keywords": self.matched_keywords,
        }


# ============================================================
# Context Selector
# ============================================================


class ContextSelector:
    """
    Select only context relevant to the current user request.

    Version 1 deliberately uses deterministic semantic rules.

    We do NOT use another LLM for context routing yet because:

        1. routing should be fast;
        2. routing should be deterministic;
        3. it avoids an additional model request;
        4. the current vehicle domain is small and structured.

    Later this selector can be upgraded to hybrid routing.
    """

    # ========================================================
    # Keyword groups
    # ========================================================

    DRIVER_KEYWORDS = {
        "困",
        "困了",
        "疲劳",
        "累",
        "累了",
        "休息",
        "清醒",
        "精神",
        "眼睛",
        "打哈欠",
        "犯困",
        "眨眼",
        "驾驶状态",
        "驾驶员状态",
        "驾驶员",
        "舱内",
        "分心",
        "注意力",
        "drowsy",
        "tired",
        "fatigue",
        "sleepy",
        "rest",
    }

    CLIMATE_KEYWORDS = {
        "热",
        "冷",
        "温度",
        "空调",
        "暖气",
        "制冷",
        "制热",
        "凉快",
        "暖和",
        "闷",
        "通风",
        "车窗",
        "窗户",
        "temperature",
        "climate",
        "ac",
        "air conditioning",
        "hot",
        "cold",
        "window",
    }

    MEDIA_KEYWORDS = {
        "音乐",
        "歌",
        "歌曲",
        "播放",
        "暂停",
        "声音",
        "音量",
        "静音",
        "歌单",
        "专辑",
        "歌手",
        "music",
        "song",
        "play",
        "pause",
        "volume",
        "mute",
    }

    NAVIGATION_KEYWORDS = {
        "导航",
        "路线",
        "目的地",
        "去哪",
        "怎么走",
        "带我去",
        "开过去",
        "过去",
        "休息区",
        "服务区",
        "停车区",
        "附近",
        "找个地方",
        "导航过去",
        "到达",
        "距离",
        "还有多远",
        "navigation",
        "navigate",
        "route",
        "destination",
        "nearby",
        "rest area",
        "service area",
    }

    ROAD_KEYWORDS = {
        "路况",
        "道路",
        "前面",
        "车辆",
        "行人",
        "车多",
        "堵车",
        "拥堵",
        "交通",
        "车道",
        "可行驶区域",
        "红绿灯",
        "交通灯",
        "交通标志",
        "road",
        "traffic",
        "lane",
        "pedestrian",
        "vehicle ahead",
        "congestion",
    }

    VEHICLE_KEYWORDS = {
        "车速",
        "速度",
        "挡位",
        "档位",
        "车辆状态",
        "现在多快",
        "开多快",
        "speed",
        "gear",
        "vehicle status",
    }

    # ========================================================
    # Detect topics
    # ========================================================

    def detect_topics(
        self,
        user_text: str,
    ) -> tuple[
        set[ContextTopic],
        dict[str, list[str]],
    ]:

        text = user_text.strip().lower()

        topics: set[ContextTopic] = set()

        matched: dict[
            str,
            list[str],
        ] = {}

        groups = {
            ContextTopic.DRIVER: self.DRIVER_KEYWORDS,
            ContextTopic.CLIMATE: self.CLIMATE_KEYWORDS,
            ContextTopic.MEDIA: self.MEDIA_KEYWORDS,
            ContextTopic.NAVIGATION: self.NAVIGATION_KEYWORDS,
            ContextTopic.ROAD: self.ROAD_KEYWORDS,
            ContextTopic.VEHICLE: self.VEHICLE_KEYWORDS,
        }

        for topic, keywords in groups.items():
            hits = [keyword for keyword in keywords if keyword in text]

            if hits:
                topics.add(topic)

                matched[topic.value] = sorted(hits)

        # ----------------------------------------------------
        # Domain coupling rules
        # ----------------------------------------------------

        if topics & {ContextTopic.MEDIA, ContextTopic.NAVIGATION, ContextTopic.DRIVER}:
            topics.update({ContextTopic.DRIVER, ContextTopic.ROAD, ContextTopic.VEHICLE})

        # Driver fatigue questions frequently involve whether
        # the vehicle is moving.
        if ContextTopic.DRIVER in topics:
            topics.add(ContextTopic.VEHICLE)

        # Navigation decisions should know whether the vehicle
        # is moving and current navigation state.
        if ContextTopic.NAVIGATION in topics:
            topics.add(ContextTopic.VEHICLE)

        # Road questions also benefit from speed / gear.
        if ContextTopic.ROAD in topics:
            topics.add(ContextTopic.VEHICLE)

        return (
            topics,
            matched,
        )

    # ========================================================
    # Build selected context
    # ========================================================

    def select(
        self,
        user_text: str,
        vehicle_context: VehicleContext,
        road_quality: QualityStatus | None = None,
        driver_quality: QualityStatus | None = None,
        vehicle_quality: QualityStatus | None = None,
        field_quality: Callable[[str, str], QualityStatus] | None = None,
    ) -> ContextSelection:

        topics, matched = self.detect_topics(user_text)

        selected: dict[
            str,
            Any,
        ] = {}

        def observed(domain: str, field: str) -> bool:
            return (
                field_quality is None
                or field_quality(domain, field) is QualityStatus.KNOWN
            )

        def include(
            domain: str, values: dict[str, Any], names: dict[str, str] | None = None
        ) -> dict[str, Any]:
            return {
                key: value
                for key, value in values.items()
                if observed(domain, (names or {}).get(key, key))
            }

        # ----------------------------------------------------
        # Driver
        # ----------------------------------------------------

        if ContextTopic.DRIVER in topics:
            selected["driver"] = {}
            if driver_quality is not None:
                selected["driver"]["quality_status"] = driver_quality
            if driver_quality is None or driver_quality is QualityStatus.KNOWN:
                selected["driver"].update(
                    include(
                        "driver",
                        {
                            "presence": vehicle_context.driver.presence,
                            "state": vehicle_context.driver.state,
                            "risk": vehicle_context.driver.risk,
                            "perclos": vehicle_context.driver.perclos,
                            "eye_closed": vehicle_context.driver.eye_closed,
                            "eye_closure_seconds": vehicle_context.driver.eye_closure_seconds,
                            "recent_yawns": vehicle_context.driver.recent_yawns,
                        },
                    )
                )

        # ----------------------------------------------------
        # Climate
        # ----------------------------------------------------

        if ContextTopic.CLIMATE in topics:
            selected["climate"] = {}
            if vehicle_quality is not None:
                selected["climate"]["quality_status"] = vehicle_quality
            if vehicle_quality is None or vehicle_quality is QualityStatus.KNOWN:
                selected["climate"].update(
                    include(
                        "vehicle",
                        {
                            "cabin_temperature_c": vehicle_context.vehicle.cabin_temperature_c,
                            "target_temperature_c": vehicle_context.vehicle.target_temperature_c,
                            "ac_enabled": vehicle_context.vehicle.ac_enabled,
                            "driver_window_open": vehicle_context.vehicle.driver_window_open,
                            "passenger_window_open": vehicle_context.vehicle.passenger_window_open,
                        },
                    )
                )

        # ----------------------------------------------------
        # Media
        # ----------------------------------------------------

        if ContextTopic.MEDIA in topics:
            selected["media"] = {}
            if vehicle_quality is not None:
                selected["media"]["quality_status"] = vehicle_quality
            if vehicle_quality is None or vehicle_quality is QualityStatus.KNOWN:
                selected["media"].update(
                    include(
                        "vehicle",
                        {
                            "media_playing": vehicle_context.vehicle.media_playing,
                            "media_title": vehicle_context.vehicle.media_title,
                            "volume": vehicle_context.vehicle.volume,
                        },
                    )
                )

        # ----------------------------------------------------
        # Navigation
        # ----------------------------------------------------

        if ContextTopic.NAVIGATION in topics:
            selected["navigation"] = {}
            if vehicle_quality is not None:
                selected["navigation"]["quality_status"] = vehicle_quality
            if vehicle_quality is None or vehicle_quality is QualityStatus.KNOWN:
                selected["navigation"].update(
                    include(
                        "vehicle",
                        {
                            "state": vehicle_context.vehicle.navigation_state,
                            "destination_id": vehicle_context.vehicle.navigation_destination_id,
                            "destination": vehicle_context.vehicle.navigation_destination,
                        },
                        {
                            "state": "navigation_state",
                            "destination_id": "navigation_destination_id",
                            "destination": "navigation_destination",
                        },
                    )
                )

        # ----------------------------------------------------
        # Road
        # ----------------------------------------------------

        if ContextTopic.ROAD in topics:
            selected["road"] = {}
            if road_quality is not None:
                selected["road"]["quality_status"] = road_quality
            if road_quality is None or road_quality is QualityStatus.KNOWN:
                selected["road"].update(
                    include(
                        "road",
                        {
                            "vehicle_count": vehicle_context.road.vehicle_count,
                            "pedestrian_count": vehicle_context.road.pedestrian_count,
                            "rider_count": vehicle_context.road.rider_count,
                            "traffic_light_count": vehicle_context.road.traffic_light_count,
                            "traffic_sign_count": vehicle_context.road.traffic_sign_count,
                            "lane_detected": vehicle_context.road.lane_detected,
                            "drivable_area_detected": vehicle_context.road.drivable_area_detected,
                            "traffic_level": vehicle_context.road.traffic_level,
                        },
                    )
                )

        # ----------------------------------------------------
        # Vehicle
        # ----------------------------------------------------

        if ContextTopic.VEHICLE in topics:
            selected["vehicle"] = {}
            if vehicle_quality is not None:
                selected["vehicle"]["quality_status"] = vehicle_quality
            if vehicle_quality is None or vehicle_quality is QualityStatus.KNOWN:
                selected["vehicle"].update(
                    include(
                        "vehicle",
                        {
                            "speed_kmh": vehicle_context.vehicle.speed_kmh,
                            "gear": vehicle_context.vehicle.gear,
                        },
                    )
                )

        # ----------------------------------------------------
        # Generic conversation
        #
        # For "你好", "你是谁", etc. we intentionally send
        # no dynamic vehicle context.
        # ----------------------------------------------------

        return ContextSelection(
            topics=sorted(
                topics,
                key=lambda topic: topic.value,
            ),
            context=selected,
            matched_keywords=matched,
        )
