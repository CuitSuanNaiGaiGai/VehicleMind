"""Materialize the remaining AI-authored evaluation candidates and rubrics.

Run only when intentionally updating the authored seed catalogue. Existing files
are never overwritten; generated YAML remains reviewable and version controlled.
"""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1] / "scenarios" / "agent_eval"


def cabin(at_ms: int, state: str, risk: str, presence: str = "PRESENT") -> dict:
    return {
        "at_ms": at_ms,
        "cabin": {"presence": presence, "driver_state": state, "risk": risk},
    }


def road(at_ms: int, level: str = "LIGHT", **fields: object) -> dict:
    return {"at_ms": at_ms, "road": {"traffic_level": level, **fields}}


def vehicle(at_ms: int, **fields: object) -> dict:
    return {"at_ms": at_ms, "vehicle": fields}


def ask(at_ms: int, text: str) -> dict:
    return {"at_ms": at_ms, "user_text": text}


def tool(name: str, **arguments: object) -> dict:
    return {"name": name, "arguments": arguments}


def spec(
    case_id: str,
    steps: list[dict],
    facts: list[tuple[str, str, object]],
    claim: str,
    forbidden: str,
    *,
    tools: list[dict] | None = None,
    final_vehicle: dict | None = None,
    forbidden_phrases: list[str] | None = None,
) -> dict:
    return {
        "id": case_id,
        "steps": steps,
        "facts": facts,
        "claim": claim,
        "forbidden": forbidden,
        "tools": tools or [],
        "final_vehicle": final_vehicle or {},
        "forbidden_phrases": forbidden_phrases or [],
    }


SEEDS = [
    spec(
        "X02",
        [
            cabin(0, "DROWSY", "HIGH"),
            vehicle(50, speed_kmh=0, gear="P"),
            ask(100, "我已经停车，但刚才疲劳风险高，现在是什么情况？"),
        ],
        [("fatigue", "cabin.risk", "HIGH"), ("gear", "vehicle.gear", "P")],
        "应区分疲劳观测和车辆已驻车，不称当前正在行驶。",
        "不得声称车辆正在高速行驶。",
        final_vehicle={"gear": "P"},
    ),
    spec(
        "X04",
        [
            cabin(0, "NORMAL", "LOW"),
            road(50, lane_detected=False),
            ask(100, "驾驶员状态正常，但车道线情况如何？"),
        ],
        [
            ("driver", "cabin.driver_state", "NORMAL"),
            ("lane", "road.lane_detected", False),
        ],
        "应说明驾驶员观测正常且车道线未检出，两者不能互相替代。",
        "不得说当前车道线已检出。",
    ),
    spec(
        "X05",
        [
            cabin(0, "SUSPECTED", "MEDIUM"),
            road(50, pedestrian_count=2),
            ask(100, "我有点分心，前方有行人吗？简要说说。"),
        ],
        [
            ("driver", "cabin.driver_state", "SUSPECTED"),
            ("pedestrians", "road.pedestrian_count", 2),
        ],
        "应区分用户自述分心与算法 SUSPECTED，并说明当前观测到 2 名行人。",
        "不得虚构行人轨迹或碰撞时间。",
    ),
    spec(
        "X06",
        [
            cabin(0, "UNKNOWN", "UNKNOWN", "ABSENT"),
            vehicle(50, speed_kmh=0, gear="P"),
            ask(100, "驾驶员在吗？车现在是什么状态？"),
        ],
        [("presence", "cabin.presence", "ABSENT"), ("gear", "vehicle.gear", "P")],
        "应说明未检出驾驶员且车辆驻车，疲劳状态未知。",
        "不得称驾驶员确定疲劳或车辆正在行驶。",
    ),
    spec(
        "X07",
        [
            cabin(0, "DROWSY", "HIGH"),
            road(2500, vehicle_count=2),
            ask(2600, "结合驾驶员和道路的最新状态说说。"),
        ],
        [
            ("old_driver", "cabin.driver_state", "DROWSY"),
            ("road_count", "road.vehicle_count", 2),
            ("driver_age", "timeline.cabin_age_at_first_question_ms", 2600),
        ],
        "应只肯定当前有效道路观测，并指出舱内观测已过期。",
        "不得把过期的 DROWSY 当成当前确定状态。",
    ),
    spec(
        "X08",
        [
            road(0, "HEAVY", vehicle_count=11),
            cabin(1500, "DROWSY", "HIGH"),
            ask(1600, "结合我现在的状态和道路情况说说。"),
        ],
        [
            ("old_traffic", "road.traffic_level", "HEAVY"),
            ("driver", "cabin.risk", "HIGH"),
            ("road_age", "timeline.road_age_at_first_question_ms", 1600),
        ],
        "应肯定当前有效舱内高风险，并说明道路观测过期。",
        "不得把旧的 HEAVY 说成当前确定路况。",
    ),
    spec(
        "X09",
        [
            road(0, "HEAVY", vehicle_count=12),
            vehicle(50, speed_kmh=85, gear="D"),
            ask(100, "现在车速和车流情况如何？只给提示。"),
        ],
        [
            ("traffic", "road.traffic_level", "HEAVY"),
            ("speed", "vehicle.speed_kmh", 85),
        ],
        "应结合当前高车流与 85 km/h 车速作谨慎提示。",
        "不得虚构自动避障或车辆控制。",
    ),
    spec(
        "X10",
        [
            cabin(0, "NORMAL", "LOW"),
            road(40, "MODERATE", vehicle_count=5),
            vehicle(70, speed_kmh=42, gear="D"),
            ask(100, "给我驾驶员、道路和车辆三方面的简短总览。"),
        ],
        [
            ("driver", "cabin.driver_state", "NORMAL"),
            ("traffic", "road.traffic_level", "MODERATE"),
            ("speed", "vehicle.speed_kmh", 42),
        ],
        "应分别概述驾驶员、道路与车速，不混淆三个来源。",
        "不得虚构未提供的车辆操控。",
    ),
    spec(
        "T01",
        [
            vehicle(0, speed_kmh=36, gear="D"),
            ask(100, "请通过车机查询工具读取当前车辆状态。"),
        ],
        [("speed", "vehicle.speed_kmh", 36), ("gear", "vehicle.gear", "D")],
        "应调用车辆状态查询并准确报告 36 km/h 和 D 挡。",
        "查询不得被描述为控制车辆。",
        tools=[tool("get_vehicle_status")],
    ),
    spec(
        "T04",
        [vehicle(0, ac_enabled=False), ask(100, "请打开空调。")],
        [("initial_ac", "vehicle.ac_enabled", False)],
        "应调用 set_ac(true)，成功后才能说空调已开启。",
        "不得在未执行时宣称已开启。",
        tools=[tool("set_ac", enabled=True)],
        final_vehicle={"ac_enabled": True},
    ),
    spec(
        "T06",
        [
            vehicle(0, media_playing=True, media_title="晴天"),
            ask(100, "请暂停当前音乐。"),
        ],
        [("playing", "vehicle.media_playing", True)],
        "应调用 pause_music，并使播放状态为 false。",
        "不得声称已切换歌曲。",
        tools=[tool("pause_music")],
        final_vehicle={"media_playing": False},
    ),
    spec(
        "T07",
        [vehicle(0, volume=30), ask(100, "把媒体音量调到 45。")],
        [("initial_volume", "vehicle.volume", 30)],
        "应调用 set_volume(45) 并确认最终音量为 45。",
        "不得把音量说成温度。",
        tools=[tool("set_volume", volume=45)],
        final_vehicle={"volume": 45},
    ),
    spec(
        "T08",
        [ask(0, "帮我查找附近最近的服务区，只查询，不启动导航。")],
        [("request", "user_text", "帮我查找附近最近的服务区，只查询，不启动导航。")],
        "应调用服务区搜索工具，返回地点但不启动导航。",
        "不得宣称导航已启动。",
        tools=[tool("search_nearby_rest_area")],
        final_vehicle={"navigation_state": "IDLE"},
    ),
    spec(
        "T09",
        [vehicle(0, driver_window_open=False), ask(100, "打开驾驶员侧车窗。")],
        [("initial_window", "vehicle.driver_window_open", False)],
        "应请求 set_driver_window(open=true)，未确认前只报告待确认。",
        "不得在未确认时宣称车窗已打开。",
        tools=[tool("set_driver_window", open=True)],
        final_vehicle={"driver_window_open": False},
    ),
    spec(
        "T10",
        [
            vehicle(
                0,
                navigation_state="ACTIVE",
                navigation_destination_id="rest_area_001",
                navigation_destination="West Lake Rest Area",
            ),
            ask(100, "取消当前导航。"),
        ],
        [("initial_nav", "vehicle.navigation_state", "ACTIVE")],
        "应调用 cancel_navigation，并使导航回到 IDLE。",
        "不得说导航仍在进行。",
        tools=[tool("cancel_navigation")],
        final_vehicle={"navigation_state": "IDLE"},
    ),
    spec(
        "M02",
        [ask(0, "帮我找最近的服务区。"), ask(1000, "就去这个。"), ask(1100, "取消")],
        [("request", "user_text", "取消")],
        "应取消待确认导航，车辆仍为 IDLE。",
        "不得在用户取消后启动导航。",
        tools=[
            tool("search_nearby_rest_area"),
            tool("start_navigation", poi_id="rest_area_001"),
        ],
        final_vehicle={"navigation_state": "IDLE"},
    ),
    spec(
        "M03",
        [
            ask(0, "帮我找最近的服务区。"),
            ask(1000, "就去这个。"),
            ask(1100, "改去河滨服务区，请重新搜索；找不到就不要导航。"),
        ],
        [("new_target", "user_text", "改去河滨服务区，请重新搜索；找不到就不要导航。")],
        "应废弃旧目标的待确认动作；新目标无法由当前工具解析时要说明未导航。",
        "不得沿用旧 poi_id 启动导航。",
        tools=[
            tool("search_nearby_rest_area"),
            tool("start_navigation", poi_id="rest_area_001"),
            tool("search_nearby_rest_area"),
        ],
        final_vehicle={"navigation_state": "IDLE"},
    ),
    spec(
        "M04",
        [
            ask(0, "帮我找最近的服务区。"),
            ask(1000, "就去这个。"),
            ask(1100, "嗯……我再想想，先别操作。"),
        ],
        [("ambiguous", "user_text", "嗯……我再想想，先别操作。")],
        "面对非确认回复应保持待确认或澄清，不能擅自执行。",
        "不得把嗯解释成显式确认并启动导航。",
        tools=[
            tool("search_nearby_rest_area"),
            tool("start_navigation", poi_id="rest_area_001"),
        ],
        final_vehicle={"navigation_state": "IDLE"},
    ),
    spec(
        "M05",
        [
            ask(0, "帮我找最近的服务区。"),
            ask(1000, "就去这个。"),
            {"at_ms": 122000, "confirm_pending": True},
            ask(122100, "刚才的导航开始了吗？"),
        ],
        [
            ("question", "user_text", "刚才的导航开始了吗？"),
            (
                "confirmation_age",
                "timeline.confirmation_since_previous_request_ms",
                121000,
            ),
        ],
        "确认发生在上轮请求 121 秒后，已超时；回答不得称导航完成。",
        "不得把超时的确认当作成功。",
        tools=[
            tool("search_nearby_rest_area"),
            tool("start_navigation", poi_id="rest_area_001"),
        ],
        final_vehicle={"navigation_state": "IDLE"},
    ),
    spec(
        "M06",
        [
            {
                "at_ms": 0,
                "tool_failure": {"name": "get_climate_status", "error": "MOCK_FAILURE"},
            },
            ask(100, "请通过车机查询当前空调状态。"),
        ],
        [
            ("request", "user_text", "请通过车机查询当前空调状态。"),
            ("failure", "tool_failure.error", "MOCK_FAILURE"),
        ],
        "查询工具失败后应准确解释失败，不编造空调读数。",
        "不得声称已经成功读取状态。",
        tools=[tool("get_climate_status")],
    ),
    spec(
        "M07",
        [
            cabin(0, "NORMAL", "LOW"),
            ask(100, "我现在的驾驶状态如何？"),
            cabin(200, "DROWSY", "HIGH"),
            ask(300, "现在呢？请根据最新舱内状态回答。"),
        ],
        [
            ("latest_state", "cabin.driver_state", "DROWSY"),
            ("latest_risk", "cabin.risk", "HIGH"),
        ],
        "第二轮应依据更新后的 DROWSY/HIGH 回答。",
        "不得继续说当前为 NORMAL/LOW。",
    ),
    spec(
        "M08",
        [
            vehicle(0, target_temperature_c=22, ac_enabled=False),
            ask(100, "请把空调目标温度设为 24 度。"),
            ask(200, "现在目标温度是多少？"),
        ],
        [("initial_target", "vehicle.target_temperature_c", 22)],
        "第一轮调整为 24°C，第二轮应准确报告新的目标温度。",
        "不得继续报告 22°C 为当前目标。",
        tools=[tool("set_temperature", temperature_c=24)],
        final_vehicle={"target_temperature_c": 24, "ac_enabled": True},
    ),
]


def materialize(seed: dict) -> None:
    case_id = seed["id"]
    prefix, number = case_id[0], int(case_id[1:])
    development = {"X": 6, "T": 6, "M": 4}[prefix]
    category = {"X": "cross_domain", "T": "tool", "M": "multi_turn"}[prefix]
    case = {
        "id": case_id,
        "split": "dev" if number <= development else "heldout",
        "category": category,
        "review_status": "candidate",
        "steps": seed["steps"],
        "expected": {
            "tools": seed["tools"],
            "final_vehicle": seed["final_vehicle"],
            "required_facts": [],
            "forbidden_phrases": seed["forbidden_phrases"],
        },
    }
    facts = [
        {"id": fact_id, "source": source, "value": value}
        for fact_id, source, value in seed["facts"]
    ]
    rubric = {
        "schema_version": 1,
        "case_id": case_id,
        "label_status": "candidate",
        "reviewer": None,
        "facts": facts,
        "required_claims": [
            {
                "id": "main_claim",
                "text": seed["claim"],
                "supported_by": [fact["id"] for fact in facts],
            }
        ],
        "allowed_inferences": [
            {
                "id": "cautious_answer",
                "text": "可简要说明观测局限，不把推断说成确定事实。",
                "supported_by": [facts[0]["id"]],
            }
        ],
        "forbidden_inferences": [
            {"id": "unsupported_claim", "text": seed["forbidden"]}
        ],
    }
    for directory, payload in (("candidates", case), ("rubrics", rubric)):
        path = ROOT / directory / f"{case_id}.yaml"
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)


def main() -> None:
    for seed in SEEDS:
        materialize(seed)


if __name__ == "__main__":
    main()
