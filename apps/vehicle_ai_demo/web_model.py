"""显式的离线演示模型；规则生成工具调用，不冒充在线语言模型。"""

import json
import re
import time
from uuid import uuid4

from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse, LLMToolCall


class WebDemoModel(BaseLLMClient):
    def __init__(self, delay_seconds: float = 0.6):
        self.delay_seconds = delay_seconds

    def _call(self, name: str, arguments: dict) -> LLMResponse:
        return LLMResponse(
            None,
            [
                LLMToolCall(
                    uuid4().hex,
                    name,
                    arguments,
                    json.dumps(arguments, ensure_ascii=False),
                )
            ],
            response_model="scripted-web",
        )

    def chat(self, messages, tools=None):
        time.sleep(self.delay_seconds)
        index = max(i for i, item in enumerate(messages) if item["role"] == "user")
        text = messages[index]["content"]
        results = [item for item in messages[index + 1 :] if item["role"] == "tool"]
        if results:
            result = json.loads(results[-1]["content"])
            if result.get("success") and "服务区" in text and len(results) == 1:
                return self._call(
                    "start_navigation", {"poi_id": result["data"]["poi_id"]}
                )
            return LLMResponse(
                result.get("message", "工具返回结果，请查看右侧执行详情。"),
                [],
                response_model="scripted-web",
            )
        if "服务区" in text:
            return self._call("search_nearby_rest_area", {})
        if "车窗" in text and any(word in text for word in ("打开", "关闭")):
            return self._call("set_driver_window", {"open": "打开" in text})
        if "温度" in text:
            number = re.search(r"(-?\d+(?:\.\d+)?)", text)
            if number:
                return self._call(
                    "set_temperature", {"temperature_c": float(number.group(1))}
                )
            return self._call("get_climate_status", {})
        if "音量" in text:
            number = re.search(r"(-?\d+)", text)
            if number:
                return self._call("set_volume", {"volume": int(number.group(1))})
            return self._call("get_media_status", {})
        if "暂停" in text and "音乐" in text:
            return self._call("pause_music", {})
        if "状态" in text:
            return self._call("get_vehicle_status", {})
        return LLMResponse(
            "离线脚本支持：查找服务区并导航、打开或关闭车窗、查询车辆状态、设置温度或音量。其他请求请显式选择 Qwen 或 GLM。",
            [],
            response_model="scripted-web",
        )
