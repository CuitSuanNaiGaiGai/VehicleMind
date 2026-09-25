from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime


def test_default_offline_runtime_never_loads_lightrag_service() -> None:
    runtime = VehicleMindRuntime(ScriptedLLMClient((ScriptedResponse(content="你好"),)))
    assert "search_vehicle_knowledge" not in runtime.tools.names()
    assert runtime.agent.chat("你好") == "你好"
