"""Deterministic LangGraph approval/resume demo; no API key required."""

from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime


def main() -> None:
    llm = ScriptedLLMClient(
        responses=(
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        id="open-window",
                        name="set_driver_window",
                        arguments={"open": True},
                        arguments_json='{"open": true}',
                    ),
                ),
            ),
        )
    )
    runtime = VehicleMindRuntime(llm=llm)

    waiting = runtime.chat_stateful("打开驾驶员车窗", debug=False)
    print("=== interrupted ===")
    print(waiting.to_dict())
    print("vehicle.driver_window_open =", runtime.context_summary())

    completed = runtime.resume_stateful("approve")
    print("\n=== resumed ===")
    print(completed.to_dict())
    print("vehicle.driver_window_open =", runtime.context_summary())


if __name__ == "__main__":
    main()
