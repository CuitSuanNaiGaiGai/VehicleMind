from __future__ import annotations

from modules.vehicle_ai.context import (
    ContextManager,
)

from modules.vehicle_ai.tools.base import (
    ToolDefinition,
    ToolResult,
)


# ============================================================
# Media Tools
# ============================================================


class MediaTools:
    def __init__(
        self,
        context_manager: ContextManager,
    ):
        self.context_manager = context_manager

    def get_media_status(self) -> ToolResult:
        vehicle = self.context_manager.get_context().vehicle
        return ToolResult(
            True,
            "Media status retrieved.",
            data={
                "media_playing": vehicle.media_playing,
                "media_title": vehicle.media_title,
                "volume": vehicle.volume,
            },
        )

    # ========================================================
    # Play music
    # ========================================================

    def play_music(
        self,
        query: str,
    ) -> ToolResult:

        query = query.strip()

        if not query:
            return ToolResult(
                success=False,
                message=("Music query cannot be empty."),
                error="EMPTY_QUERY",
            )

        self.context_manager.update_vehicle(
            media_playing=True,
            media_title=query,
        )

        return ToolResult(
            success=True,
            message=(f"Playing '{query}'."),
            data={
                "media_playing": True,
                "media_title": query,
            },
        )

    # ========================================================
    # Pause
    # ========================================================

    def pause_music(
        self,
    ) -> ToolResult:

        self.context_manager.update_vehicle(media_playing=False)

        return ToolResult(
            success=True,
            message="Media paused.",
            data={
                "media_playing": False,
            },
        )

    # ========================================================
    # Volume
    # ========================================================

    def set_volume(
        self,
        volume: int,
    ) -> ToolResult:

        volume = int(volume)

        if not (0 <= volume <= 100):
            return ToolResult(
                success=False,
                message=("Volume must be between 0 and 100."),
                error=("VOLUME_OUT_OF_RANGE"),
            )

        self.context_manager.update_vehicle(volume=volume)

        return ToolResult(
            success=True,
            message=(f"Volume set to {volume}."),
            data={
                "volume": volume,
            },
        )


# ============================================================
# Registration
# ============================================================


def build_media_tools(
    context_manager: ContextManager,
) -> list[ToolDefinition]:

    tools = MediaTools(context_manager)

    return [
        ToolDefinition(
            name="get_media_status",
            description="Get current media playback and volume status.",
            category="media",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            handler=tools.get_media_status,
        ),
        ToolDefinition(
            name="play_music",
            description=("Play music matching the user's request."),
            category="media",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": ("Song, artist, playlist or music description."),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=(tools.play_music),
        ),
        ToolDefinition(
            name="pause_music",
            description=("Pause the currently playing media."),
            category="media",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            handler=(tools.pause_music),
        ),
        ToolDefinition(
            name="set_volume",
            description=("Set media volume from 0 to 100."),
            category="media",
            parameters={
                "type": "object",
                "properties": {
                    "volume": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "required": ["volume"],
                "additionalProperties": False,
            },
            handler=(tools.set_volume),
        ),
    ]
