from dataclasses import dataclass


@dataclass
class RestStopRecommendation:
    name: str
    distance_km: float
    eta_minutes: int


class MockRestAdvisor:
    """
    Mock recommendation provider used by the Cabin Demo.

    This module intentionally contains no real map service yet.
    It will later be replaced by an OpenStreetMap/navigation adapter.
    """

    def recommend(self) -> RestStopRecommendation:

        return RestStopRecommendation(
            name="Nearby Rest Area",
            distance_km=6.8,
            eta_minutes=8,
        )

