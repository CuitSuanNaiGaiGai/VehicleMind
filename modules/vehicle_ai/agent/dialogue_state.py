"""Small immutable records for state shared across dialogue turns."""

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4


class DialogueIntent(StrEnum):
    START = "START"
    UPDATE_CONSTRAINTS = "UPDATE_CONSTRAINTS"
    SELECT = "SELECT"
    ASK_CANDIDATE = "ASK_CANDIDATE"
    SIDE_QUESTION = "SIDE_QUESTION"
    RESUME = "RESUME"
    CANCEL = "CANCEL"
    UNCLEAR = "UNCLEAR"


@dataclass(frozen=True)
class ConstraintValue:
    value: float | str
    source_turn_id: int
    evidence: str


@dataclass(frozen=True)
class CandidateSnapshot:
    candidate_set_id: str
    constraint_revision: int
    created_at: float
    presented_turn_id: int | None
    expires_at: float
    presented_poi_ids: tuple[str, ...]


@dataclass
class DialogueSession:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    turn_id: int = 0

    def next_turn(self) -> int:
        self.turn_id += 1
        return self.turn_id
