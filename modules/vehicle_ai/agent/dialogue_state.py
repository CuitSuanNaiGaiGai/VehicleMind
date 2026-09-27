"""Small immutable records for state shared across dialogue turns."""

from dataclasses import dataclass, field
from uuid import uuid4


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
