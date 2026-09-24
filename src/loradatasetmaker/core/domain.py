from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DatasetStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REVIEW = "REVIEW"
    REJECTED = "REJECTED"


@dataclass(slots=True)
class ImageRecord:
    source_file: Path
    auto_status: DatasetStatus = DatasetStatus.REVIEW
    final_status: DatasetStatus = DatasetStatus.REVIEW
    auto_reasons: list[str] = field(default_factory=list)
    user_override: bool = False
    identity_similarity: float | None = None
    quality_score: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    direction_caption: str = ""
    caption: str = ""

    @property
    def display_name(self) -> str:
        return self.source_file.name

    def set_final_status(self, status: DatasetStatus) -> None:
        self.final_status = status
        self.user_override = status != self.auto_status
