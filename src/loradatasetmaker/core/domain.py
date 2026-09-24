from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DatasetStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REVIEW = "REVIEW"
    REJECTED = "REJECTED"


@dataclass(slots=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass(slots=True)
class ImageRecord:
    source_file: Path
    auto_status: DatasetStatus = DatasetStatus.REVIEW
    final_status: DatasetStatus = DatasetStatus.REVIEW
    auto_reasons: list[str] = field(default_factory=list)
    user_override: bool = False
    face_box: Rect | None = None
    crop_box: Rect | None = None
    detected_faces_count: int = 0
    detection_confidence: float | None = None
    direction_caption: str = ""
    caption: str = ""
    identity_similarity: float | None = None
    is_reference: bool = False

    @property
    def display_name(self) -> str:
        return self.source_file.name

    def set_final_status(self, status: DatasetStatus) -> None:
        self.final_status = status
        self.user_override = status != self.auto_status

    def reset_user_override(self) -> None:
        self.final_status = self.auto_status
        self.user_override = False

    def remove_reason_prefix(self, prefixes: tuple[str, ...]) -> None:
        self.auto_reasons = [
            reason
            for reason in self.auto_reasons
            if not reason.startswith(prefixes)
        ]
