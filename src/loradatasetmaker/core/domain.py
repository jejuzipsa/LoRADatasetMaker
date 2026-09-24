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

    blur_score: float | None = None
    face_pixel_size: int | None = None
    face_size_ratio: float | None = None
    exposure_mean: float | None = None
    crop_touches_edge: bool = False
    quality_score: float | None = None
    quality_status: DatasetStatus | None = None

    vision_status: DatasetStatus | None = None
    vision_usable_for_lora: str = ""
    vision_hair_visible: str = ""
    vision_occlusion: str = ""
    vision_blur: str = ""
    vision_crop_quality: str = ""
    vision_notes: str = ""
    vision_reasons: list[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.source_file.name

    def set_final_status(self, status: DatasetStatus) -> None:
        self.final_status = status
        self.user_override = status != self.auto_status

    def apply_auto_status(self, status: DatasetStatus) -> None:
        self.auto_status = status
        if not self.user_override:
            self.final_status = status

    def reset_user_override(self) -> None:
        self.final_status = self.auto_status
        self.user_override = False

    def clear_vision_review(self) -> None:
        self.remove_reason_prefix(("vision_",))
        self.vision_status = None
        self.vision_usable_for_lora = ""
        self.vision_hair_visible = ""
        self.vision_occlusion = ""
        self.vision_blur = ""
        self.vision_crop_quality = ""
        self.vision_notes = ""
        self.vision_reasons = []

    def remove_reason_prefix(self, prefixes: tuple[str, ...]) -> None:
        self.auto_reasons = [
            reason
            for reason in self.auto_reasons
            if not reason.startswith(prefixes)
        ]
