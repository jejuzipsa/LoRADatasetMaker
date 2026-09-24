from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .cropper import make_head_crop
from .domain import DatasetStatus, ImageRecord, Rect


class FaceAnalyzer:
    def __init__(self, max_detection_size: int = 1280) -> None:
        self.max_detection_size = max_detection_size
        self.frontal = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        self.profile = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_profileface.xml")
        )

        if self.frontal.empty() or self.profile.empty():
            raise RuntimeError(
                "OpenCV 얼굴 검출 데이터(haarcascade)를 불러오지 못했어."
            )

    def analyze(self, record: ImageRecord, trigger_token: str = "") -> ImageRecord:
        image = cv2.imdecode(
            np.fromfile(record.source_file, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            record.auto_status = DatasetStatus.REJECTED
            record.final_status = DatasetStatus.REJECTED
            record.auto_reasons = ["image_load_failed"]
            return record

        detect_image, scale = self._prepare_detection_image(image)
        gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        frontal_faces = self.frontal.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        profile_faces = self.profile.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        flipped = cv2.flip(gray, 1)
        flipped_profiles = self.profile.detectMultiScale(
            flipped, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        candidates: list[tuple[Rect, str]] = []
        for x, y, w, h in frontal_faces:
            candidates.append(
                (self._restore_rect(x, y, w, h, scale), "front")
            )
        for x, y, w, h in profile_faces:
            candidates.append(
                (self._restore_rect(x, y, w, h, scale), "left_profile")
            )

        detect_width = gray.shape[1]
        for x, y, w, h in flipped_profiles:
            real_x = detect_width - int(x) - int(w)
            candidates.append(
                (
                    self._restore_rect(real_x, y, w, h, scale),
                    "right_profile",
                )
            )

        candidates = self._deduplicate_candidates(candidates)

        record.detected_faces_count = len(candidates)
        record.auto_reasons = []
        record.face_box = None
        record.crop_box = None
        record.direction_caption = ""

        if not candidates:
            record.auto_status = DatasetStatus.REJECTED
            record.final_status = DatasetStatus.REJECTED
            record.auto_reasons.append("face_not_detected")
            return record

        best_box, direction = max(candidates, key=lambda item: item[0].area)
        record.face_box = best_box
        record.crop_box = make_head_crop(
            best_box, image.shape[1], image.shape[0]
        )
        record.direction_caption = direction

        if len(candidates) > 1:
            record.auto_reasons.append("multiple_faces_detected")
            record.auto_status = DatasetStatus.REVIEW
            record.final_status = DatasetStatus.REVIEW
        else:
            record.auto_status = DatasetStatus.ACCEPTED
            record.final_status = DatasetStatus.ACCEPTED

        pieces = [trigger_token.strip()] if trigger_token.strip() else []
        pieces.extend(["person", direction.replace("_", " ")])
        record.caption = ", ".join(part for part in pieces if part)
        return record

    def _prepare_detection_image(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        longest = max(width, height)
        if longest <= self.max_detection_size:
            return image, 1.0

        scale = self.max_detection_size / float(longest)
        resized = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    @staticmethod
    def _restore_rect(
        x: int,
        y: int,
        w: int,
        h: int,
        scale: float,
    ) -> Rect:
        inverse = 1.0 / scale
        return Rect(
            int(round(x * inverse)),
            int(round(y * inverse)),
            int(round(w * inverse)),
            int(round(h * inverse)),
        )

    def _deduplicate_candidates(
        self,
        candidates: list[tuple[Rect, str]],
    ) -> list[tuple[Rect, str]]:
        kept: list[tuple[Rect, str]] = []
        for candidate in sorted(
            candidates,
            key=lambda item: item[0].area,
            reverse=True,
        ):
            box, _label = candidate
            if any(self._iou(box, existing[0]) >= 0.45 for existing in kept):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _iou(a: Rect, b: Rect) -> float:
        ax1, ay1 = a.x, a.y
        ax2, ay2 = a.x + a.w, a.y + a.h
        bx1, by1 = b.x, b.y
        bx2, by2 = b.x + b.w, b.y + b.h

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = iw * ih
        if intersection == 0:
            return 0.0

        union = a.area + b.area - intersection
        return intersection / float(union) if union > 0 else 0.0
