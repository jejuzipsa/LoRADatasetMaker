from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .cropper import make_head_crop
from .domain import DatasetStatus, ImageRecord, Rect


class FaceAnalyzer:
    def __init__(self) -> None:
        self.frontal = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        self.profile = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_profileface.xml")
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

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        frontal_faces = self.frontal.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        profile_faces = self.profile.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        flipped = cv2.flip(gray, 1)
        flipped_profiles = self.profile.detectMultiScale(
            flipped, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        candidates: list[tuple[Rect, str]] = []
        for x, y, w, h in frontal_faces:
            candidates.append((Rect(int(x), int(y), int(w), int(h)), "front"))
        for x, y, w, h in profile_faces:
            candidates.append((Rect(int(x), int(y), int(w), int(h)), "left_profile"))

        width = gray.shape[1]
        for x, y, w, h in flipped_profiles:
            real_x = width - int(x) - int(w)
            candidates.append(
                (Rect(real_x, int(y), int(w), int(h)), "right_profile")
            )

        record.detected_faces_count = len(candidates)
        record.auto_reasons = []
        record.face_box = None
        record.crop_box = None

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
