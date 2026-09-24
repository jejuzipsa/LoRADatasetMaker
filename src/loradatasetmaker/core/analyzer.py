from __future__ import annotations

import cv2
import numpy as np

from .cropper import make_head_crop
from .domain import DatasetStatus, ImageRecord, Rect
from .model_manager import ensure_yunet_model


class FaceAnalyzer:
    def __init__(
        self,
        max_detection_size: int = 1280,
        score_threshold: float = 0.90,
        nms_threshold: float = 0.30,
    ) -> None:
        self.max_detection_size = max_detection_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.detector: cv2.FaceDetectorYN | None = None

    def prepare(self) -> None:
        if self.detector is not None:
            return

        model_path = ensure_yunet_model()
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            5000,
        )

    def analyze(self, record: ImageRecord, trigger_token: str = "") -> ImageRecord:
        self.prepare()
        assert self.detector is not None

        image = cv2.imdecode(
            np.fromfile(record.source_file, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            record.apply_auto_status(DatasetStatus.REJECTED)
            record.auto_reasons = ["image_load_failed"]
            record.detection_confidence = None
            return record

        detect_image, scale = self._prepare_detection_image(image)
        height, width = detect_image.shape[:2]
        self.detector.setInputSize((width, height))

        _retval, faces = self.detector.detect(detect_image)

        record.auto_reasons = []
        record.face_box = None
        record.crop_box = None
        record.direction_caption = ""
        record.detection_confidence = None

        if faces is None or len(faces) == 0:
            record.detected_faces_count = 0
            record.apply_auto_status(DatasetStatus.REJECTED)
            record.auto_reasons.append("face_not_detected")
            return record

        candidates = [
            self._candidate_from_row(row, scale)
            for row in faces
            if float(row[-1]) >= self.score_threshold
        ]

        if not candidates:
            record.detected_faces_count = 0
            record.apply_auto_status(DatasetStatus.REJECTED)
            record.auto_reasons.append("face_not_detected")
            return record

        record.detected_faces_count = len(candidates)

        best_box, landmarks, confidence = max(
            candidates,
            key=lambda item: item[0].area * (item[2] ** 2),
        )
        record.face_box = best_box
        record.crop_box = make_head_crop(
            best_box,
            image.shape[1],
            image.shape[0],
        )
        record.detection_confidence = confidence
        record.direction_caption = self._estimate_direction(landmarks)

        if len(candidates) > 1:
            record.auto_reasons.append("multiple_faces_detected")
            record.apply_auto_status(DatasetStatus.REVIEW)
        else:
            record.apply_auto_status(DatasetStatus.ACCEPTED)

        pieces = [trigger_token.strip()] if trigger_token.strip() else []
        pieces.extend(["person", record.direction_caption])
        record.caption = ", ".join(part for part in pieces if part)
        return record

    def _candidate_from_row(
        self,
        row: np.ndarray,
        scale: float,
    ) -> tuple[Rect, np.ndarray, float]:
        inverse = 1.0 / scale

        x, y, w, h = [float(value) for value in row[:4]]
        box = Rect(
            int(round(x * inverse)),
            int(round(y * inverse)),
            int(round(w * inverse)),
            int(round(h * inverse)),
        )

        landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
        landmarks *= inverse
        confidence = float(row[14])
        return box, landmarks, confidence

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
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    @staticmethod
    def _estimate_direction(landmarks: np.ndarray) -> str:
        eye_x = sorted([float(landmarks[0, 0]), float(landmarks[1, 0])])
        left_eye_x, right_eye_x = eye_x
        eye_distance = max(1.0, right_eye_x - left_eye_x)
        eye_mid_x = (left_eye_x + right_eye_x) / 2.0
        nose_x = float(landmarks[2, 0])

        shift = (nose_x - eye_mid_x) / eye_distance

        if abs(shift) < 0.12:
            return "front view"
        if shift <= -0.28:
            return "left three-quarter view"
        if shift < -0.12:
            return "slightly left"
        if shift >= 0.28:
            return "right three-quarter view"
        return "slightly right"
