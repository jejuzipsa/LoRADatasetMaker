from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cropper import make_head_crop
from .domain import DatasetStatus, ImageRecord, Rect
from .model_manager import ensure_sface_model, ensure_yunet_model


IDENTITY_REASON_PREFIXES = (
    "not_same_person",
    "identity_uncertain",
    "identity_face_not_detected",
)


@dataclass(slots=True)
class ReferenceIdentity:
    source_file: Path
    feature: np.ndarray
    face_box: Rect


class IdentityMatcher:
    """YuNet + SFace based same-person matcher."""

    def __init__(
        self,
        max_detection_size: int = 1280,
        face_score_threshold: float = 0.90,
        same_threshold: float = 0.45,
        review_threshold: float = 0.363,
    ) -> None:
        self.max_detection_size = max_detection_size
        self.face_score_threshold = face_score_threshold
        self.same_threshold = same_threshold
        self.review_threshold = review_threshold
        self.detector: cv2.FaceDetectorYN | None = None
        self.recognizer: cv2.FaceRecognizerSF | None = None

    def prepare(self) -> None:
        if self.detector is not None and self.recognizer is not None:
            return

        yunet_path = ensure_yunet_model()
        sface_path = ensure_sface_model()

        self.detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (320, 320),
            self.face_score_threshold,
            0.30,
            5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(sface_path),
            "",
        )

    def build_reference(self, record: ImageRecord) -> ReferenceIdentity | None:
        self.prepare()
        image = self._read_image(record.source_file)
        if image is None:
            return None

        faces = self._detect_faces(image)
        if not faces:
            return None

        if record.face_box is not None:
            face = max(
                faces,
                key=lambda candidate: self._iou(
                    record.face_box,
                    self._rect_from_face(candidate),
                ),
            )
        else:
            face = max(faces, key=lambda candidate: self._rect_from_face(candidate).area)

        feature = self._extract_feature(image, face)
        if feature is None:
            return None

        box = self._rect_from_face(face)
        record.face_box = box
        record.crop_box = make_head_crop(box, image.shape[1], image.shape[0])
        record.detection_confidence = float(face[14])

        return ReferenceIdentity(
            source_file=record.source_file,
            feature=feature,
            face_box=box,
        )

    def classify_records(
        self,
        records: list[ImageRecord],
        reference: ReferenceIdentity,
    ) -> None:
        self.prepare()
        for record in records:
            self.classify_record(record, reference)

    def classify_record(
        self,
        record: ImageRecord,
        reference: ReferenceIdentity,
    ) -> None:
        self.prepare()
        assert self.recognizer is not None

        record.is_reference = record.source_file == reference.source_file
        record.remove_reason_prefix(IDENTITY_REASON_PREFIXES)

        image = self._read_image(record.source_file)
        if image is None:
            record.identity_similarity = None
            return

        faces = self._detect_faces(image)
        record.detected_faces_count = len(faces)

        if not faces:
            record.identity_similarity = None
            record.apply_auto_status(DatasetStatus.REJECTED)
            record.auto_reasons.append("identity_face_not_detected")
            return

        best_face: np.ndarray | None = None
        best_similarity = -1.0

        for face in faces:
            feature = self._extract_feature(image, face)
            if feature is None:
                continue

            similarity = float(
                self.recognizer.match(
                    reference.feature,
                    feature,
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_face = face

        if best_face is None:
            record.identity_similarity = None
            record.apply_auto_status(DatasetStatus.REVIEW)
            record.auto_reasons.append("identity_uncertain:no_feature")
            return

        record.identity_similarity = best_similarity
        box = self._rect_from_face(best_face)
        record.face_box = box
        record.crop_box = make_head_crop(
            box,
            image.shape[1],
            image.shape[0],
        )
        record.detection_confidence = float(best_face[14])
        record.direction_caption = self._estimate_direction(
            np.asarray(best_face[4:14], dtype=np.float32).reshape(5, 2)
        )

        if record.is_reference:
            record.identity_similarity = 1.0
            record.auto_status = DatasetStatus.ACCEPTED
            record.reset_user_override()
            return

        if best_similarity < self.review_threshold:
            record.apply_auto_status(DatasetStatus.REJECTED)
            record.auto_reasons.append(
                f"not_same_person:{best_similarity:.3f}"
            )
            return

        if best_similarity < self.same_threshold:
            record.apply_auto_status(DatasetStatus.REVIEW)
            record.auto_reasons.append(
                f"identity_uncertain:{best_similarity:.3f}"
            )
            return

        if len(faces) > 1:
            if "multiple_faces_detected" not in record.auto_reasons:
                record.auto_reasons.append("multiple_faces_detected")
            record.apply_auto_status(DatasetStatus.REVIEW)
        else:
            record.apply_auto_status(DatasetStatus.ACCEPTED)

    def _detect_faces(self, image: np.ndarray) -> list[np.ndarray]:
        assert self.detector is not None

        detect_image, scale = self._prepare_detection_image(image)
        height, width = detect_image.shape[:2]
        self.detector.setInputSize((width, height))
        _retval, faces = self.detector.detect(detect_image)

        if faces is None:
            return []

        restored: list[np.ndarray] = []
        inverse = 1.0 / scale

        for row in faces:
            if float(row[14]) < self.face_score_threshold:
                continue

            face = np.asarray(row, dtype=np.float32).copy()
            face[:14] *= inverse
            restored.append(face)

        return restored

    def _extract_feature(
        self,
        image: np.ndarray,
        face: np.ndarray,
    ) -> np.ndarray | None:
        assert self.recognizer is not None
        try:
            aligned = self.recognizer.alignCrop(image, face[:14])
            return self.recognizer.feature(aligned)
        except cv2.error:
            return None

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
    def _read_image(path: Path) -> np.ndarray | None:
        return cv2.imdecode(
            np.fromfile(path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

    @staticmethod
    def _rect_from_face(face: np.ndarray) -> Rect:
        x, y, w, h = [int(round(float(value))) for value in face[:4]]
        return Rect(x, y, w, h)

    @staticmethod
    def _iou(a: Rect, b: Rect) -> float:
        ax2 = a.x + a.w
        ay2 = a.y + a.h
        bx2 = b.x + b.w
        by2 = b.y + b.h

        ix1 = max(a.x, b.x)
        iy1 = max(a.y, b.y)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = iw * ih
        union = a.area + b.area - intersection

        return intersection / float(union) if union > 0 else 0.0

    @staticmethod
    def _estimate_direction(landmarks: np.ndarray) -> str:
        eye_x = sorted(
            [float(landmarks[0, 0]), float(landmarks[1, 0])]
        )
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
