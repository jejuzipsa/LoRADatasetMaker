from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .domain import DatasetStatus, ImageRecord, Rect


IDENTITY_REASON_PREFIXES = ("not_same_person", "identity_uncertain")


@dataclass(slots=True)
class ReferenceIdentity:
    source_file: Path
    descriptor: np.ndarray
    face_box: Rect


class IdentityMatcher:
    """Temporary local identity matcher.

    This is intentionally a lightweight v0.x descriptor. A dedicated face
    embedding model will replace it in a later patch.
    """

    def build_reference(self, record: ImageRecord) -> ReferenceIdentity | None:
        if record.face_box is None:
            return None
        image = self._read_image(record.source_file)
        if image is None:
            return None
        descriptor = self._extract_descriptor(image, record.face_box)
        if descriptor is None:
            return None
        return ReferenceIdentity(
            source_file=record.source_file,
            descriptor=descriptor,
            face_box=record.face_box,
        )

    def classify_records(
        self,
        records: list[ImageRecord],
        reference: ReferenceIdentity,
        same_threshold: float = 0.78,
        review_threshold: float = 0.68,
    ) -> None:
        for record in records:
            record.is_reference = record.source_file == reference.source_file
            record.remove_reason_prefix(IDENTITY_REASON_PREFIXES)

            if record.face_box is None:
                record.identity_similarity = None
                continue

            image = self._read_image(record.source_file)
            if image is None:
                record.identity_similarity = None
                continue

            descriptor = self._extract_descriptor(image, record.face_box)
            if descriptor is None:
                record.identity_similarity = None
                continue

            similarity = self._cosine_similarity(reference.descriptor, descriptor)
            record.identity_similarity = similarity

            if record.is_reference:
                record.identity_similarity = 1.0
                record.auto_status = DatasetStatus.ACCEPTED
                record.reset_user_override()
                continue

            if similarity < review_threshold:
                record.auto_status = DatasetStatus.REJECTED
                record.auto_reasons.append(f"not_same_person:{similarity:.3f}")
                record.reset_user_override()
                continue

            if similarity < same_threshold:
                record.auto_status = DatasetStatus.REVIEW
                record.auto_reasons.append(f"identity_uncertain:{similarity:.3f}")
                record.reset_user_override()
                continue

            if "multiple_faces_detected" in record.auto_reasons:
                record.auto_status = DatasetStatus.REVIEW
            else:
                record.auto_status = DatasetStatus.ACCEPTED
            record.reset_user_override()

    @staticmethod
    def _read_image(path: Path) -> np.ndarray | None:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def _extract_descriptor(
        image: np.ndarray,
        face_box: Rect,
    ) -> np.ndarray | None:
        x, y, w, h = face_box.as_tuple()
        pad_x = int(w * 0.18)
        pad_y = int(h * 0.18)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(image.shape[1], x + w + pad_x)
        y1 = min(image.shape[0], y + h + pad_y)

        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        small = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray_vec = gray.astype(np.float32).reshape(-1)
        gray_vec /= np.linalg.norm(gray_vec) + 1e-8

        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]
        ).astype(np.float32)
        hist = hist.reshape(-1)
        hist /= hist.sum() + 1e-8

        descriptor = np.concatenate(
            [gray_vec * 0.7, hist * 0.3]
        ).astype(np.float32)
        descriptor /= np.linalg.norm(descriptor) + 1e-8
        return descriptor

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        similarity = float(
            np.dot(a, b)
            / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8)
        )
        return max(0.0, min(1.0, similarity))
