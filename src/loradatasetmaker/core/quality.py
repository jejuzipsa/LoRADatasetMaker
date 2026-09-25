from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .domain import DatasetStatus, ImageRecord


QUALITY_REASON_PREFIXES = ("quality_",)

HARD_REJECT_PREFIXES = (
    "face_not_detected",
    "image_load_failed",
    "identity_face_not_detected",
    "not_same_person",
)


class QualityAnalyzer:
    """Fast deterministic quality checks before optional Vision review."""

    def analyze(self, record: ImageRecord) -> ImageRecord:
        record.remove_reason_prefix(QUALITY_REASON_PREFIXES)
        record.blur_score = None
        record.face_pixel_size = None
        record.face_size_ratio = None
        record.exposure_mean = None
        record.crop_touches_edge = False
        record.quality_score = None
        record.quality_status = None

        if record.face_box is None or record.crop_box is None:
            return record

        image = self._read_image(record.source_file)
        if image is None:
            return record

        image_h, image_w = image.shape[:2]
        x, y, w, h = record.face_box.as_tuple()
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(image_w, x + w)
        y1 = min(image_h, y + h)

        face = image[y0:y1, x0:x1]
        if face.size == 0:
            return record

        face_px = min(face.shape[:2])
        record.face_pixel_size = int(face_px)
        record.face_size_ratio = float(
            max(1, w) * max(1, h) / max(1, image_w * image_h)
        )

        normalized = cv2.resize(
            face,
            (256, 256),
            interpolation=cv2.INTER_AREA
            if max(face.shape[:2]) > 256
            else cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        record.blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        record.exposure_mean = float(gray.mean())

        cx, cy, cw, ch = record.crop_box.as_tuple()
        record.crop_touches_edge = (
            cx <= 0
            or cy <= 0
            or cx + cw >= image_w
            or cy + ch >= image_h
        )

        score = 100.0
        review = False
        reject = False

        # Keep normal usable images ACCEPTED by default.  These thresholds are
        # intentionally conservative: only clearly weak training samples should
        # be demoted automatically.
        if face_px < 48:
            score -= 55
            reject = True
            record.auto_reasons.append("quality_face_too_small")
        elif face_px < 72:
            score -= 25
            review = True
            record.auto_reasons.append("quality_face_small")
        elif face_px < 96:
            score -= 8
            record.auto_reasons.append("quality_face_marginal")

        if record.blur_score < 12:
            score -= 50
            reject = True
            record.auto_reasons.append("quality_blur_heavy")
        elif record.blur_score < 28:
            score -= 22
            review = True
            record.auto_reasons.append("quality_blur")
        elif record.blur_score < 45:
            score -= 6
            record.auto_reasons.append("quality_blur_mild")

        if record.exposure_mean < 10 or record.exposure_mean > 245:
            score -= 35
            reject = True
            record.auto_reasons.append("quality_exposure_extreme")
        elif record.exposure_mean < 25 or record.exposure_mean > 230:
            score -= 14
            review = True
            record.auto_reasons.append("quality_exposure")

        # A crop touching the source boundary is useful diagnostic information,
        # but is not a defect by itself.  Hair/head content may still be fully
        # present, so do not demote the image for this reason alone.
        if record.crop_touches_edge:
            score -= 3
            record.auto_reasons.append("quality_crop_touches_edge")

        # YuNet already has its own detection threshold.  A slightly lower
        # confidence should be visible in the log, not force a REVIEW.
        if (
            record.detection_confidence is not None
            and record.detection_confidence < 0.90
        ):
            score -= 3
            record.auto_reasons.append("quality_detection_confidence")

        record.quality_score = max(0.0, min(100.0, score))

        if reject:
            record.quality_status = DatasetStatus.REJECTED
        elif review:
            record.quality_status = DatasetStatus.REVIEW
        else:
            record.quality_status = DatasetStatus.ACCEPTED

        if self._has_hard_reject(record):
            return record

        if record.quality_status is DatasetStatus.REJECTED:
            record.apply_auto_status(DatasetStatus.REJECTED)
        elif (
            record.quality_status is DatasetStatus.REVIEW
            and record.auto_status is DatasetStatus.ACCEPTED
        ):
            record.apply_auto_status(DatasetStatus.REVIEW)

        return record

    @staticmethod
    def _has_hard_reject(record: ImageRecord) -> bool:
        return any(
            reason.startswith(HARD_REJECT_PREFIXES)
            for reason in record.auto_reasons
        )

    @staticmethod
    def _read_image(path: Path) -> np.ndarray | None:
        return cv2.imdecode(
            np.fromfile(path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
