from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .domain import DatasetStatus, ImageRecord


class DatasetExporter:
    def export(
        self,
        output_dir: Path,
        records: list[ImageRecord],
        resolution: int = 1024,
    ) -> None:
        accepted_dir = output_dir / "accepted"
        review_dir = output_dir / "review"
        rejected_dir = output_dir / "rejected"
        logs_dir = output_dir / "logs"

        for directory in (
            accepted_dir,
            review_dir,
            rejected_dir,
            logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        accepted_records = [
            record
            for record in records
            if record.final_status is DatasetStatus.ACCEPTED
        ]
        review_records = [
            record
            for record in records
            if record.final_status is DatasetStatus.REVIEW
        ]
        rejected_records = [
            record
            for record in records
            if record.final_status is DatasetStatus.REJECTED
        ]

        self._export_accepted(
            accepted_dir,
            accepted_records,
            resolution,
        )
        self._copy_preserved(review_dir, review_records)
        self._copy_preserved(rejected_dir, rejected_records)

        decisions = [self._decision(record) for record in records]
        quality_scores = [
            {
                "source_file": str(record.source_file),
                "quality_status": (
                    record.quality_status.value
                    if record.quality_status is not None
                    else None
                ),
                "quality_score": record.quality_score,
                "blur_score": record.blur_score,
                "face_pixel_size": record.face_pixel_size,
                "face_size_ratio": record.face_size_ratio,
                "exposure_mean": record.exposure_mean,
                "crop_touches_edge": record.crop_touches_edge,
            }
            for record in records
        ]
        vision_reviews = [
            {
                "source_file": str(record.source_file),
                "vision_status": (
                    record.vision_status.value
                    if record.vision_status is not None
                    else None
                ),
                "usable_for_lora": record.vision_usable_for_lora,
                "hair_visible": record.vision_hair_visible,
                "occlusion": record.vision_occlusion,
                "blur": record.vision_blur,
                "crop_quality": record.vision_crop_quality,
                "notes": record.vision_notes,
                "reasons": record.vision_reasons,
            }
            for record in records
            if record.vision_status is not None
            or record.vision_notes
        ]

        summary = {
            "total": len(records),
            "accepted": len(accepted_records),
            "review": len(review_records),
            "rejected": len(rejected_records),
            "vision_reviewed": sum(
                record.vision_status is not None
                for record in records
            ),
            "user_overrides": sum(
                record.user_override
                for record in records
            ),
        }

        self._write_json(logs_dir / "decisions.json", decisions)
        self._write_json(
            logs_dir / "quality_scores.json",
            quality_scores,
        )
        self._write_json(
            logs_dir / "vision_reviews.json",
            vision_reviews,
        )
        self._write_json(
            logs_dir / "dataset_summary.json",
            summary,
        )

    def _export_accepted(
        self,
        accepted_dir: Path,
        records: list[ImageRecord],
        resolution: int,
    ) -> None:
        for index, record in enumerate(records, start=1):
            if record.crop_box is None:
                continue

            image = self._read_image(record.source_file)
            if image is None:
                continue

            x, y, w, h = record.crop_box.as_tuple()
            crop = image[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop)
            pil_image = pil_image.resize(
                (resolution, resolution),
                Image.Resampling.LANCZOS,
            )

            stem = f"{index:04d}"
            pil_image.save(accepted_dir / f"{stem}.png")
            (accepted_dir / f"{stem}.txt").write_text(
                (record.caption or "person").strip() + "\n",
                encoding="utf-8",
            )

    @staticmethod
    def _copy_preserved(
        output_dir: Path,
        records: list[ImageRecord],
    ) -> None:
        for index, record in enumerate(records, start=1):
            suffix = record.source_file.suffix.lower() or ".img"
            target = output_dir / (
                f"{index:04d}_{record.source_file.stem}{suffix}"
            )
            try:
                shutil.copy2(record.source_file, target)
            except OSError:
                continue

    @staticmethod
    def _decision(record: ImageRecord) -> dict[str, object]:
        return {
            "source_file": str(record.source_file),
            "auto_status": record.auto_status.value,
            "final_status": record.final_status.value,
            "auto_reasons": record.auto_reasons,
            "user_override": record.user_override,
            "detected_faces_count": record.detected_faces_count,
            "detection_confidence": record.detection_confidence,
            "direction_caption": record.direction_caption,
            "caption": record.caption,
            "identity_similarity": record.identity_similarity,
            "is_reference": record.is_reference,
            "quality_status": (
                record.quality_status.value
                if record.quality_status is not None
                else None
            ),
            "quality_score": record.quality_score,
            "blur_score": record.blur_score,
            "face_pixel_size": record.face_pixel_size,
            "face_size_ratio": record.face_size_ratio,
            "exposure_mean": record.exposure_mean,
            "crop_touches_edge": record.crop_touches_edge,
            "vision_status": (
                record.vision_status.value
                if record.vision_status is not None
                else None
            ),
            "vision_usable_for_lora": record.vision_usable_for_lora,
            "vision_hair_visible": record.vision_hair_visible,
            "vision_occlusion": record.vision_occlusion,
            "vision_blur": record.vision_blur,
            "vision_crop_quality": record.vision_crop_quality,
            "vision_notes": record.vision_notes,
            "vision_reasons": record.vision_reasons,
            "face_box": (
                record.face_box.as_tuple()
                if record.face_box
                else None
            ),
            "crop_box": (
                record.crop_box.as_tuple()
                if record.crop_box
                else None
            ),
        }

    @staticmethod
    def _read_image(path: Path) -> np.ndarray | None:
        return cv2.imdecode(
            np.fromfile(path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
