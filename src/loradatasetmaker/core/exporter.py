from __future__ import annotations

import json
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
        logs_dir = output_dir / "logs"
        accepted_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        accepted_records = [
            record
            for record in records
            if record.final_status is DatasetStatus.ACCEPTED
        ]

        for index, record in enumerate(accepted_records, start=1):
            if record.crop_box is None:
                continue

            image = cv2.imdecode(
                np.fromfile(record.source_file, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
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

        decisions: list[dict[str, object]] = []
        for record in records:
            decisions.append(
                {
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
                    "face_box": (
                        record.face_box.as_tuple() if record.face_box else None
                    ),
                    "crop_box": (
                        record.crop_box.as_tuple() if record.crop_box else None
                    ),
                }
            )

        (logs_dir / "decisions.json").write_text(
            json.dumps(decisions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
