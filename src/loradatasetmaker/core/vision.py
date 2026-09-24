from __future__ import annotations

import base64
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .domain import DatasetStatus, ImageRecord


VISION_REASON_PREFIXES = ("vision_",)


@dataclass(slots=True)
class VisionReviewResult:
    usable_for_lora: str
    hair_visible: str
    occlusion: str
    blur: str
    crop_quality: str
    notes: str


class OllamaVisionReviewer:
    """Optional semantic review through a local Ollama vision model."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: int = 120,
    ) -> None:
        self.endpoint = endpoint.strip() or "http://127.0.0.1:11434/api/chat"
        self.model = model.strip()
        self.timeout = timeout

        if not self.model:
            raise ValueError("Vision 모델 이름이 비어 있어.")

    def review(self, record: ImageRecord) -> VisionReviewResult:
        if record.crop_box is None:
            raise ValueError("Head Crop이 없는 항목이야.")

        image = self._read_image(record.source_file)
        if image is None:
            raise ValueError("이미지를 읽지 못했어.")

        original = self._annotated_original(image, record)
        crop = self._crop(image, record)

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "user",
                    "content": self._prompt(record),
                    "images": [
                        self._encode_jpeg(original),
                        self._encode_jpeg(crop),
                    ],
                }
            ],
        }

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "LoRADatasetMaker/0.0.10",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        content = body.get("message", {}).get("content", "")
        parsed = self._parse_json_content(content)

        result = VisionReviewResult(
            usable_for_lora=self._choice(
                parsed.get("usable_for_lora"),
                {"yes", "no", "maybe"},
                "maybe",
            ),
            hair_visible=self._choice(
                parsed.get("hair_fully_visible"),
                {"yes", "partial", "no"},
                "partial",
            ),
            occlusion=self._choice(
                parsed.get("face_occluded"),
                {"none", "mild", "heavy"},
                "mild",
            ),
            blur=self._choice(
                parsed.get("blur_level"),
                {"low", "medium", "high"},
                "medium",
            ),
            crop_quality=self._choice(
                parsed.get("crop_quality"),
                {"good", "acceptable", "bad"},
                "acceptable",
            ),
            notes=str(parsed.get("notes", "")).strip()[:400],
        )
        self.apply_result(record, result)
        return result

    def apply_result(
        self,
        record: ImageRecord,
        result: VisionReviewResult,
    ) -> None:
        record.remove_reason_prefix(VISION_REASON_PREFIXES)
        record.vision_reasons = []
        record.vision_usable_for_lora = result.usable_for_lora
        record.vision_hair_visible = result.hair_visible
        record.vision_occlusion = result.occlusion
        record.vision_blur = result.blur
        record.vision_crop_quality = result.crop_quality
        record.vision_notes = result.notes

        reject = (
            result.usable_for_lora == "no"
            or result.occlusion == "heavy"
            or result.blur == "high"
            or result.crop_quality == "bad"
        )
        review = (
            result.usable_for_lora == "maybe"
            or result.hair_visible != "yes"
            or result.occlusion == "mild"
            or result.blur == "medium"
            or result.crop_quality == "acceptable"
        )

        if result.usable_for_lora == "no":
            record.vision_reasons.append("vision_not_usable")
        if result.hair_visible == "partial":
            record.vision_reasons.append("vision_hair_partial")
        elif result.hair_visible == "no":
            record.vision_reasons.append("vision_hair_cut")
        if result.occlusion != "none":
            record.vision_reasons.append(
                f"vision_occlusion_{result.occlusion}"
            )
        if result.blur != "low":
            record.vision_reasons.append(f"vision_blur_{result.blur}")
        if result.crop_quality != "good":
            record.vision_reasons.append(
                f"vision_crop_{result.crop_quality}"
            )

        record.auto_reasons.extend(record.vision_reasons)

        if reject:
            record.vision_status = DatasetStatus.REJECTED
            record.apply_auto_status(DatasetStatus.REJECTED)
        elif review:
            record.vision_status = DatasetStatus.REVIEW
            if record.auto_status is DatasetStatus.ACCEPTED:
                record.apply_auto_status(DatasetStatus.REVIEW)
        else:
            record.vision_status = DatasetStatus.ACCEPTED

    @staticmethod
    def _prompt(record: ImageRecord) -> str:
        quality = (
            f"{record.quality_score:.1f}"
            if record.quality_score is not None
            else "unknown"
        )
        similarity = (
            f"{record.identity_similarity:.3f}"
            if record.identity_similarity is not None
            else "unknown"
        )
        return f"""You are reviewing a candidate image for a head/identity LoRA training dataset.

Image 1 is the original image with the detected target face marked.
Image 2 is the proposed head crop.

Judge training suitability, not attractiveness or demographics.
Focus on whether the target head/face is clear, sufficiently visible, not badly occluded, not badly blurred, and whether the proposed crop preserves the full head/hair reasonably well.

Existing machine signals:
- quality_score: {quality}/100
- identity_similarity: {similarity}
- direction: {record.direction_caption or "unknown"}

Return strict JSON only:
{{
  "usable_for_lora": "yes|no|maybe",
  "hair_fully_visible": "yes|partial|no",
  "face_occluded": "none|mild|heavy",
  "blur_level": "low|medium|high",
  "crop_quality": "good|acceptable|bad",
  "notes": "one short Korean sentence"
}}
"""

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, object]:
        text = content.strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith(fence):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Vision 응답 JSON이 객체가 아니야.")
        return parsed

    @staticmethod
    def _choice(
        value: object,
        allowed: set[str],
        fallback: str,
    ) -> str:
        normalized = str(value or "").strip().lower()
        return normalized if normalized in allowed else fallback

    @staticmethod
    def _read_image(path: Path) -> np.ndarray | None:
        return cv2.imdecode(
            np.fromfile(path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

    @staticmethod
    def _annotated_original(
        image: np.ndarray,
        record: ImageRecord,
    ) -> np.ndarray:
        out = image.copy()
        if record.face_box is not None:
            x, y, w, h = record.face_box.as_tuple()
            cv2.rectangle(
                out,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                max(2, int(round(max(out.shape[:2]) / 700))),
            )
        return OllamaVisionReviewer._resize_for_vision(out)

    @staticmethod
    def _crop(image: np.ndarray, record: ImageRecord) -> np.ndarray:
        assert record.crop_box is not None
        x, y, w, h = record.crop_box.as_tuple()
        crop = image[y : y + h, x : x + w]
        if crop.size == 0:
            raise ValueError("Head Crop 영역이 비어 있어.")
        return OllamaVisionReviewer._resize_for_vision(crop)

    @staticmethod
    def _resize_for_vision(image: np.ndarray, limit: int = 768) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= limit:
            return image

        scale = limit / float(longest)
        return cv2.resize(
            image,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _encode_jpeg(image: np.ndarray) -> str:
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
        if not ok:
            raise ValueError("Vision용 JPEG 인코딩에 실패했어.")
        return base64.b64encode(encoded.tobytes()).decode("ascii")
