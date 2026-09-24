from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path


YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def models_dir() -> Path:
    path = app_base_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_yunet_model() -> Path:
    target = models_dir() / YUNET_FILENAME

    if target.exists() and _sha256(target) == YUNET_SHA256:
        return target

    if target.exists():
        target.unlink(missing_ok=True)

    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)

    request = urllib.request.Request(
        YUNET_URL,
        headers={"User-Agent": "LoRADatasetMaker/0.0.7"},
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
    except Exception as exc:  # noqa: BLE001
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "YuNet 얼굴 검출 모델을 다운로드하지 못했어. "
            "인터넷 연결을 확인한 뒤 다시 자동 분석을 눌러줘."
        ) from exc

    actual_hash = _sha256(temporary)
    if actual_hash != YUNET_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "다운로드한 YuNet 모델의 SHA256 검증에 실패했어."
        )

    os.replace(temporary, target)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
