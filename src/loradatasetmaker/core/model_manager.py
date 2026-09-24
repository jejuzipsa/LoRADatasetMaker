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

SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"
SFACE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def models_dir() -> Path:
    path = app_base_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_yunet_model() -> Path:
    return _ensure_model(
        filename=YUNET_FILENAME,
        url=YUNET_URL,
        sha256=YUNET_SHA256,
        label="YuNet 얼굴 검출 모델",
    )


def ensure_sface_model() -> Path:
    return _ensure_model(
        filename=SFACE_FILENAME,
        url=SFACE_URL,
        sha256=SFACE_SHA256,
        label="SFace 얼굴 임베딩 모델",
    )


def _ensure_model(
    filename: str,
    url: str,
    sha256: str,
    label: str,
) -> Path:
    target = models_dir() / filename

    if target.exists() and _sha256(target) == sha256:
        return target

    if target.exists():
        target.unlink(missing_ok=True)

    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "LoRADatasetMaker/0.0.8"},
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
    except Exception as exc:  # noqa: BLE001
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{label}을 다운로드하지 못했어. "
            "인터넷 연결을 확인한 뒤 다시 시도해줘."
        ) from exc

    actual_hash = _sha256(temporary)
    if actual_hash != sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"다운로드한 {label}의 SHA256 검증에 실패했어."
        )

    os.replace(temporary, target)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
