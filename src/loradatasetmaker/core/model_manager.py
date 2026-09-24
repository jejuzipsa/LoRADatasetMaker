from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Iterable


YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"
SFACE_URLS = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
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
        urls=YUNET_URLS,
        sha256=YUNET_SHA256,
        label="YuNet 얼굴 검출 모델",
    )


def ensure_sface_model() -> Path:
    return _ensure_model(
        filename=SFACE_FILENAME,
        urls=SFACE_URLS,
        sha256=SFACE_SHA256,
        label="SFace 얼굴 임베딩 모델",
    )


def _ensure_model(
    filename: str,
    urls: Iterable[str],
    sha256: str,
    label: str,
) -> Path:
    target = models_dir() / filename

    if target.exists() and _sha256(target) == sha256:
        return target

    target.unlink(missing_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)

    last_problem = "알 수 없는 다운로드 오류"

    for url in urls:
        temporary.unlink(missing_ok=True)

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "LoRADatasetMaker/0.0.9",
                "Accept": "application/octet-stream",
            },
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
            last_problem = f"{type(exc).__name__}: {exc}"
            continue

        actual_hash = _sha256(temporary)
        if actual_hash == sha256:
            os.replace(temporary, target)
            return target

        if _looks_like_git_lfs_pointer(temporary):
            last_problem = (
                "Git LFS 실제 모델 대신 포인터 파일을 받았어."
            )
        else:
            last_problem = (
                "SHA256 불일치 "
                f"(expected={sha256[:12]}..., actual={actual_hash[:12]}...)"
            )

    temporary.unlink(missing_ok=True)
    raise RuntimeError(
        f"{label} 다운로드/검증에 실패했어. "
        f"{last_problem} "
        "모델 다운로드 경로를 바꾼 최신 빌드인지 확인해줘."
    )


def _looks_like_git_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_bytes()[:200]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
