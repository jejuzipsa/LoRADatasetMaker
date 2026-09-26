from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from PySide6.QtCore import QObject, Signal, Slot


MUSUBI_VERSION = "v0.3.5"
MUSUBI_ARCHIVE_URL = (
    "https://github.com/kohya-ss/musubi-tuner/"
    f"archive/refs/tags/{MUSUBI_VERSION}.zip"
)
UV_ARCHIVE_URL = (
    "https://github.com/astral-sh/uv/releases/latest/download/"
    "uv-x86_64-pc-windows-msvc.zip"
)


def default_tools_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "tools"
    return Path.cwd() / "tools"


def validate_python_executable(python_exe: Path) -> tuple[bool, str]:
    """Check that a venv python.exe is more than an existing uv trampoline."""
    if not python_exe.is_file():
        return False, "python.exe 파일 없음"

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW

    try:
        result = subprocess.run(
            [str(python_exe), "-V"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            creationflags=creationflags,
        )
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)

    output = (result.stdout or "").strip()
    if result.returncode != 0:
        return False, output or f"exit code {result.returncode}"
    return True, output or "Python OK"


class MusubiInstallWorker(QObject):
    status = Signal(str)
    finished = Signal(str, str, str)
    failed = Signal(str)

    def __init__(self, tools_root: Path | None = None) -> None:
        super().__init__()
        self.tools_root = (tools_root or default_tools_root()).resolve()

    @Slot()
    def run(self) -> None:
        try:
            self.tools_root.mkdir(parents=True, exist_ok=True)

            uv_dir = self.tools_root / "uv"
            uv_exe = uv_dir / "uv.exe"
            if not uv_exe.is_file():
                self.status.emit("uv 설치 도구 다운로드 중...")
                self._install_uv(uv_dir)

            musubi_dir = self.tools_root / "musubi-tuner"
            version_marker = musubi_dir / ".loradatasetmaker_version"
            current_version = (
                version_marker.read_text(encoding="utf-8").strip()
                if version_marker.is_file()
                else ""
            )
            if not musubi_dir.is_dir() or current_version != MUSUBI_VERSION:
                self.status.emit(
                    f"Musubi Tuner {MUSUBI_VERSION} 다운로드 중..."
                )
                self._install_musubi_source(musubi_dir)
                version_marker.write_text(
                    MUSUBI_VERSION + "\n",
                    encoding="utf-8",
                )

            env = os.environ.copy()
            env["UV_PYTHON_INSTALL_DIR"] = str(
                self.tools_root / "python"
            )
            env["UV_CACHE_DIR"] = str(self.tools_root / "uv-cache")
            env["PYTHONUTF8"] = "1"

            self.status.emit("Musubi용 Python 3.11 준비 중...")
            self._run_command(
                [str(uv_exe), "python", "install", "3.11"],
                cwd=musubi_dir,
                env=env,
            )

            python_exe = musubi_dir / ".venv" / "Scripts" / "python.exe"
            if python_exe.exists():
                valid, detail = validate_python_executable(python_exe)
                if not valid:
                    self.status.emit(
                        "기존 Musubi Python 환경 손상 감지 - .venv 자동 복구 중..."
                    )
                    try:
                        shutil.rmtree(musubi_dir / ".venv")
                    except OSError as exc:
                        raise RuntimeError(
                            "손상된 Musubi .venv를 삭제하지 못했어. "
                            "실행 중인 Python/Training 프로세스를 종료한 뒤 다시 시도해줘. "
                            f"상세: {exc}"
                        ) from exc

            self.status.emit(
                "Musubi 의존성 설치/복구 중... (PyTorch CUDA 12.8 포함, 시간이 걸릴 수 있어)"
            )
            self._run_command(
                [
                    str(uv_exe),
                    "sync",
                    "--extra",
                    "cu128",
                    "--python",
                    "3.11",
                ],
                cwd=musubi_dir,
                env=env,
            )

            python_exe = musubi_dir / ".venv" / "Scripts" / "python.exe"
            valid, detail = validate_python_executable(python_exe)
            if not valid:
                raise RuntimeError(
                    "Musubi .venv Python이 생성됐지만 실행할 수 없어. "
                    f"상세: {detail}"
                )

            self.status.emit("설치 검증 중...")
            self._run_command(
                [
                    str(python_exe),
                    "-c",
                    (
                        "import torch, accelerate, safetensors; "
                        "print('torch', torch.__version__); "
                        "print('cuda', torch.cuda.is_available())"
                    ),
                ],
                cwd=musubi_dir,
                env=env,
            )

            self.finished.emit(
                str(musubi_dir),
                str(python_exe),
                MUSUBI_VERSION,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _install_uv(self, uv_dir: Path) -> None:
        tmp_zip = self.tools_root / "uv-download.zip"
        tmp_extract = self.tools_root / "uv-extract"
        try:
            self._download(UV_ARCHIVE_URL, tmp_zip)
            if tmp_extract.exists():
                shutil.rmtree(tmp_extract)
            tmp_extract.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp_zip, "r") as archive:
                archive.extractall(tmp_extract)

            uv_candidates = list(tmp_extract.rglob("uv.exe"))
            if not uv_candidates:
                raise RuntimeError("uv.exe를 다운로드 파일에서 찾지 못했어.")

            uv_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(uv_candidates[0], uv_dir / "uv.exe")
        finally:
            if tmp_zip.exists():
                tmp_zip.unlink()
            if tmp_extract.exists():
                shutil.rmtree(tmp_extract, ignore_errors=True)

    def _install_musubi_source(self, target: Path) -> None:
        tmp_zip = self.tools_root / "musubi-download.zip"
        tmp_extract = self.tools_root / "musubi-extract"
        try:
            self._download(MUSUBI_ARCHIVE_URL, tmp_zip)
            if tmp_extract.exists():
                shutil.rmtree(tmp_extract)
            tmp_extract.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp_zip, "r") as archive:
                archive.extractall(tmp_extract)

            candidates = [
                path
                for path in tmp_extract.iterdir()
                if path.is_dir() and path.name.lower().startswith("musubi-tuner")
            ]
            if not candidates:
                raise RuntimeError(
                    "Musubi Tuner 압축을 풀었지만 소스 폴더를 찾지 못했어."
                )

            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(candidates[0]), str(target))
        finally:
            if tmp_zip.exists():
                tmp_zip.unlink()
            if tmp_extract.exists():
                shutil.rmtree(tmp_extract, ignore_errors=True)

    def _download(self, url: str, target: Path) -> None:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "LoRADatasetMaker/0.0.19"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("Content-Length") or 0)
            read = 0
            with target.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    read += len(chunk)
                    if total > 0:
                        percent = int(read * 100 / total)
                        self.status.emit(
                            f"다운로드 중... {percent}%"
                        )

    def _run_command(
        self,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
    ) -> None:
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        assert process.stdout is not None

        tail: list[str] = []
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            tail.append(line)
            tail = tail[-12:]
            self.status.emit(line[:240])

        code = process.wait()
        if code != 0:
            detail = "\n".join(tail[-8:])
            raise RuntimeError(
                "설치 명령이 실패했어.\n"
                + ("\n" + detail if detail else "")
            )
