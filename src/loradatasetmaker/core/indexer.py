from __future__ import annotations

from pathlib import Path

from .domain import ImageRecord


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def index_image_folder(folder: Path, recursive: bool = True) -> list[ImageRecord]:
    """Index supported image files without modifying the source folder."""
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    files = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return [ImageRecord(source_file=path) for path in files]
