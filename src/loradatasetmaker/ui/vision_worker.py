from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from loradatasetmaker.core.domain import ImageRecord
from loradatasetmaker.core.vision import OllamaVisionReviewer


class VisionWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(int, int)
    failed = Signal(str)

    def __init__(
        self,
        records: list[ImageRecord],
        endpoint: str,
        model: str,
    ) -> None:
        super().__init__()
        self.records = records
        self.endpoint = endpoint
        self.model = model

    @Slot()
    def run(self) -> None:
        try:
            reviewer = OllamaVisionReviewer(
                endpoint=self.endpoint,
                model=self.model,
            )

            total = len(self.records)
            reviewed = 0
            errors = 0

            for index, record in enumerate(self.records, start=1):
                try:
                    reviewer.review(record)
                    reviewed += 1
                except Exception as exc:  # noqa: BLE001
                    record.vision_notes = (
                        f"ERROR: {type(exc).__name__}: {exc}"
                    )[:400]
                    errors += 1
                self.progress.emit(index, total)

            self.finished.emit(reviewed, errors)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
