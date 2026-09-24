from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from loradatasetmaker.core.domain import ImageRecord
from loradatasetmaker.core.identity import IdentityMatcher, ReferenceIdentity
from loradatasetmaker.core.quality import QualityAnalyzer


class IdentityWorker(QObject):
    progress = Signal(int, int)
    finished = Signal()
    failed = Signal(str)

    def __init__(
        self,
        records: list[ImageRecord],
        reference: ReferenceIdentity,
    ) -> None:
        super().__init__()
        self.records = records
        self.reference = reference

    @Slot()
    def run(self) -> None:
        try:
            matcher = IdentityMatcher()
            matcher.prepare()
            quality = QualityAnalyzer()

            total = len(self.records)
            for index, record in enumerate(self.records, start=1):
                record.clear_vision_review()
                matcher.classify_record(record, self.reference)
                quality.analyze(record)
                self.progress.emit(index, total)

            self.finished.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
