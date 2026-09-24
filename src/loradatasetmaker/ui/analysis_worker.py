from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from loradatasetmaker.core.analyzer import FaceAnalyzer
from loradatasetmaker.core.domain import ImageRecord
from loradatasetmaker.core.model_manager import ensure_sface_model


class AnalysisWorker(QObject):
    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, records: list[ImageRecord], trigger_token: str) -> None:
        super().__init__()
        self.records = records
        self.trigger_token = trigger_token

    @Slot()
    def run(self) -> None:
        try:
            self.status.emit("얼굴 AI 모델 확인 중...")
            analyzer = FaceAnalyzer()
            analyzer.prepare()
            ensure_sface_model()

            total = len(self.records)
            self.status.emit(f"자동 분석 시작 / {total}장")

            for index, record in enumerate(self.records, start=1):
                record.is_reference = False
                record.identity_similarity = None
                analyzer.analyze(record, trigger_token=self.trigger_token)
                self.progress.emit(index, total)

            self.finished.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
