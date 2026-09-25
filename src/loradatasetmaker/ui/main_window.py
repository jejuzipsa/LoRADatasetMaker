from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize, QThread, Qt
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from loradatasetmaker.core.domain import DatasetStatus, ImageRecord
from loradatasetmaker.core.exporter import DatasetExporter
from loradatasetmaker.core.identity import IdentityMatcher, ReferenceIdentity
from loradatasetmaker.core.indexer import (
    SUPPORTED_EXTENSIONS,
    index_image_folder,
)
from loradatasetmaker.ui.analysis_worker import AnalysisWorker
from loradatasetmaker.ui.identity_worker import IdentityWorker
from loradatasetmaker.ui.vision_worker import VisionWorker


ROLE_RECORD_INDEX = Qt.ItemDataRole.UserRole


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[ImageRecord] = []
        self.reference_identity: ReferenceIdentity | None = None
        self.identity_matcher = IdentityMatcher()
        self.exporter = DatasetExporter()

        self.analysis_thread: QThread | None = None
        self.analysis_worker: AnalysisWorker | None = None
        self.identity_thread: QThread | None = None
        self.identity_worker: IdentityWorker | None = None
        self.vision_thread: QThread | None = None
        self.vision_worker: VisionWorker | None = None

        self.setWindowTitle("LoRA Dataset Maker")
        self.resize(1500, 920)
        self.setMinimumSize(1180, 760)
        self.setAcceptDrops(True)

        self._build_ui()
        self._apply_dark_theme()
        self._refresh_counts()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        controls = QHBoxLayout()

        self.folder_button = QPushButton("사진 폴더 선택")
        self.folder_button.clicked.connect(self._choose_folder)
        controls.addWidget(self.folder_button)

        self.folder_label = QLabel("선택된 폴더 없음 / 폴더나 이미지를 드래그앤드롭해도 돼")
        self.folder_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        controls.addWidget(self.folder_label, 1)

        controls.addWidget(QLabel("Trigger"))
        self.trigger_edit = QLineEdit()
        self.trigger_edit.setPlaceholderText("예: personA")
        self.trigger_edit.setMaximumWidth(180)
        controls.addWidget(self.trigger_edit)

        self.analyze_button = QPushButton("자동 분석")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self._run_analysis)
        controls.addWidget(self.analyze_button)

        self.reference_button = QPushButton("현재 항목을 기준 인물로 지정")
        self.reference_button.setEnabled(False)
        self.reference_button.clicked.connect(self._set_reference)
        controls.addWidget(self.reference_button)

        self.classify_button = QPushButton("기준 인물로 재분류")
        self.classify_button.setEnabled(False)
        self.classify_button.clicked.connect(self._classify_identity)
        controls.addWidget(self.classify_button)

        self.export_button = QPushButton("Export")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_dataset)
        controls.addWidget(self.export_button)

        layout.addLayout(controls)

        self.reference_label = QLabel("기준 인물: 미지정")
        layout.addWidget(self.reference_label)

        self.count_label = QLabel()
        layout.addWidget(self.count_label)

        analysis_row = QHBoxLayout()
        self.analysis_status_label = QLabel("대기")
        analysis_row.addWidget(self.analysis_status_label)

        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        self.analysis_progress.setTextVisible(True)
        analysis_row.addWidget(self.analysis_progress, 1)
        layout.addLayout(analysis_row)

        vision_row = QHBoxLayout()
        vision_row.addWidget(QLabel("Vision URL"))

        self.vision_url_edit = QLineEdit(
            "http://127.0.0.1:11434/api/chat"
        )
        self.vision_url_edit.setMaximumWidth(360)
        vision_row.addWidget(self.vision_url_edit)

        vision_row.addWidget(QLabel("Model"))
        self.vision_model_edit = QLineEdit()
        self.vision_model_edit.setPlaceholderText(
            "Ollama vision model name"
        )
        self.vision_model_edit.setMaximumWidth(220)
        self.vision_model_edit.textChanged.connect(
            self._sync_vision_button
        )
        vision_row.addWidget(self.vision_model_edit)

        self.vision_button = QPushButton("Vision 검수")
        self.vision_button.setEnabled(False)
        self.vision_button.clicked.connect(self._run_vision_review)
        vision_row.addWidget(self.vision_button)

        self.vision_target_label = QLabel("대상 0")
        vision_row.addWidget(self.vision_target_label)
        vision_row.addStretch(1)
        layout.addLayout(vision_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setMovement(QListWidget.Movement.Static)
        self.list_widget.setIconSize(QSize(170, 170))
        self.list_widget.setGridSize(QSize(215, 250))
        self.list_widget.currentItemChanged.connect(self._show_current_item)
        splitter.addWidget(self.list_widget)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)

        previews = QHBoxLayout()
        self.original_label = self._make_preview_label("원본")
        self.crop_label = self._make_preview_label("크롭")
        previews.addWidget(self.original_label, 1)
        previews.addWidget(self.crop_label, 1)
        detail_layout.addLayout(previews, 1)

        self.file_label = QLabel("-")
        self.file_label.setWordWrap(True)
        self.file_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        detail_layout.addWidget(self.file_label)

        self.meta_label = QLabel("-")
        self.meta_label.setWordWrap(True)
        detail_layout.addWidget(self.meta_label)

        detail_layout.addWidget(QLabel("Caption"))
        self.caption_edit = QLineEdit()
        self.caption_edit.editingFinished.connect(self._save_caption)
        detail_layout.addWidget(self.caption_edit)

        state_row = QHBoxLayout()
        for title, status in (
            ("채택", DatasetStatus.ACCEPTED),
            ("보류", DatasetStatus.REVIEW),
            ("제외", DatasetStatus.REJECTED),
        ):
            button = QPushButton(title)
            button.clicked.connect(
                lambda _checked=False, value=status: self._set_current_status(value)
            )
            state_row.addWidget(button)
        detail_layout.addLayout(state_row)

        splitter.addWidget(detail)
        splitter.setSizes([900, 600])
        self.setCentralWidget(root)

    def _make_preview_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(300, 300)
        label.setStyleSheet("border: 1px solid #3b3f46;")
        return label

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #17191d;
                color: #e7e9ec;
                font-size: 13px;
            }
            QPushButton {
                background: #2a2e34;
                border: 1px solid #444a52;
                border-radius: 5px;
                padding: 7px 12px;
            }
            QPushButton:hover { background: #343942; }
            QPushButton:disabled { color: #777b82; background: #22252a; }
            QLineEdit {
                background: #202329;
                border: 1px solid #444a52;
                border-radius: 4px;
                padding: 6px;
            }
            QProgressBar {
                background: #202329;
                border: 1px solid #444a52;
                border-radius: 4px;
                text-align: center;
                min-height: 18px;
            }
            QProgressBar::chunk {
                background: #48515d;
                border-radius: 3px;
            }
            QListWidget {
                background: #111317;
                border: 1px solid #30343a;
                border-radius: 5px;
            }
            QListWidget::item {
                border: 2px solid transparent;
                padding: 3px;
            }
            QListWidget::item:selected {
                background: #30353d;
                border: 2px solid #e7e9ec;
                border-radius: 4px;
            }
            """
        )

    def _choose_folder(self) -> None:
        if self._is_busy():
            return

        selected = QFileDialog.getExistingDirectory(self, "사진 폴더 선택")
        if not selected:
            return

        folder = Path(selected)
        try:
            records = index_image_folder(folder)
        except OSError as exc:
            QMessageBox.critical(self, "폴더 읽기 실패", str(exc))
            return

        self.records = records
        self.folder_label.setText(str(folder))
        self._reset_after_source_change()
        self._populate_list()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._is_busy():
            event.ignore()
            return
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        if self._is_busy():
            event.ignore()
            return

        dropped_paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        added = self._add_dropped_paths(dropped_paths)
        if added:
            event.acceptProposedAction()
        else:
            event.ignore()

    def _add_dropped_paths(self, paths: list[Path]) -> int:
        candidates: list[ImageRecord] = []

        for path in paths:
            try:
                if path.is_dir():
                    candidates.extend(index_image_folder(path))
                elif (
                    path.is_file()
                    and path.suffix.lower() in SUPPORTED_EXTENSIONS
                ):
                    candidates.append(ImageRecord(source_file=path))
            except OSError:
                continue

        existing = {
            str(record.source_file.resolve()).lower()
            for record in self.records
        }
        unique: list[ImageRecord] = []
        for record in candidates:
            key = str(record.source_file.resolve()).lower()
            if key in existing:
                continue
            existing.add(key)
            unique.append(record)

        if not unique:
            return 0

        self.records.extend(unique)
        self.records.sort(key=lambda record: str(record.source_file).lower())
        self.folder_label.setText(
            f"드래그앤드롭으로 {len(unique)}개 추가 / 전체 {len(self.records)}개"
        )
        self._reset_after_source_change()
        self._populate_list()
        return len(unique)

    def _reset_after_source_change(self) -> None:
        self.reference_identity = None
        self.reference_label.setText("기준 인물: 미지정")
        self.analyze_button.setEnabled(bool(self.records))
        self.reference_button.setEnabled(False)
        self.classify_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.analysis_status_label.setText("미분석")
        self.analysis_progress.setRange(0, max(1, len(self.records)))
        self.analysis_progress.setValue(0)
        self._refresh_counts()
        self._sync_vision_button()

    def _run_analysis(self) -> None:
        if not self.records or self._is_busy():
            return

        self.reference_identity = None
        self.reference_label.setText("기준 인물: 미지정")
        self.analysis_progress.setRange(0, len(self.records))
        self.analysis_progress.setValue(0)
        self.analysis_status_label.setText(
            f"자동 분석 준비 중... 0 / {len(self.records)}"
        )
        self._set_analysis_busy(True)

        self.analysis_thread = QThread(self)
        self.analysis_worker = AnalysisWorker(
            self.records,
            self.trigger_edit.text().strip(),
        )
        self.analysis_worker.moveToThread(self.analysis_thread)

        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.progress.connect(self._on_analysis_progress)
        self.analysis_worker.status.connect(self.analysis_status_label.setText)
        self.analysis_worker.finished.connect(self._on_analysis_finished)
        self.analysis_worker.failed.connect(self._on_analysis_failed)

        self.analysis_worker.finished.connect(self.analysis_thread.quit)
        self.analysis_worker.failed.connect(self.analysis_thread.quit)
        self.analysis_worker.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_worker.failed.connect(self.analysis_worker.deleteLater)

        self.analysis_thread.finished.connect(self._on_analysis_thread_finished)
        self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)
        self.analysis_thread.start()

    def _on_analysis_progress(self, current: int, total: int) -> None:
        self.analysis_progress.setRange(0, max(1, total))
        self.analysis_progress.setValue(current)
        self.analysis_status_label.setText(
            f"자동 분석 중... {current} / {total}"
        )

    def _on_analysis_finished(self) -> None:
        self._populate_list()
        self._refresh_counts()
        self.reference_button.setEnabled(
            any(record.face_box is not None for record in self.records)
        )
        self.classify_button.setEnabled(False)
        self._sync_export_button()
        self.analysis_progress.setValue(len(self.records))
        self.analysis_status_label.setText(
            f"분석 완료 / {len(self.records)}장"
        )
        self._set_analysis_busy(False)
        self._sync_vision_button()

    def _on_analysis_failed(self, message: str) -> None:
        self.analysis_status_label.setText("분석 실패")
        self._set_analysis_busy(False)
        QMessageBox.critical(
            self,
            "자동 분석 실패",
            "자동 분석 중 오류가 발생했어.\n\n" + message,
        )

    def _on_analysis_thread_finished(self) -> None:
        self.analysis_thread = None
        self.analysis_worker = None
        self._sync_vision_button()

    def _is_busy(self) -> bool:
        return (
            self.analysis_thread is not None
            or self.identity_thread is not None
            or self.vision_thread is not None
        )

    def _set_analysis_busy(self, busy: bool) -> None:
        self.folder_button.setEnabled(not busy)
        self.vision_url_edit.setEnabled(not busy)
        self.vision_model_edit.setEnabled(not busy)
        self.analyze_button.setEnabled(not busy and bool(self.records))
        self.reference_button.setEnabled(
            not busy and any(record.face_box is not None for record in self.records)
        )
        self.classify_button.setEnabled(
            not busy and self.reference_identity is not None
        )
        self.export_button.setEnabled(
            not busy
            and any(
                record.final_status is DatasetStatus.ACCEPTED
                for record in self.records
            )
        )
        if not busy:
            self._sync_vision_button()
        else:
            self.vision_button.setEnabled(False)

    def _set_reference(self) -> None:
        if self._is_busy():
            return
        record = self._current_record()
        if record is None or record.face_box is None:
            QMessageBox.warning(
                self,
                "기준 인물 지정 실패",
                "얼굴이 검출된 항목을 선택해줘.",
            )
            return

        reference = self.identity_matcher.build_reference(record)
        if reference is None:
            QMessageBox.warning(
                self,
                "기준 인물 지정 실패",
                "기준 얼굴 descriptor를 만들지 못했어.",
            )
            return

        self.reference_identity = reference
        self.reference_label.setText(f"기준 인물: {record.display_name}")
        self.classify_button.setEnabled(True)
        self.analysis_status_label.setText(
            "기준 인물 준비 완료 / SFace 임베딩 사용"
        )

    def _classify_identity(self) -> None:
        if self.reference_identity is None or self._is_busy():
            return

        self.analysis_progress.setRange(0, len(self.records))
        self.analysis_progress.setValue(0)
        self.analysis_status_label.setText(
            f"SFace 동일인물 판정 준비 중... 0 / {len(self.records)}"
        )
        self._set_analysis_busy(True)

        self.identity_thread = QThread(self)
        self.identity_worker = IdentityWorker(
            self.records,
            self.reference_identity,
        )
        self.identity_worker.moveToThread(self.identity_thread)

        self.identity_thread.started.connect(self.identity_worker.run)
        self.identity_worker.progress.connect(self._on_identity_progress)
        self.identity_worker.finished.connect(self._on_identity_finished)
        self.identity_worker.failed.connect(self._on_identity_failed)

        self.identity_worker.finished.connect(self.identity_thread.quit)
        self.identity_worker.failed.connect(self.identity_thread.quit)
        self.identity_worker.finished.connect(self.identity_worker.deleteLater)
        self.identity_worker.failed.connect(self.identity_worker.deleteLater)

        self.identity_thread.finished.connect(self._on_identity_thread_finished)
        self.identity_thread.finished.connect(self.identity_thread.deleteLater)
        self.identity_thread.start()

    def _on_identity_progress(self, current: int, total: int) -> None:
        self.analysis_progress.setRange(0, max(1, total))
        self.analysis_progress.setValue(current)
        self.analysis_status_label.setText(
            f"SFace 동일인물 판정 중... {current} / {total}"
        )

    def _on_identity_finished(self) -> None:
        self._populate_list()
        self._refresh_counts()
        self._sync_export_button()
        self.analysis_progress.setValue(len(self.records))
        self.analysis_status_label.setText(
            f"SFace 동일인물 판정 완료 / {len(self.records)}장"
        )
        self._set_analysis_busy(False)
        self._sync_vision_button()

    def _on_identity_failed(self, message: str) -> None:
        self.analysis_status_label.setText("동일인물 판정 실패")
        self._set_analysis_busy(False)
        QMessageBox.critical(
            self,
            "동일인물 판정 실패",
            "SFace 동일인물 판정 중 오류가 발생했어.\n\n" + message,
        )

    def _on_identity_thread_finished(self) -> None:
        self.identity_thread = None
        self.identity_worker = None
        self._sync_vision_button()

    def _vision_candidates(self) -> list[ImageRecord]:
        candidates: list[ImageRecord] = []
        for record in self.records:
            if record.crop_box is None:
                continue
            if record.auto_status is DatasetStatus.REJECTED:
                continue

            quality = record.quality_score
            if (
                record.auto_status is DatasetStatus.REVIEW
                or quality is None
                or quality < 88
                or record.crop_touches_edge
                or record.detected_faces_count > 1
            ):
                candidates.append(record)

        return candidates

    def _sync_vision_button(self) -> None:
        if not hasattr(self, "vision_button"):
            return

        candidates = self._vision_candidates()
        self.vision_target_label.setText(f"대상 {len(candidates)}")
        self.vision_button.setEnabled(
            not self._is_busy()
            and bool(candidates)
            and bool(self.vision_model_edit.text().strip())
        )

    def _run_vision_review(self) -> None:
        if self._is_busy():
            return

        model = self.vision_model_edit.text().strip()
        if not model:
            QMessageBox.warning(
                self,
                "Vision 모델 필요",
                "Ollama에서 사용할 Vision 모델 이름을 입력해줘.",
            )
            return

        candidates = self._vision_candidates()
        if not candidates:
            QMessageBox.information(
                self,
                "Vision 검수",
                "현재 Vision 2차 검수가 필요한 후보가 없어.",
            )
            return

        self.analysis_progress.setRange(0, len(candidates))
        self.analysis_progress.setValue(0)
        self.analysis_status_label.setText(
            f"Vision 검수 준비 중... 0 / {len(candidates)}"
        )
        self._set_analysis_busy(True)

        self.vision_thread = QThread(self)
        self.vision_worker = VisionWorker(
            candidates,
            self.vision_url_edit.text().strip(),
            model,
        )
        self.vision_worker.moveToThread(self.vision_thread)

        self.vision_thread.started.connect(self.vision_worker.run)
        self.vision_worker.progress.connect(self._on_vision_progress)
        self.vision_worker.finished.connect(self._on_vision_finished)
        self.vision_worker.failed.connect(self._on_vision_failed)

        self.vision_worker.finished.connect(self.vision_thread.quit)
        self.vision_worker.failed.connect(self.vision_thread.quit)
        self.vision_worker.finished.connect(self.vision_worker.deleteLater)
        self.vision_worker.failed.connect(self.vision_worker.deleteLater)

        self.vision_thread.finished.connect(
            self._on_vision_thread_finished
        )
        self.vision_thread.finished.connect(
            self.vision_thread.deleteLater
        )
        self.vision_thread.start()

    def _on_vision_progress(self, current: int, total: int) -> None:
        self.analysis_progress.setRange(0, max(1, total))
        self.analysis_progress.setValue(current)
        self.analysis_status_label.setText(
            f"Vision 검수 중... {current} / {total}"
        )

    def _on_vision_finished(self, reviewed: int, errors: int) -> None:
        self._populate_list()
        self._refresh_counts()
        self._sync_export_button()
        self.analysis_progress.setValue(
            self.analysis_progress.maximum()
        )
        self.analysis_status_label.setText(
            f"Vision 검수 완료 / 성공 {reviewed} / 오류 {errors}"
        )
        self._set_analysis_busy(False)
        self._sync_vision_button()

        if errors:
            QMessageBox.warning(
                self,
                "Vision 검수 일부 실패",
                f"{errors}개 항목에서 Vision 응답 오류가 발생했어. "
                "각 항목의 Vision 메모에서 오류를 확인할 수 있어.",
            )

    def _on_vision_failed(self, message: str) -> None:
        self.analysis_status_label.setText("Vision 검수 실패")
        self._set_analysis_busy(False)
        self._sync_vision_button()
        QMessageBox.critical(
            self,
            "Vision 검수 실패",
            "Vision 검수 중 오류가 발생했어.\n\n" + message,
        )

    def _on_vision_thread_finished(self) -> None:
        self.vision_thread = None
        self.vision_worker = None
        self._sync_vision_button()

    def _export_dataset(self) -> None:
        if not self.records:
            return

        selected = QFileDialog.getExistingDirectory(self, "Export 폴더 선택")
        if not selected:
            return

        try:
            self.exporter.export(Path(selected), self.records)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export 실패", str(exc))
            return

        QMessageBox.information(
            self,
            "Export 완료",
            "accepted / review / rejected / logs 폴더로 내보냈어.",
        )

    def _populate_list(self) -> None:
        self.list_widget.clear()

        for index, record in enumerate(self.records):
            item = QListWidgetItem()
            item.setData(ROLE_RECORD_INDEX, index)
            item.setText(self._item_text(record))

            icon = self._record_icon(record)
            if icon is not None:
                item.setIcon(icon)

            self.list_widget.addItem(item)

        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)

    @staticmethod
    def _status_color(status: DatasetStatus) -> str:
        if status is DatasetStatus.ACCEPTED:
            return "#3b82f6"
        if status is DatasetStatus.REVIEW:
            return "#f59e0b"
        return "#ef4444"

    def _record_icon(self, record: ImageRecord) -> QIcon | None:
        pixmap = QPixmap(str(record.source_file))
        if pixmap.isNull():
            return None

        icon_size = self.list_widget.iconSize()
        border_width = 5
        inset = border_width + 3
        image_size = QSize(
            max(1, icon_size.width() - inset * 2),
            max(1, icon_size.height() - inset * 2),
        )
        scaled = pixmap.scaled(
            image_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        canvas = QPixmap(icon_size)
        canvas.fill(Qt.GlobalColor.transparent)

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        x = (icon_size.width() - scaled.width()) // 2
        y = (icon_size.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

        pen = QPen(QColor(self._status_color(record.final_status)))
        pen.setWidth(border_width)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(
            border_width // 2,
            border_width // 2,
            icon_size.width() - border_width,
            icon_size.height() - border_width,
            7,
            7,
        )
        painter.end()
        return QIcon(canvas)

    def _apply_preview_status_border(
        self,
        status: DatasetStatus,
    ) -> None:
        color = self._status_color(status)
        style = (
            f"border: 3px solid {color}; "
            "border-radius: 5px; "
            "background: #111317;"
        )
        self.original_label.setStyleSheet(style)
        self.crop_label.setStyleSheet(style)

    def _item_text(self, record: ImageRecord) -> str:
        ref = "[REF] " if record.is_reference else ""
        similarity = (
            f"\nSIM {record.identity_similarity:.3f}"
            if record.identity_similarity is not None
            else ""
        )

        if (
            record.face_box is None
            and not record.auto_reasons
            and not record.caption
        ):
            reasons = "미분석"
        else:
            reasons = (
                ", ".join(record.auto_reasons)
                if record.auto_reasons
                else "ok"
            )

        return (
            f"{ref}[{record.final_status.value}]\n"
            f"{record.display_name}{similarity}\n{reasons}"
        )

    def _current_record(self) -> ImageRecord | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None

        index = item.data(ROLE_RECORD_INDEX)
        if not isinstance(index, int):
            return None
        if not 0 <= index < len(self.records):
            return None
        return self.records[index]

    def _show_current_item(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return

        record = self._current_record()
        if record is None:
            return

        self._set_preview(
            self.original_label,
            self._original_pixmap(record),
            "원본",
        )
        self._set_preview(
            self.crop_label,
            self._crop_pixmap(record),
            "크롭",
        )
        self._apply_preview_status_border(record.final_status)

        reasons = (
            ", ".join(record.auto_reasons)
            if record.auto_reasons
            else "-"
        )
        similarity = (
            f"{record.identity_similarity:.3f}"
            if record.identity_similarity is not None
            else "-"
        )
        confidence = (
            f"{record.detection_confidence:.3f}"
            if record.detection_confidence is not None
            else "-"
        )
        quality = (
            f"{record.quality_score:.1f}"
            if record.quality_score is not None
            else "-"
        )
        blur = (
            f"{record.blur_score:.1f}"
            if record.blur_score is not None
            else "-"
        )
        face_px = (
            str(record.face_pixel_size)
            if record.face_pixel_size is not None
            else "-"
        )
        vision_status = (
            record.vision_status.value
            if record.vision_status is not None
            else "-"
        )

        self.file_label.setText(str(record.source_file))
        self.meta_label.setText(
            f"자동 상태: {record.auto_status.value}\n"
            f"최종 상태: {record.final_status.value}\n"
            f"자동 사유: {reasons}\n"
            f"검출 얼굴 수: {record.detected_faces_count}\n"
            f"얼굴 신뢰도: {confidence}\n"
            f"방향: {record.direction_caption or '-'}\n"
            f"유사도: {similarity}\n"
            f"품질 점수: {quality}\n"
            f"블러 점수: {blur}\n"
            f"얼굴 픽셀: {face_px}\n"
            f"크롭 경계 접촉: {'예' if record.crop_touches_edge else '아니오'}\n"
            f"Vision 상태: {vision_status}\n"
            f"Vision 적합성: {record.vision_usable_for_lora or '-'}\n"
            f"Vision 가림: {record.vision_occlusion or '-'}\n"
            f"Vision 크롭: {record.vision_crop_quality or '-'}\n"
            f"Vision 메모: {record.vision_notes or '-'}\n"
            f"기준 인물: {'예' if record.is_reference else '아니오'}\n"
            f"사용자 수정: {'예' if record.user_override else '아니오'}"
        )
        self.caption_edit.setText(record.caption)

        if not self._is_busy():
            self.reference_button.setEnabled(record.face_box is not None)

    def _set_preview(
        self,
        label: QLabel,
        pixmap: QPixmap | None,
        fallback: str,
    ) -> None:
        if pixmap is None or pixmap.isNull():
            label.setPixmap(QPixmap())
            label.setText(fallback)
            return

        label.setText("")
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    @staticmethod
    def _original_pixmap(record: ImageRecord) -> QPixmap | None:
        image = cv2.imdecode(
            np.fromfile(record.source_file, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            return None

        if record.face_box is not None:
            x, y, w, h = record.face_box.as_tuple()
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                max(2, int(round(max(image.shape[:2]) / 700))),
            )

            if record.detection_confidence is not None:
                cv2.putText(
                    image,
                    f"{record.detection_confidence:.2f}",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        qimage = QImage(
            image.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        )
        return QPixmap.fromImage(qimage.copy())

    @staticmethod
    def _crop_pixmap(record: ImageRecord) -> QPixmap | None:
        if record.crop_box is None:
            return None

        image = cv2.imdecode(
            np.fromfile(record.source_file, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            return None

        x, y, w, h = record.crop_box.as_tuple()
        crop = image[y : y + h, x : x + w]
        if crop.size == 0:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        height, width, channels = crop.shape
        qimage = QImage(
            crop.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        )
        return QPixmap.fromImage(qimage.copy())

    def _set_current_status(self, status: DatasetStatus) -> None:
        record = self._current_record()
        item = self.list_widget.currentItem()
        if record is None or item is None:
            return

        record.set_final_status(status)
        item.setText(self._item_text(record))
        icon = self._record_icon(record)
        if icon is not None:
            item.setIcon(icon)
        self._show_current_item(item, None)
        self._refresh_counts()
        self._sync_export_button()
        self._sync_vision_button()

    def _save_caption(self) -> None:
        record = self._current_record()
        if record is not None:
            record.caption = self.caption_edit.text().strip()

    def _refresh_counts(self) -> None:
        accepted = sum(
            record.final_status is DatasetStatus.ACCEPTED
            for record in self.records
        )
        review = sum(
            record.final_status is DatasetStatus.REVIEW
            for record in self.records
        )
        rejected = sum(
            record.final_status is DatasetStatus.REJECTED
            for record in self.records
        )
        self.count_label.setText(
            f"전체 {len(self.records)}  |  "
            f"채택 {accepted}  |  보류 {review}  |  제외 {rejected}"
        )

    def _sync_export_button(self) -> None:
        self.export_button.setEnabled(
            not self._is_busy()
            and any(
                record.final_status is DatasetStatus.ACCEPTED
                for record in self.records
            )
        )
