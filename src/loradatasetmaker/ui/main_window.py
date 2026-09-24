from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from loradatasetmaker.core.analyzer import FaceAnalyzer
from loradatasetmaker.core.domain import DatasetStatus, ImageRecord
from loradatasetmaker.core.exporter import DatasetExporter
from loradatasetmaker.core.identity import IdentityMatcher, ReferenceIdentity
from loradatasetmaker.core.indexer import index_image_folder


ROLE_RECORD_INDEX = Qt.ItemDataRole.UserRole


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[ImageRecord] = []
        self.reference_identity: ReferenceIdentity | None = None
        self.analyzer = FaceAnalyzer()
        self.identity_matcher = IdentityMatcher()
        self.exporter = DatasetExporter()

        self.setWindowTitle("LoRA Dataset Maker")
        self.resize(1500, 920)
        self.setMinimumSize(1180, 760)

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

        self.folder_label = QLabel("선택된 폴더 없음")
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
            QListWidget {
                background: #111317;
                border: 1px solid #30343a;
                border-radius: 5px;
            }
            QListWidget::item:selected {
                background: #30353d;
                border: 1px solid #626a75;
            }
            """
        )

    def _choose_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "사진 폴더 선택")
        if not selected:
            return

        folder = Path(selected)
        try:
            self.records = index_image_folder(folder)
        except OSError as exc:
            QMessageBox.critical(self, "폴더 읽기 실패", str(exc))
            return

        self.reference_identity = None
        self.reference_label.setText("기준 인물: 미지정")
        self.folder_label.setText(str(folder))
        self.analyze_button.setEnabled(bool(self.records))
        self.reference_button.setEnabled(False)
        self.classify_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self._populate_list()
        self._refresh_counts()

    def _run_analysis(self) -> None:
        trigger = self.trigger_edit.text().strip()
        self.reference_identity = None
        self.reference_label.setText("기준 인물: 미지정")

        for record in self.records:
            record.is_reference = False
            record.identity_similarity = None
            self.analyzer.analyze(record, trigger_token=trigger)

        self._populate_list()
        self._refresh_counts()
        self.reference_button.setEnabled(
            any(record.face_box is not None for record in self.records)
        )
        self.classify_button.setEnabled(False)
        self._sync_export_button()
        QMessageBox.information(
            self,
            "분석 완료",
            "얼굴 검출과 기본 Head Crop 분석을 끝냈어. "
            "이제 기준 인물을 지정하면 돼.",
        )

    def _set_reference(self) -> None:
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

    def _classify_identity(self) -> None:
        if self.reference_identity is None:
            return

        self.identity_matcher.classify_records(
            self.records,
            self.reference_identity,
        )
        self._populate_list()
        self._refresh_counts()
        self._sync_export_button()

    def _export_dataset(self) -> None:
        if not self.records:
            return

        selected = QFileDialog.getExistingDirectory(self, "Export 폴더 선택")
        if not selected:
            return

        try:
            self.exporter.export(Path(selected), self.records)
        except Exception as exc:
            QMessageBox.critical(self, "Export 실패", str(exc))
            return

        QMessageBox.information(
            self,
            "Export 완료",
            "accepted 폴더와 logs/decisions.json을 만들었어.",
        )

    def _populate_list(self) -> None:
        self.list_widget.clear()

        for index, record in enumerate(self.records):
            item = QListWidgetItem()
            item.setData(ROLE_RECORD_INDEX, index)
            item.setText(self._item_text(record))

            pixmap = QPixmap(str(record.source_file))
            if not pixmap.isNull():
                item.setIcon(
                    QIcon(
                        pixmap.scaled(
                            self.list_widget.iconSize(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                )

            self.list_widget.addItem(item)

        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)

    def _item_text(self, record: ImageRecord) -> str:
        ref = "[REF] " if record.is_reference else ""
        similarity = (
            f"\nSIM {record.identity_similarity:.3f}"
            if record.identity_similarity is not None
            else ""
        )
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
            self._pixmap_from_path(record.source_file),
            "원본",
        )
        self._set_preview(
            self.crop_label,
            self._crop_pixmap(record),
            "크롭",
        )

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

        self.file_label.setText(str(record.source_file))
        self.meta_label.setText(
            f"자동 상태: {record.auto_status.value}\n"
            f"최종 상태: {record.final_status.value}\n"
            f"자동 사유: {reasons}\n"
            f"검출 얼굴 수: {record.detected_faces_count}\n"
            f"방향: {record.direction_caption or '-'}\n"
            f"유사도: {similarity}\n"
            f"기준 인물: {'예' if record.is_reference else '아니오'}\n"
            f"사용자 수정: {'예' if record.user_override else '아니오'}"
        )
        self.caption_edit.setText(record.caption)
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
    def _pixmap_from_path(path: Path) -> QPixmap | None:
        pixmap = QPixmap(str(path))
        return None if pixmap.isNull() else pixmap

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
        self._show_current_item(item, None)
        self._refresh_counts()
        self._sync_export_button()

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
            any(
                record.final_status is DatasetStatus.ACCEPTED
                for record in self.records
            )
        )
