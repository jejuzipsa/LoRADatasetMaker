from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
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

from loradatasetmaker.core.domain import DatasetStatus, ImageRecord
from loradatasetmaker.core.indexer import index_image_folder


ROLE_RECORD_INDEX = Qt.ItemDataRole.UserRole


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[ImageRecord] = []
        self.current_folder: Path | None = None

        self.setWindowTitle("LoRA Dataset Maker")
        self.resize(1440, 900)
        self.setMinimumSize(1080, 700)

        self._build_ui()
        self._apply_dark_theme()
        self._refresh_counts()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        controls = QHBoxLayout()
        self.folder_button = QPushButton("사진 폴더 선택")
        self.folder_button.clicked.connect(self._choose_folder)
        controls.addWidget(self.folder_button)

        self.folder_label = QLabel("선택된 폴더 없음")
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        controls.addWidget(self.folder_label, 1)

        controls.addWidget(QLabel("Trigger"))
        self.trigger_edit = QLineEdit()
        self.trigger_edit.setPlaceholderText("예: personA")
        self.trigger_edit.setMaximumWidth(220)
        controls.addWidget(self.trigger_edit)

        self.analyze_button = QPushButton("자동 분석")
        self.analyze_button.setEnabled(False)
        self.analyze_button.setToolTip("얼굴 검출/동일인물/품질 분석은 다음 패치에서 연결")
        controls.addWidget(self.analyze_button)

        root_layout.addLayout(controls)

        status_bar = QHBoxLayout()
        self.count_label = QLabel()
        status_bar.addWidget(self.count_label)
        status_bar.addStretch(1)
        root_layout.addLayout(status_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, 1)

        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setMovement(QListWidget.Movement.Static)
        self.list_widget.setIconSize(QSize(180, 180))
        self.list_widget.setGridSize(QSize(210, 235))
        self.list_widget.setSpacing(6)
        self.list_widget.currentItemChanged.connect(self._show_current_item)
        splitter.addWidget(self.list_widget)

        detail_frame = QFrame()
        detail_frame.setMinimumWidth(360)
        detail_layout = QVBoxLayout(detail_frame)

        self.preview_label = QLabel("이미지를 선택해줘")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(320, 320)
        self.preview_label.setStyleSheet("border: 1px solid #3b3f46;")
        detail_layout.addWidget(self.preview_label, 1)

        self.file_label = QLabel("-")
        self.file_label.setWordWrap(True)
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        detail_layout.addWidget(self.file_label)

        self.meta_label = QLabel("-")
        self.meta_label.setWordWrap(True)
        detail_layout.addWidget(self.meta_label)

        detail_layout.addWidget(QLabel("Caption"))
        self.caption_edit = QLineEdit()
        self.caption_edit.setPlaceholderText("자동 캡션 생성 후 여기서 수정")
        self.caption_edit.editingFinished.connect(self._save_caption)
        detail_layout.addWidget(self.caption_edit)

        state_buttons = QHBoxLayout()
        self.accept_button = QPushButton("채택")
        self.review_button = QPushButton("보류")
        self.reject_button = QPushButton("제외")
        self.accept_button.clicked.connect(
            lambda: self._set_current_status(DatasetStatus.ACCEPTED)
        )
        self.review_button.clicked.connect(
            lambda: self._set_current_status(DatasetStatus.REVIEW)
        )
        self.reject_button.clicked.connect(
            lambda: self._set_current_status(DatasetStatus.REJECTED)
        )
        state_buttons.addWidget(self.accept_button)
        state_buttons.addWidget(self.review_button)
        state_buttons.addWidget(self.reject_button)
        detail_layout.addLayout(state_buttons)

        splitter.addWidget(detail_frame)
        splitter.setSizes([1000, 400])

        self.setCentralWidget(root)

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
            QListWidget::item {
                border: 1px solid transparent;
                padding: 5px;
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
            records = index_image_folder(folder)
        except OSError as exc:
            QMessageBox.critical(self, "폴더 읽기 실패", str(exc))
            return

        self.current_folder = folder
        self.records = records
        self.folder_label.setText(str(folder))
        self._populate_list()
        self._refresh_counts()

        if not records:
            QMessageBox.information(
                self,
                "이미지 없음",
                "JPG / JPEG / PNG / WEBP 파일을 찾지 못했어.",
            )

    def _populate_list(self) -> None:
        self.list_widget.clear()

        for index, record in enumerate(self.records):
            item = QListWidgetItem()
            item.setData(ROLE_RECORD_INDEX, index)
            item.setText(self._item_text(record))

            pixmap = QPixmap(str(record.source_file))
            if not pixmap.isNull():
                thumbnail = pixmap.scaled(
                    self.list_widget.iconSize(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                item.setIcon(QIcon(thumbnail))

            self.list_widget.addItem(item)

        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)

    def _item_text(self, record: ImageRecord) -> str:
        return f"[{record.final_status.value}]\n{record.display_name}"

    def _current_record(self) -> ImageRecord | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        index = item.data(ROLE_RECORD_INDEX)
        if not isinstance(index, int) or not (0 <= index < len(self.records)):
            return None
        return self.records[index]

    def _show_current_item(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            self.preview_label.setText("이미지를 선택해줘")
            return

        record = self._current_record()
        if record is None:
            return

        pixmap = QPixmap(str(record.source_file))
        if pixmap.isNull():
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("미리보기 실패")
        else:
            scaled = pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setText("")
            self.preview_label.setPixmap(scaled)

        self.file_label.setText(str(record.source_file))
        self.meta_label.setText(
            f"자동 상태: {record.auto_status.value}\n"
            f"최종 상태: {record.final_status.value}\n"
            f"사용자 수정: {'예' if record.user_override else '아니오'}\n"
            f"방향: {record.direction_caption or '-'}"
        )
        self.caption_edit.setText(record.caption)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._current_record() is not None:
            self._show_current_item(self.list_widget.currentItem(), None)

    def _set_current_status(self, status: DatasetStatus) -> None:
        record = self._current_record()
        item = self.list_widget.currentItem()
        if record is None or item is None:
            return

        record.set_final_status(status)
        item.setText(self._item_text(record))
        self._show_current_item(item, None)
        self._refresh_counts()

    def _save_caption(self) -> None:
        record = self._current_record()
        if record is not None:
            record.caption = self.caption_edit.text().strip()

    def _refresh_counts(self) -> None:
        total = len(self.records)
        accepted = sum(r.final_status is DatasetStatus.ACCEPTED for r in self.records)
        review = sum(r.final_status is DatasetStatus.REVIEW for r in self.records)
        rejected = sum(r.final_status is DatasetStatus.REJECTED for r in self.records)
        self.count_label.setText(
            f"전체 {total}  |  채택 {accepted}  |  보류 {review}  |  제외 {rejected}"
        )
