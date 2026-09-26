from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np
from PySide6.QtCore import QProcess, QSize, QThread, Qt
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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
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
from loradatasetmaker.core.training import TrainingPreparer, TrainingPrepOptions
from loradatasetmaker.ui.analysis_worker import AnalysisWorker
from loradatasetmaker.ui.identity_worker import IdentityWorker
from loradatasetmaker.ui.musubi_install_worker import (
    MUSUBI_VERSION,
    MusubiInstallWorker,
    default_tools_root,
)
from loradatasetmaker.ui.vision_worker import VisionWorker


ROLE_RECORD_INDEX = Qt.ItemDataRole.UserRole


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[ImageRecord] = []
        self.reference_identity: ReferenceIdentity | None = None
        self.identity_matcher = IdentityMatcher()
        self.exporter = DatasetExporter()
        self.training_preparer = TrainingPreparer()
        self.training_run_script: Path | None = None
        self.training_process: QProcess | None = None
        self.training_stage = 0
        self.training_log_tail: list[str] = []

        self.analysis_thread: QThread | None = None
        self.analysis_worker: AnalysisWorker | None = None
        self.identity_thread: QThread | None = None
        self.identity_worker: IdentityWorker | None = None
        self.vision_thread: QThread | None = None
        self.vision_worker: VisionWorker | None = None
        self.musubi_thread: QThread | None = None
        self.musubi_worker: MusubiInstallWorker | None = None

        self.setWindowTitle("LoRA Dataset Maker")
        self.resize(1500, 920)
        self.setMinimumSize(1180, 760)
        self.setAcceptDrops(True)

        self._build_ui()
        self._apply_dark_theme()
        self._refresh_counts()

    def _build_ui(self) -> None:
        dataset_page = QWidget()
        layout = QVBoxLayout(dataset_page)
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

        self.tabs = QTabWidget()
        self.tabs.addTab(dataset_page, "Dataset")
        self.tabs.addTab(self._build_training_page(), "Training")
        self.setCentralWidget(self.tabs)

    def _build_training_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Identity LoRA Training - Control 없음")
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        layout.addWidget(title)

        note = QLabel(
            "기본 Training은 accepted 얼굴 데이터셋만 쓰는 Identity LoRA 모드야. "
            "Control/source 이미지는 필요 없어. Musubi의 표준 Qwen-Image "
            "(model_version=original)로 학습하며, Edit-2511 직접 학습은 "
            "control/source pair가 필요한 별도 방식이라 여기서는 사용하지 않아."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        data_group = QGroupBox("Dataset")
        data_layout = QVBoxLayout(data_group)

        self.training_dataset_edit = QLineEdit()
        self.training_dataset_edit.setPlaceholderText("accepted 폴더")
        data_layout.addLayout(
            self._training_path_row(
                "Accepted",
                self.training_dataset_edit,
                False,
            )
        )

        self.training_workspace_edit = QLineEdit()
        self.training_workspace_edit.setPlaceholderText("학습 작업 폴더")
        data_layout.addLayout(
            self._training_path_row(
                "Workspace",
                self.training_workspace_edit,
                False,
            )
        )

        trigger_row = QHBoxLayout()
        trigger_row.addWidget(QLabel("Trigger"))
        self.training_trigger_edit = QLineEdit()
        self.training_trigger_edit.setPlaceholderText("예: personA")
        trigger_row.addWidget(self.training_trigger_edit, 1)
        trigger_row.addWidget(QLabel("Output name"))
        self.training_output_name_edit = QLineEdit("identity_qwen")
        trigger_row.addWidget(self.training_output_name_edit, 1)
        data_layout.addLayout(trigger_row)
        layout.addWidget(data_group)

        engine_group = QGroupBox("Musubi Tuner / Model")
        engine_layout = QVBoxLayout(engine_group)

        comfy_row = QHBoxLayout()
        comfy_label = QLabel("ComfyUI")
        comfy_label.setMinimumWidth(95)
        comfy_row.addWidget(comfy_label)
        self.training_comfyui_edit = QLineEdit()
        self.training_comfyui_edit.setPlaceholderText(
            "ComfyUI 루트 폴더 (예: D:\\ComfyUI)"
        )
        comfy_row.addWidget(self.training_comfyui_edit, 1)
        comfy_browse = QPushButton("찾기")
        comfy_browse.clicked.connect(
            lambda: self._choose_training_path(
                self.training_comfyui_edit,
                False,
            )
        )
        comfy_row.addWidget(comfy_browse)
        comfy_scan = QPushButton("모델 자동 검색")
        comfy_scan.clicked.connect(self._scan_comfyui_models)
        comfy_row.addWidget(comfy_scan)
        engine_layout.addLayout(comfy_row)

        self.training_model_scan_label = QLabel(
            "ComfyUI 경로를 지정하면 Qwen-Image DiT / VAE / Text Encoder를 자동으로 찾아줘."
        )
        self.training_model_scan_label.setWordWrap(True)
        engine_layout.addWidget(self.training_model_scan_label)

        musubi_row = QHBoxLayout()
        musubi_label = QLabel("Musubi")
        musubi_label.setMinimumWidth(95)
        musubi_row.addWidget(musubi_label)
        self.training_musubi_edit = QLineEdit()
        self.training_musubi_edit.setPlaceholderText("musubi-tuner 폴더")
        musubi_row.addWidget(self.training_musubi_edit, 1)
        musubi_browse = QPushButton("찾기")
        musubi_browse.clicked.connect(
            lambda: self._choose_training_path(
                self.training_musubi_edit,
                False,
            )
        )
        musubi_row.addWidget(musubi_browse)
        self.training_musubi_install_button = QPushButton("자동 설치")
        self.training_musubi_install_button.clicked.connect(
            self._install_musubi
        )
        musubi_row.addWidget(self.training_musubi_install_button)
        engine_layout.addLayout(musubi_row)

        self.training_musubi_status_label = QLabel(
            f"자동 설치: Musubi Tuner {MUSUBI_VERSION} + 전용 Python 3.11 + CUDA 12.8 환경"
        )
        self.training_musubi_status_label.setWordWrap(True)
        engine_layout.addWidget(self.training_musubi_status_label)

        self.training_python_edit = QLineEdit()
        self.training_python_edit.setPlaceholderText(
            "Musubi 전용 .venv의 python.exe"
        )
        engine_layout.addLayout(
            self._training_path_row("Python", self.training_python_edit, True)
        )

        self.training_dit_edit = QLineEdit()
        self.training_dit_edit.setPlaceholderText(
            "qwen_image_bf16.safetensors"
        )
        engine_layout.addLayout(
            self._training_path_row(
                "Qwen-Image DiT",
                self.training_dit_edit,
                True,
            )
        )

        self.training_vae_edit = QLineEdit()
        self.training_vae_edit.setPlaceholderText(
            "qwen_image_vae.safetensors"
        )
        engine_layout.addLayout(
            self._training_path_row("VAE", self.training_vae_edit, True)
        )

        self.training_text_encoder_edit = QLineEdit()
        self.training_text_encoder_edit.setPlaceholderText(
            "qwen_2.5_vl_7b*.safetensors"
        )
        engine_layout.addLayout(
            self._training_path_row(
                "Text Encoder",
                self.training_text_encoder_edit,
                True,
            )
        )

        self.training_output_dir_edit = QLineEdit()
        self.training_output_dir_edit.setPlaceholderText("LoRA 출력 폴더")
        engine_layout.addLayout(
            self._training_path_row(
                "Output folder",
                self.training_output_dir_edit,
                False,
            )
        )
        layout.addWidget(engine_group)

        settings_group = QGroupBox("1차 학습 설정")
        settings = QHBoxLayout(settings_group)

        settings.addWidget(QLabel("Resolution"))
        self.training_resolution_spin = QSpinBox()
        self.training_resolution_spin.setRange(512, 1536)
        self.training_resolution_spin.setSingleStep(64)
        self.training_resolution_spin.setValue(1024)
        settings.addWidget(self.training_resolution_spin)

        settings.addWidget(QLabel("Rank"))
        self.training_rank_spin = QSpinBox()
        self.training_rank_spin.setRange(4, 256)
        self.training_rank_spin.setValue(16)
        settings.addWidget(self.training_rank_spin)

        settings.addWidget(QLabel("Epoch"))
        self.training_epoch_spin = QSpinBox()
        self.training_epoch_spin.setRange(1, 100)
        self.training_epoch_spin.setValue(8)
        settings.addWidget(self.training_epoch_spin)

        settings.addWidget(QLabel("LR"))
        self.training_lr_edit = QLineEdit("5e-5")
        self.training_lr_edit.setMaximumWidth(90)
        settings.addWidget(self.training_lr_edit)

        settings.addWidget(QLabel("Blocks swap"))
        self.training_blocks_spin = QSpinBox()
        self.training_blocks_spin.setRange(0, 60)
        self.training_blocks_spin.setValue(45)
        settings.addWidget(self.training_blocks_spin)
        settings.addStretch(1)
        layout.addWidget(settings_group)

        memory_note = QLabel(
            "12GB VRAM에서는 block swap/fp8 절약 옵션이 필요할 수 있고 "
            "시스템 RAM 사용량이 크게 늘 수 있어. 이 모드는 Control 없는 "
            "Qwen-Image Identity LoRA 학습이야. Edit-2511에서의 사용성은 "
            "첫 LoRA 결과를 실제 ComfyUI에서 확인하면서 판단하면 돼."
        )
        memory_note.setWordWrap(True)
        layout.addWidget(memory_note)

        action_row = QHBoxLayout()
        self.training_prepare_button = QPushButton("학습 준비")
        self.training_prepare_button.clicked.connect(self._prepare_training)
        action_row.addWidget(self.training_prepare_button)

        self.training_run_button = QPushButton("Train 실행")
        self.training_run_button.setEnabled(False)
        self.training_run_button.clicked.connect(self._run_training)
        action_row.addWidget(self.training_run_button)

        self.training_stop_button = QPushButton("중지")
        self.training_stop_button.setEnabled(False)
        self.training_stop_button.clicked.connect(self._stop_training)
        action_row.addWidget(self.training_stop_button)

        self.training_status_label = QLabel("대기")
        action_row.addWidget(self.training_status_label, 1)
        layout.addLayout(action_row)

        progress_row = QHBoxLayout()
        self.training_stage_label = QLabel("대기")
        self.training_stage_label.setMinimumWidth(150)
        progress_row.addWidget(self.training_stage_label)

        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_progress.setTextVisible(True)
        progress_row.addWidget(self.training_progress, 1)

        self.training_detail_label = QLabel("-")
        self.training_detail_label.setMinimumWidth(240)
        progress_row.addWidget(self.training_detail_label)
        layout.addLayout(progress_row)

        self.training_command_preview = QPlainTextEdit()
        self.training_command_preview.setReadOnly(True)
        self.training_command_preview.document().setMaximumBlockCount(2500)
        self.training_command_preview.setPlaceholderText(
            "학습 준비 전에는 명령 미리보기, Train 실행 후에는 실시간 로그가 표시돼."
        )
        layout.addWidget(self.training_command_preview, 1)

        self._detect_local_musubi_install()
        return page

    def _training_path_row(
        self,
        title: str,
        edit: QLineEdit,
        file_mode: bool,
    ) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel(title)
        label.setMinimumWidth(95)
        row.addWidget(label)
        row.addWidget(edit, 1)
        button = QPushButton("찾기")
        button.clicked.connect(
            lambda _checked=False, target=edit, is_file=file_mode:
            self._choose_training_path(target, is_file)
        )
        row.addWidget(button)
        return row

    def _choose_training_path(
        self,
        target: QLineEdit,
        file_mode: bool,
    ) -> None:
        if file_mode:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "파일 선택",
            )
        else:
            selected = QFileDialog.getExistingDirectory(
                self,
                "폴더 선택",
            )
        if selected:
            target.setText(selected)

    def _detect_local_musubi_install(self) -> None:
        tools_root = default_tools_root()
        musubi_dir = tools_root / "musubi-tuner"
        python_exe = musubi_dir / ".venv" / "Scripts" / "python.exe"
        if musubi_dir.is_dir() and python_exe.is_file():
            self.training_musubi_edit.setText(str(musubi_dir))
            self.training_python_edit.setText(str(python_exe))
            self.training_musubi_status_label.setText(
                f"설치됨: {MUSUBI_VERSION} / 전용 Python 환경 확인됨"
            )

    def _install_musubi(self) -> None:
        if self.musubi_thread is not None:
            return

        self.training_musubi_install_button.setEnabled(False)
        self.training_prepare_button.setEnabled(False)
        self.training_run_button.setEnabled(False)
        self.training_musubi_status_label.setText(
            "Musubi 자동 설치 준비 중..."
        )

        self.musubi_thread = QThread(self)
        self.musubi_worker = MusubiInstallWorker()
        self.musubi_worker.moveToThread(self.musubi_thread)

        self.musubi_thread.started.connect(self.musubi_worker.run)
        self.musubi_worker.status.connect(
            self.training_musubi_status_label.setText
        )
        self.musubi_worker.finished.connect(
            self._on_musubi_install_finished
        )
        self.musubi_worker.failed.connect(
            self._on_musubi_install_failed
        )

        self.musubi_worker.finished.connect(self.musubi_thread.quit)
        self.musubi_worker.failed.connect(self.musubi_thread.quit)
        self.musubi_worker.finished.connect(
            self.musubi_worker.deleteLater
        )
        self.musubi_worker.failed.connect(
            self.musubi_worker.deleteLater
        )
        self.musubi_thread.finished.connect(
            self._on_musubi_install_thread_finished
        )
        self.musubi_thread.finished.connect(
            self.musubi_thread.deleteLater
        )
        self.musubi_thread.start()

    def _on_musubi_install_finished(
        self,
        musubi_path: str,
        python_path: str,
        version: str,
    ) -> None:
        self.training_musubi_edit.setText(musubi_path)
        self.training_python_edit.setText(python_path)
        self.training_musubi_status_label.setText(
            f"설치 완료: Musubi Tuner {version} / 전용 Python 3.11"
        )
        self.training_prepare_button.setEnabled(True)
        QMessageBox.information(
            self,
            "Musubi 설치 완료",
            "Musubi Tuner와 전용 Python 환경 설치가 끝났어. "
            "이제 학습 준비를 누르면 돼.",
        )

    def _on_musubi_install_failed(self, message: str) -> None:
        self.training_musubi_status_label.setText("Musubi 설치 실패")
        self.training_prepare_button.setEnabled(True)
        QMessageBox.critical(
            self,
            "Musubi 자동 설치 실패",
            message,
        )

    def _on_musubi_install_thread_finished(self) -> None:
        self.musubi_thread = None
        self.musubi_worker = None
        self.training_musubi_install_button.setEnabled(True)

    def _scan_comfyui_models(self) -> None:
        root_text = self.training_comfyui_edit.text().strip()
        if not root_text:
            selected = QFileDialog.getExistingDirectory(
                self,
                "ComfyUI 폴더 선택",
            )
            if not selected:
                return
            root_text = selected
            self.training_comfyui_edit.setText(selected)

        root = Path(root_text)
        if not root.is_dir():
            QMessageBox.warning(
                self,
                "ComfyUI 모델 검색",
                "선택한 ComfyUI 폴더가 존재하지 않아.",
            )
            return

        models_root = root / "models"
        if not models_root.is_dir():
            QMessageBox.warning(
                self,
                "ComfyUI 모델 검색",
                "선택한 폴더 아래에서 models 폴더를 찾지 못했어.",
            )
            return

        dit = self._find_comfyui_model(
            models_root,
            preferred_dirs=("diffusion_models", "unet", "checkpoints"),
            exact_names=("qwen_image_bf16.safetensors",),
            include_tokens=("qwen", "image"),
            exclude_tokens=("edit", "layered", "vae", "text"),
        )
        vae = self._find_comfyui_model(
            models_root,
            preferred_dirs=("vae",),
            exact_names=("qwen_image_vae.safetensors",),
            include_tokens=("qwen", "image", "vae"),
            exclude_tokens=(),
        )
        text_encoder = self._find_comfyui_model(
            models_root,
            preferred_dirs=("text_encoders", "clip"),
            exact_names=("qwen_2.5_vl_7b.safetensors",),
            include_tokens=("qwen", "2.5", "vl", "7b"),
            exclude_tokens=("fp8", "scaled"),
        )

        found: list[str] = []
        missing: list[str] = []

        if dit is not None:
            self.training_dit_edit.setText(str(dit))
            found.append(f"DiT: {dit.name}")
        else:
            missing.append("Qwen-Image DiT")

        if vae is not None:
            self.training_vae_edit.setText(str(vae))
            found.append(f"VAE: {vae.name}")
        else:
            missing.append("VAE")

        if text_encoder is not None:
            self.training_text_encoder_edit.setText(str(text_encoder))
            found.append(f"Text Encoder: {text_encoder.name}")
        else:
            missing.append("Text Encoder")

        if found:
            self.training_model_scan_label.setText(
                "자동 검색: " + " / ".join(found)
            )
        else:
            self.training_model_scan_label.setText(
                "자동 검색 결과: 호환 모델을 찾지 못했어."
            )

        message = []
        if found:
            message.append("찾은 모델\n" + "\n".join(found))
        if missing:
            message.append(
                "못 찾은 항목\n" + "\n".join(missing)
                + "\n\n해당 항목만 직접 찾아서 지정하면 돼."
            )
        QMessageBox.information(
            self,
            "ComfyUI 모델 자동 검색",
            "\n\n".join(message),
        )

    @staticmethod
    def _find_comfyui_model(
        models_root: Path,
        preferred_dirs: tuple[str, ...],
        exact_names: tuple[str, ...],
        include_tokens: tuple[str, ...],
        exclude_tokens: tuple[str, ...],
    ) -> Path | None:
        search_roots: list[Path] = []
        for folder_name in preferred_dirs:
            candidate = models_root / folder_name
            if candidate.is_dir():
                search_roots.append(candidate)
        if not search_roots:
            search_roots.append(models_root)

        exact_lower = {name.lower() for name in exact_names}

        def compatible(path: Path) -> bool:
            name = path.name.lower()
            if path.suffix.lower() != ".safetensors":
                return False
            if any(token not in name for token in include_tokens):
                return False
            if any(token in name for token in exclude_tokens):
                return False
            return True

        # Exact known filenames first.
        for search_root in search_roots:
            try:
                for path in search_root.rglob("*.safetensors"):
                    if path.name.lower() in exact_lower:
                        return path
            except OSError:
                continue

        # Then accept a conservative compatible filename.
        for search_root in search_roots:
            try:
                candidates = sorted(
                    (
                        path
                        for path in search_root.rglob("*.safetensors")
                        if compatible(path)
                    ),
                    key=lambda path: (len(path.name), str(path).lower()),
                )
            except OSError:
                continue
            if candidates:
                return candidates[0]

        return None

    def _prepare_training(self) -> None:
        dataset_text = self.training_dataset_edit.text().strip()
        workspace_text = self.training_workspace_edit.text().strip()
        output_text = self.training_output_dir_edit.text().strip()
        if not dataset_text or not workspace_text or not output_text:
            QMessageBox.warning(
                self,
                "Training 준비",
                "Accepted / Workspace / Output folder를 먼저 지정해줘.",
            )
            return

        options = TrainingPrepOptions(
            dataset_dir=Path(dataset_text),
            workspace_dir=Path(workspace_text),
            musubi_dir=(
                Path(self.training_musubi_edit.text().strip())
                if self.training_musubi_edit.text().strip()
                else None
            ),
            python_exe=(
                Path(self.training_python_edit.text().strip())
                if self.training_python_edit.text().strip()
                else None
            ),
            dit_path=(
                Path(self.training_dit_edit.text().strip())
                if self.training_dit_edit.text().strip()
                else None
            ),
            vae_path=(
                Path(self.training_vae_edit.text().strip())
                if self.training_vae_edit.text().strip()
                else None
            ),
            text_encoder_path=(
                Path(self.training_text_encoder_edit.text().strip())
                if self.training_text_encoder_edit.text().strip()
                else None
            ),
            output_dir=Path(output_text),
            trigger_token=self.training_trigger_edit.text().strip(),
            output_name=self.training_output_name_edit.text().strip(),
            resolution=self.training_resolution_spin.value(),
            rank=self.training_rank_spin.value(),
            epochs=self.training_epoch_spin.value(),
            learning_rate=self.training_lr_edit.text().strip(),
            blocks_to_swap=self.training_blocks_spin.value(),
        )

        try:
            result = self.training_preparer.prepare(options)
        except Exception as exc:  # noqa: BLE001
            self.training_run_script = None
            self.training_run_button.setEnabled(False)
            QMessageBox.critical(self, "Training 준비 실패", str(exc))
            return

        self.training_run_script = result.run_script
        self.training_run_button.setEnabled(result.ready_to_train)
        self.training_command_preview.setPlainText(result.command_preview)

        if result.ready_to_train:
            self.training_status_label.setText(
                f"준비 완료 / {result.item_count}장 / Train 가능"
            )
        else:
            self.training_status_label.setText(
                f"Dataset 준비 완료 / {result.item_count}장 / 모델 경로 확인 필요"
            )

        if result.warnings:
            QMessageBox.information(
                self,
                "Training 준비 결과",
                "\n".join(result.warnings),
            )

    def _run_training(self) -> None:
        if self.training_process is not None:
            return
        if self.training_run_script is None:
            return

        script = self.training_run_script
        if not script.is_file():
            QMessageBox.warning(
                self,
                "Train 실행",
                "run_all.bat을 찾지 못했어. 학습 준비를 다시 실행해줘.",
            )
            return

        self.training_stage = 0
        self.training_log_tail = []
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_stage_label.setText("시작 준비")
        self.training_detail_label.setText("-")
        self.training_command_preview.clear()
        self.training_command_preview.appendPlainText(
            f"Training 시작: {script}\n"
        )

        process = QProcess(self)
        process.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels
        )
        process.setWorkingDirectory(str(script.parent))
        process.readyReadStandardOutput.connect(
            self._on_training_output
        )
        process.started.connect(self._on_training_started)
        process.finished.connect(self._on_training_finished)
        process.errorOccurred.connect(self._on_training_process_error)
        self.training_process = process

        self.training_prepare_button.setEnabled(False)
        self.training_run_button.setEnabled(False)
        self.training_stop_button.setEnabled(True)
        self.training_status_label.setText("Training 시작 중...")

        python_text = self.training_python_edit.text().strip()
        python_exe = Path(python_text) if python_text else None
        if python_exe is None or not python_exe.is_file():
            self.training_process = None
            process.deleteLater()
            self.training_prepare_button.setEnabled(True)
            self.training_stop_button.setEnabled(False)
            QMessageBox.critical(
                self,
                "Train 실행 실패",
                "Musubi 전용 python.exe 경로를 찾지 못했어.",
            )
            return

        process.start(
            str(python_exe),
            ["-X", "utf8", "-u", str(script)],
        )

    def _on_training_started(self) -> None:
        self.training_status_label.setText("Training 실행 중")
        self.training_stage_label.setText("1/3 Latent cache")
        self.training_progress.setValue(1)

    def _on_training_output(self) -> None:
        process = self.training_process
        if process is None:
            return

        raw = bytes(process.readAllStandardOutput())
        if not raw:
            return

        text = raw.decode("utf-8", errors="replace")
        normalized = text.replace("\r", "\n")
        clean_lines = [
            line.rstrip()
            for line in normalized.splitlines()
            if line.strip()
        ]
        if clean_lines:
            self.training_command_preview.appendPlainText(
                "\n".join(clean_lines)
            )
            self.training_log_tail.extend(clean_lines)
            self.training_log_tail = self.training_log_tail[-30:]

        self._parse_training_output(normalized)

    def _parse_training_output(self, text: str) -> None:
        stage_match = re.findall(
            r"__LDM_STAGE__\s+([123])\s+([^\r\n]+)",
            text,
        )
        for stage_text, title in stage_match:
            self.training_stage = int(stage_text)
            self.training_stage_label.setText(
                f"{self.training_stage}/3 {title.strip()}"
            )
            stage_start = {1: 0, 2: 20, 3: 35}[self.training_stage]
            self.training_progress.setValue(
                max(self.training_progress.value(), stage_start)
            )
            self.training_detail_label.setText("-")

        failed_match = re.search(
            r"__LDM_FAILED__\s+stage=(\d+)\s+code=(-?\d+)",
            text,
        )
        if failed_match:
            stage_no = int(failed_match.group(1))
            code = int(failed_match.group(2))
            self.training_stage_label.setText(
                f"{stage_no}/3 단계 실패"
            )
            self.training_detail_label.setText(
                f"종료 코드 {code}"
            )

        if "__LDM_DONE__" in text:
            self.training_progress.setValue(100)
            self.training_stage_label.setText("완료")

        percent_matches = re.findall(r"(\d{1,3})%\|", text)
        if percent_matches and self.training_stage in (1, 2, 3):
            raw_percent = max(
                0,
                min(100, int(percent_matches[-1])),
            )
            start, end = {
                1: (0, 20),
                2: (20, 35),
                3: (35, 100),
            }[self.training_stage]
            overall = start + int(
                (end - start) * raw_percent / 100
            )
            self.training_progress.setValue(
                max(self.training_progress.value(), overall)
            )

        if self.training_stage == 3:
            step_matches = re.findall(
                r"(\d+)\s*/\s*(\d+)",
                text,
            )
            loss_matches = re.findall(
                r"avr_loss=([0-9.eE+\-]+)",
                text,
            )
            details: list[str] = []
            if step_matches:
                current, total = step_matches[-1]
                details.append(f"Step {current}/{total}")
            if loss_matches:
                details.append(f"Loss {loss_matches[-1]}")
            if details:
                self.training_detail_label.setText(" | ".join(details))

    def _on_training_finished(
        self,
        exit_code: int,
        _exit_status: QProcess.ExitStatus,
    ) -> None:
        # Drain any final buffered output before releasing the process.
        self._on_training_output()

        success = exit_code == 0
        if success:
            self.training_progress.setValue(100)
            self.training_stage_label.setText("완료")
            output_dir = Path(
                self.training_output_dir_edit.text().strip()
            )
            results = (
                sorted(
                    output_dir.glob("*.safetensors"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                if output_dir.is_dir()
                else []
            )
            if results:
                self.training_status_label.setText(
                    f"학습 완료: {results[0].name}"
                )
                self.training_detail_label.setText(
                    str(results[0])
                )
            else:
                self.training_status_label.setText("학습 완료")
                self.training_detail_label.setText(
                    "출력 폴더 확인"
                )
        else:
            self.training_status_label.setText(
                f"Training 실패 / 종료 코드 {exit_code}"
            )
            self.training_stage_label.setText("실패")
            detail = "\n".join(self.training_log_tail[-12:])
            QMessageBox.critical(
                self,
                "Training 실패",
                (
                    f"학습 프로세스가 종료 코드 {exit_code}로 끝났어.\n\n"
                    f"{detail}"
                ),
            )

        process = self.training_process
        self.training_process = None
        if process is not None:
            process.deleteLater()

        self.training_prepare_button.setEnabled(True)
        self.training_run_button.setEnabled(
            self.training_run_script is not None
            and self.training_run_script.is_file()
        )
        self.training_stop_button.setEnabled(False)

    def _on_training_process_error(
        self,
        error: QProcess.ProcessError,
    ) -> None:
        if error == QProcess.ProcessError.Crashed:
            return
        self.training_status_label.setText(
            f"Training 실행 오류: {error.name}"
        )

    def _stop_training(self) -> None:
        process = self.training_process
        if process is None:
            return

        self.training_status_label.setText("Training 중지 중...")
        self.training_stop_button.setEnabled(False)
        pid = int(process.processId())
        if pid > 0:
            QProcess.startDetached(
                "taskkill.exe",
                ["/PID", str(pid), "/T", "/F"],
            )
        else:
            process.kill()

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
                or quality < 70
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

        export_root = Path(selected)
        accepted_dir = export_root / "accepted"
        self.training_dataset_edit.setText(str(accepted_dir))
        if not self.training_workspace_edit.text().strip():
            self.training_workspace_edit.setText(
                str(export_root / "training_identity_qwen")
            )
        if not self.training_output_dir_edit.text().strip():
            self.training_output_dir_edit.setText(
                str(export_root / "lora_output")
            )
        trigger = self.trigger_edit.text().strip()
        if trigger and not self.training_trigger_edit.text().strip():
            self.training_trigger_edit.setText(trigger)

        QMessageBox.information(
            self,
            "Export 완료",
            "accepted / review / rejected / logs 폴더로 내보냈어. "
            "Training 탭의 Accepted 경로도 자동으로 연결했어.",
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
