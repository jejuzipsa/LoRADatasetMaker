from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import shutil


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(slots=True)
class TrainingPrepOptions:
    dataset_dir: Path
    control_dir: Path | None
    workspace_dir: Path
    musubi_dir: Path | None
    python_exe: Path | None
    dit_path: Path | None
    vae_path: Path | None
    text_encoder_path: Path | None
    output_dir: Path
    trigger_token: str
    output_name: str
    resolution: int = 1024
    rank: int = 16
    epochs: int = 8
    learning_rate: str = "5e-5"
    blocks_to_swap: int = 45


@dataclass(slots=True)
class TrainingPrepResult:
    item_count: int
    ready_to_train: bool
    workspace_dir: Path
    dataset_config: Path | None = None
    run_script: Path | None = None
    command_preview: str = ""
    warnings: list[str] = field(default_factory=list)


class TrainingPreparer:
    """Prepare a Musubi Tuner Qwen-Image-Edit-2511 LoRA workspace.

    Qwen-Image-Edit training is pair-based: each target image needs a matching
    control/source image.  The preparer never mutates the original accepted
    dataset.  It copies targets/captions into a workspace and injects the
    trigger token there.
    """

    def prepare(self, options: TrainingPrepOptions) -> TrainingPrepResult:
        dataset_dir = options.dataset_dir.resolve()
        workspace = options.workspace_dir.resolve()
        output_dir = options.output_dir.resolve()

        if not dataset_dir.is_dir():
            raise ValueError("accepted 데이터셋 폴더가 존재하지 않아.")

        trigger = options.trigger_token.strip().strip(",")
        if not trigger:
            raise ValueError("Trigger token을 입력해줘.")

        output_name = options.output_name.strip()
        if not output_name:
            raise ValueError("Output name을 입력해줘.")

        if options.resolution < 256:
            raise ValueError("Resolution 값이 너무 작아.")
        if options.rank < 1 or options.epochs < 1:
            raise ValueError("Rank/Epoch 값은 1 이상이어야 해.")

        images = sorted(
            path for path in dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError("accepted 폴더에서 학습 이미지를 찾지 못했어.")

        missing_captions = [
            image.name for image in images
            if not image.with_suffix(".txt").is_file()
        ]
        if missing_captions:
            sample = ", ".join(missing_captions[:5])
            raise ValueError(
                f"대응 TXT가 없는 이미지가 {len(missing_captions)}개 있어: {sample}"
            )

        target_dir = workspace / "dataset" / "target"
        control_out = workspace / "dataset" / "control"
        cache_dir = workspace / "cache"
        scripts_dir = workspace / "scripts"
        target_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        scripts_dir.mkdir(parents=True, exist_ok=True)

        # Rebuild only generated target files so stale captions/images do not
        # leak into a later training run.
        self._clear_generated_files(target_dir)
        for image in images:
            shutil.copy2(image, target_dir / image.name)
            caption = image.with_suffix(".txt").read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            normalized = self._with_trigger(caption, trigger)
            (target_dir / image.with_suffix(".txt").name).write_text(
                normalized + "\n",
                encoding="utf-8",
            )

        warnings: list[str] = []
        paired_count = 0
        missing_controls: list[str] = []

        if options.control_dir is None:
            warnings.append(
                "Qwen-Image-Edit-2511 학습에는 target과 짝이 되는 control/source "
                "이미지가 필요해. Control 폴더를 지정해야 실제 Train이 활성화돼."
            )
        else:
            control_dir = options.control_dir.resolve()
            if not control_dir.is_dir():
                warnings.append("Control 폴더가 존재하지 않아.")
            else:
                control_out.mkdir(parents=True, exist_ok=True)
                self._clear_generated_files(control_out)
                control_by_stem = {
                    path.stem.lower(): path
                    for path in control_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                }
                for image in images:
                    control = control_by_stem.get(image.stem.lower())
                    if control is None:
                        missing_controls.append(image.name)
                        continue
                    # Musubi matches control images by filename stem, so keep
                    # the target stem even when the source extension differs.
                    dst = control_out / f"{image.stem}{control.suffix.lower()}"
                    shutil.copy2(control, dst)
                    paired_count += 1

                if missing_controls:
                    warnings.append(
                        f"Control 짝이 없는 이미지가 {len(missing_controls)}개 있어. "
                        "파일명 stem을 target과 맞춰줘."
                    )

        config_path = workspace / "dataset.toml"
        config_path.write_text(
            self._dataset_toml(
                target_dir=target_dir,
                control_dir=control_out,
                cache_dir=cache_dir,
                resolution=options.resolution,
            ),
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        missing_training_paths = self._missing_training_paths(options)
        if missing_training_paths:
            warnings.append(
                "실행 파일/모델 경로 미지정: " + ", ".join(missing_training_paths)
            )

        pair_ready = (
            options.control_dir is not None
            and paired_count == len(images)
            and not missing_controls
        )
        ready = pair_ready and not missing_training_paths

        run_script: Path | None = None
        preview = ""
        if ready:
            scripts = self._write_scripts(options, config_path, scripts_dir)
            run_script = scripts["run_all"]
            preview = scripts["preview"]
        else:
            preview = (
                "Training workspace 준비 완료.\n"
                f"Target: {len(images)}장\n"
                f"Control pair: {paired_count}/{len(images)}\n"
                "Qwen-Image-Edit-2511은 source/control + target pair가 모두 있어야 "
                "실제 LoRA 학습 명령을 만들 수 있어."
            )

        manifest = {
            "model_version": "edit-2511",
            "target_count": len(images),
            "control_pair_count": paired_count,
            "trigger_token": trigger,
            "resolution": options.resolution,
            "rank": options.rank,
            "epochs": options.epochs,
            "learning_rate": options.learning_rate,
            "blocks_to_swap": options.blocks_to_swap,
            "ready_to_train": ready,
            "missing_controls": missing_controls,
            "warnings": warnings,
        }
        (workspace / "training_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return TrainingPrepResult(
            item_count=len(images),
            ready_to_train=ready,
            workspace_dir=workspace,
            dataset_config=config_path,
            run_script=run_script,
            command_preview=preview,
            warnings=warnings,
        )

    @staticmethod
    def _with_trigger(caption: str, trigger: str) -> str:
        parts = [part.strip() for part in caption.split(",") if part.strip()]
        if parts and parts[0].lower() == trigger.lower():
            return ", ".join(parts)
        if not parts:
            return f"{trigger}, person"
        return ", ".join([trigger, *parts])

    @staticmethod
    def _clear_generated_files(folder: Path) -> None:
        for path in folder.iterdir():
            if path.is_file() and (
                path.suffix.lower() in IMAGE_EXTENSIONS or path.suffix.lower() == ".txt"
            ):
                path.unlink()

    @staticmethod
    def _dataset_toml(
        target_dir: Path,
        control_dir: Path,
        cache_dir: Path,
        resolution: int,
    ) -> str:
        def q(path: Path) -> str:
            return str(path).replace("\\", "/").replace('"', '\\"')

        return (
            "[general]\n"
            f"resolution = [{resolution}, {resolution}]\n"
            'caption_extension = ".txt"\n'
            "batch_size = 1\n"
            "enable_bucket = true\n"
            "bucket_no_upscale = false\n\n"
            "[[datasets]]\n"
            f'image_directory = "{q(target_dir)}"\n'
            f'control_directory = "{q(control_dir)}"\n'
            f'cache_directory = "{q(cache_dir)}"\n'
            "control_resolution = [1024, 1024]\n"
            "no_resize_control = false\n"
            "num_repeats = 1\n"
        )

    @staticmethod
    def _missing_training_paths(options: TrainingPrepOptions) -> list[str]:
        values = (
            ("Musubi", options.musubi_dir, True),
            ("Python", options.python_exe, False),
            ("DiT", options.dit_path, False),
            ("VAE", options.vae_path, False),
            ("Text Encoder", options.text_encoder_path, False),
        )
        missing: list[str] = []
        for name, path, is_dir in values:
            if path is None:
                missing.append(name)
                continue
            resolved = path.resolve()
            if is_dir and not resolved.is_dir():
                missing.append(name)
            elif not is_dir and not resolved.is_file():
                missing.append(name)
        return missing

    def _write_scripts(
        self,
        options: TrainingPrepOptions,
        config_path: Path,
        scripts_dir: Path,
    ) -> dict[str, Path | str]:
        assert options.musubi_dir is not None
        assert options.python_exe is not None
        assert options.dit_path is not None
        assert options.vae_path is not None
        assert options.text_encoder_path is not None

        musubi = options.musubi_dir.resolve()
        python = options.python_exe.resolve()
        dit = options.dit_path.resolve()
        vae = options.vae_path.resolve()
        text_encoder = options.text_encoder_path.resolve()
        output_dir = options.output_dir.resolve()

        cache_latents = scripts_dir / "01_cache_latents.bat"
        cache_text = scripts_dir / "02_cache_text.bat"
        train = scripts_dir / "03_train.bat"
        run_all = scripts_dir / "run_all.bat"

        common_header = (
            "@echo off\n"
            "setlocal\n"
            f'set "PYTHON={python}"\n'
            f'set "MUSUBI={musubi}"\n'
            f'set "CONFIG={config_path.resolve()}"\n'
            f'set "DIT={dit}"\n'
            f'set "VAE={vae}"\n'
            f'set "TEXT_ENCODER={text_encoder}"\n'
            f'set "OUTPUT_DIR={output_dir}"\n'
            "\n"
        )

        latent_cmd = (
            '"%PYTHON%" "%MUSUBI%\\src\\musubi_tuner\\qwen_image_cache_latents.py" '
            '--dataset_config "%CONFIG%" --vae "%VAE%" --model_version edit-2511'
        )
        text_cmd = (
            '"%PYTHON%" "%MUSUBI%\\src\\musubi_tuner\\qwen_image_cache_text_encoder_outputs.py" '
            '--dataset_config "%CONFIG%" --text_encoder "%TEXT_ENCODER%" '
            '--batch_size 1 --model_version edit-2511 --fp8_vl'
        )

        train_args = [
            '"%PYTHON%"', "-m", "accelerate.commands.launch",
            "--num_cpu_threads_per_process", "1",
            "--mixed_precision", "bf16",
            '"%MUSUBI%\\src\\musubi_tuner\\qwen_image_train_network.py"',
            "--dit", '"%DIT%"',
            "--vae", '"%VAE%"',
            "--text_encoder", '"%TEXT_ENCODER%"',
            "--dataset_config", '"%CONFIG%"',
            "--model_version", "edit-2511",
            "--sdpa",
            "--mixed_precision", "bf16",
            "--timestep_sampling", "shift",
            "--weighting_scheme", "none",
            "--discrete_flow_shift", "2.2",
            "--optimizer_type", "adamw8bit",
            "--learning_rate", options.learning_rate,
            "--gradient_checkpointing",
            "--max_data_loader_n_workers", "2",
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_qwen_image",
            "--network_dim", str(options.rank),
            "--network_alpha", str(options.rank),
            "--max_train_epochs", str(options.epochs),
            "--save_every_n_epochs", "1",
            "--seed", "42",
            "--output_dir", '"%OUTPUT_DIR%"',
            "--output_name", options.output_name,
            "--fp8_base",
            "--fp8_scaled",
            "--fp8_vl",
        ]
        if options.blocks_to_swap > 0:
            train_args += ["--blocks_to_swap", str(options.blocks_to_swap)]
        train_cmd = " ".join(train_args)

        cache_latents.write_text(
            common_header + latent_cmd + "\nif errorlevel 1 exit /b %errorlevel%\n",
            encoding="utf-8",
        )
        cache_text.write_text(
            common_header + text_cmd + "\nif errorlevel 1 exit /b %errorlevel%\n",
            encoding="utf-8",
        )
        train.write_text(
            common_header + train_cmd + "\nif errorlevel 1 exit /b %errorlevel%\n",
            encoding="utf-8",
        )
        run_all.write_text(
            "@echo off\n"
            "setlocal\n"
            f'call "{cache_latents}"\n'
            "if errorlevel 1 goto :fail\n"
            f'call "{cache_text}"\n'
            "if errorlevel 1 goto :fail\n"
            f'call "{train}"\n'
            "if errorlevel 1 goto :fail\n"
            "echo.\n"
            "echo Training completed.\n"
            "pause\n"
            "exit /b 0\n"
            ":fail\n"
            "echo.\n"
            "echo Training failed. Check the message above.\n"
            "pause\n"
            "exit /b 1\n",
            encoding="utf-8",
        )

        preview = (
            "[1] Latent cache\n" + latent_cmd + "\n\n"
            "[2] Text encoder cache\n" + text_cmd + "\n\n"
            "[3] LoRA train\n" + train_cmd
        )
        return {
            "cache_latents": cache_latents,
            "cache_text": cache_text,
            "train": train,
            "run_all": run_all,
            "preview": preview,
        }
