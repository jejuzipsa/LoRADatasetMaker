from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import shutil


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(slots=True)
class TrainingPrepOptions:
    dataset_dir: Path
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
    """Prepare a no-control identity LoRA workspace with Musubi Tuner.

    This mode intentionally uses standard Qwen-Image training
    (model_version=original). Qwen-Image-Edit-2511 direct training is not used
    here because Musubi's Edit-2511 training path expects control/source images.

    The accepted dataset is never modified. Images/captions are copied into a
    workspace and the trigger token is injected only into those copies.
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
            path
            for path in dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError("accepted 폴더에서 학습 이미지를 찾지 못했어.")

        missing_captions = [
            image.name
            for image in images
            if not image.with_suffix(".txt").is_file()
        ]
        if missing_captions:
            sample = ", ".join(missing_captions[:5])
            raise ValueError(
                f"대응 TXT가 없는 이미지가 {len(missing_captions)}개 있어: {sample}"
            )

        target_dir = workspace / "dataset" / "target"
        cache_dir = workspace / "cache"
        scripts_dir = workspace / "scripts"
        target_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        scripts_dir.mkdir(parents=True, exist_ok=True)

        self._clear_generated_files(target_dir)
        for image in images:
            shutil.copy2(image, target_dir / image.name)
            caption = image.with_suffix(".txt").read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()
            normalized = self._with_trigger(caption, trigger)
            (target_dir / image.with_suffix(".txt").name).write_text(
                normalized + "\n",
                encoding="utf-8",
            )

        config_path = workspace / "dataset.toml"
        config_path.write_text(
            self._dataset_toml(
                target_dir=target_dir,
                cache_dir=cache_dir,
                resolution=options.resolution,
            ),
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        missing_training_paths = self._missing_training_paths(options)
        warnings: list[str] = []
        if missing_training_paths:
            warnings.append(
                "실행 파일/모델 경로 미지정: "
                + ", ".join(missing_training_paths)
            )

        if options.dit_path is not None:
            dit_name = options.dit_path.name.lower()
            if "edit" in dit_name:
                raise ValueError(
                    "Identity LoRA 모드는 Control 없는 표준 Qwen-Image 학습이야. "
                    "qwen_image_edit_2511 계열 DiT가 아니라 "
                    "qwen_image_bf16.safetensors 같은 Qwen-Image base DiT를 지정해줘."
                )

        ready = not missing_training_paths

        run_script: Path | None = None
        preview = (
            "Identity LoRA workspace 준비 완료.\n"
            f"Dataset: {len(images)}장\n"
            "Mode: Qwen-Image original / Control 없음\n"
            "주의: Edit-2511 직접 학습은 Musubi 규격상 control/source가 필요해서 "
            "이 모드에서는 사용하지 않아."
        )

        if ready:
            scripts = self._write_scripts(options, config_path, scripts_dir)
            run_script = scripts["run_all"]
            preview = scripts["preview"]

        manifest = {
            "training_mode": "identity_no_control",
            "model_version": "original",
            "target_count": len(images),
            "trigger_token": trigger,
            "resolution": options.resolution,
            "rank": options.rank,
            "epochs": options.epochs,
            "learning_rate": options.learning_rate,
            "blocks_to_swap": options.blocks_to_swap,
            "ready_to_train": ready,
            "note": (
                "No-control identity LoRA is trained on standard Qwen-Image. "
                "Direct Qwen-Image-Edit-2511 training is not used in this mode "
                "because Edit-2511 expects control/source images."
            ),
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
                path.suffix.lower() in IMAGE_EXTENSIONS
                or path.suffix.lower() == ".txt"
            ):
                path.unlink()

    @staticmethod
    def _dataset_toml(
        target_dir: Path,
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
            f'cache_directory = "{q(cache_dir)}"\n'
            "num_repeats = 1\n"
        )

    @staticmethod
    def _missing_training_paths(options: TrainingPrepOptions) -> list[str]:
        values = (
            ("Musubi", options.musubi_dir, True),
            ("Python", options.python_exe, False),
            ("Qwen-Image DiT", options.dit_path, False),
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
        run_all_bat = scripts_dir / "run_all.bat"
        run_all_py = scripts_dir / "run_all.py"

        latent_args = [
            str(python),
            str(musubi / "src" / "musubi_tuner" / "qwen_image_cache_latents.py"),
            "--dataset_config",
            str(config_path.resolve()),
            "--vae",
            str(vae),
            "--model_version",
            "original",
        ]
        text_args = [
            str(python),
            str(
                musubi
                / "src"
                / "musubi_tuner"
                / "qwen_image_cache_text_encoder_outputs.py"
            ),
            "--dataset_config",
            str(config_path.resolve()),
            "--text_encoder",
            str(text_encoder),
            "--batch_size",
            "1",
            "--model_version",
            "original",
            "--fp8_vl",
        ]
        train_args = [
            str(python),
            "-m",
            "accelerate.commands.launch",
            "--num_cpu_threads_per_process",
            "1",
            "--mixed_precision",
            "bf16",
            str(musubi / "src" / "musubi_tuner" / "qwen_image_train_network.py"),
            "--dit",
            str(dit),
            "--vae",
            str(vae),
            "--text_encoder",
            str(text_encoder),
            "--dataset_config",
            str(config_path.resolve()),
            "--model_version",
            "original",
            "--sdpa",
            "--mixed_precision",
            "bf16",
            "--timestep_sampling",
            "shift",
            "--weighting_scheme",
            "none",
            "--discrete_flow_shift",
            "2.2",
            "--optimizer_type",
            "adamw8bit",
            "--learning_rate",
            options.learning_rate,
            "--gradient_checkpointing",
            "--max_data_loader_n_workers",
            "2",
            "--persistent_data_loader_workers",
            "--network_module",
            "networks.lora_qwen_image",
            "--network_dim",
            str(options.rank),
            "--network_alpha",
            str(options.rank),
            "--max_train_epochs",
            str(options.epochs),
            "--save_every_n_epochs",
            "1",
            "--seed",
            "42",
            "--output_dir",
            str(output_dir),
            "--output_name",
            options.output_name,
            "--fp8_base",
            "--fp8_scaled",
            "--fp8_vl",
        ]
        if options.blocks_to_swap > 0:
            train_args += ["--blocks_to_swap", str(options.blocks_to_swap)]

        def quote_cmd(args: list[str]) -> str:
            return " ".join(
                '"' + value.replace('"', '\\"') + '"'
                if (" " in value or "\t" in value or any(ord(ch) > 127 for ch in value))
                else value
                for value in args
            )

        latent_cmd = quote_cmd(latent_args)
        text_cmd = quote_cmd(text_args)
        train_cmd = quote_cmd(train_args)

        # Keep individual BAT files for manual debugging, but the app does not
        # execute them. Windows cmd.exe can misread UTF-8 batch files when the
        # workspace path contains Korean/non-ASCII characters.
        common_header = (
            "@echo off\n"
            "setlocal\n"
            "chcp 65001 >nul\n"
            "set PYTHONUTF8=1\n"
            "set PYTHONUNBUFFERED=1\n"
        )
        cache_latents.write_text(
            common_header
            + latent_cmd
            + "\nif errorlevel 1 exit /b %errorlevel%\n",
            encoding="utf-8-sig",
        )
        cache_text.write_text(
            common_header
            + text_cmd
            + "\nif errorlevel 1 exit /b %errorlevel%\n",
            encoding="utf-8-sig",
        )
        train.write_text(
            common_header
            + train_cmd
            + "\nif errorlevel 1 exit /b %errorlevel%\n",
            encoding="utf-8-sig",
        )
        run_all_bat.write_text(
            common_header
            + "echo This BAT is for manual debugging only.\n"
            + f'call "{cache_latents}"\n'
            + "if errorlevel 1 exit /b %errorlevel%\n"
            + f'call "{cache_text}"\n'
            + "if errorlevel 1 exit /b %errorlevel%\n"
            + f'call "{train}"\n'
            + "exit /b %errorlevel%\n",
            encoding="utf-8-sig",
        )

        stages = [
            {
                "number": 1,
                "title": "Latent cache",
                "command": latent_args,
            },
            {
                "number": 2,
                "title": "Text encoder cache",
                "command": text_args,
            },
            {
                "number": 3,
                "title": "LoRA training",
                "command": train_args,
            },
        ]
        runner_source = (
            "from __future__ import annotations\n"
            "import os\n"
            "import subprocess\n"
            "import sys\n\n"
            f"WORKDIR = {json.dumps(str(musubi), ensure_ascii=False)}\n"
            f"STAGES = {json.dumps(stages, ensure_ascii=False, indent=2)}\n\n"
            "env = os.environ.copy()\n"
            "env['PYTHONUTF8'] = '1'\n"
            "env['PYTHONUNBUFFERED'] = '1'\n"
            "env['PYTHONIOENCODING'] = 'utf-8'\n\n"
            "for stage in STAGES:\n"
            "    print(\n"
            "        f\"__LDM_STAGE__ {stage['number']} {stage['title']}\",\n"
            "        flush=True,\n"
            "    )\n"
            "    process = subprocess.Popen(\n"
            "        stage['command'],\n"
            "        cwd=WORKDIR,\n"
            "        env=env,\n"
            "        stdout=subprocess.PIPE,\n"
            "        stderr=subprocess.STDOUT,\n"
            "        bufsize=0,\n"
            "    )\n"
            "    assert process.stdout is not None\n"
            "    while True:\n"
            "        reader = getattr(process.stdout, 'read1', process.stdout.read)\n"
            "        chunk = reader(4096)\n"
            "        if not chunk:\n"
            "            break\n"
            "        sys.stdout.write(chunk.decode('utf-8', errors='replace'))\n"
            "        sys.stdout.flush()\n"
            "    code = process.wait()\n"
            "    if code != 0:\n"
            "        print(\n"
            "            f\"__LDM_FAILED__ stage={stage['number']} code={code}\",\n"
            "            flush=True,\n"
            "        )\n"
            "        raise SystemExit(code)\n\n"
            "print('__LDM_DONE__', flush=True)\n"
        )
        run_all_py.write_text(
            runner_source,
            encoding="utf-8",
        )

        preview = (
            "[Identity LoRA / Qwen-Image original / Control 없음]\n"
            "[Unicode-safe Python runner]\n\n"
            "[1] Latent cache\n"
            + latent_cmd
            + "\n\n[2] Text encoder cache\n"
            + text_cmd
            + "\n\n[3] LoRA train\n"
            + train_cmd
        )
        return {
            "cache_latents": cache_latents,
            "cache_text": cache_text,
            "train": train,
            "run_all": run_all_py,
            "run_all_bat": run_all_bat,
            "preview": preview,
        }

