# LoRADatasetMaker

Windows용 로컬 LoRA 학습 데이터셋 전처리/검수 도구.

## 현재 상태
- 0001: 프로젝트 골격
- 0002: 프리뷰/검수 UI
- 0003: 기본 얼굴 검출 + Head Crop + Export 기초
- 0004: 기준 인물 선택 + 기본 동일인물 재분류
- 0005: GitHub Actions CI / Windows 빌드 artifact
- 0006: 백그라운드 자동 분석 + 진행률 + 드래그앤드롭
- 0007: YuNet 얼굴 검출 + 모델 자동 다운로드 + 검출 박스 표시
- 0008: SFace 얼굴 임베딩 + 동일인물 판정
- 0009: Git LFS 모델 다운로드 핫픽스
- 0010: 품질 점수 + 선택적 로컬 Vision 2차 검수 + 확장 Export
- 0011: 상태별 색상 테두리 + 선택 항목 강조
- 0012: 자동 품질 판정 완화 + 불필요한 REVIEW 감소
- 0013: Qwen Image Edit 2511 Training 준비/실행 섹션
- 0014: Identity LoRA 기본 모드로 전환 / Control 요구 제거
- 0015: ComfyUI Qwen 모델 자동 검색
- 0016: Musubi Tuner + 전용 Python 자동 설치
- 0017: Training 실시간 로그 / 진행률 / 중지 기능

## 1차 목표
사진을 대량 투입하면 동일 인물 후보를 분류하고, Head/Portrait 학습용 크롭·품질 검사·Vision 검수·중복 판정·얼굴 방향 분석·캡션 생성을 수행한 뒤, 사용자가 모든 채택/제외/보류 결과를 최종 검수하여 LoRA 학습용 데이터셋으로 Export한다.

## 현재 지원
- 폴더 이미지 인덱싱
- 폴더 / 다중 이미지 드래그앤드롭
- PySide6 다크모드 검수 UI
- 백그라운드 자동 분석 및 진행률 표시
- YuNet 기반 얼굴 검출
- YuNet/SFace 모델 자동 다운로드 및 SHA256 검증
- 원본 프리뷰 얼굴 검출 박스 / confidence 표시
- Head Crop 생성
- 기준 인물 지정
- SFace 얼굴 임베딩 기반 동일인물 재분류
- 여러 얼굴이 있는 사진에서는 기준 인물과 가장 비슷한 얼굴 선택
- blur / 얼굴 픽셀 크기 / 노출 / crop edge / detector confidence 기반 품질 점수
- 품질 신호를 ACCEPTED / REVIEW / REJECTED 추천에 반영
- 로컬 Ollama Vision 모델을 이용한 선택적 2차 검수
- Vision은 애매한 후보만 대상으로 원본+Head Crop을 함께 검토
- ACCEPTED / REVIEW / REJECTED 수동 조정 및 사용자 override 보존
- ACCEPTED 파랑 / REVIEW 노랑 / REJECTED 빨강 상태 테두리
- caption 수동 수정
- accepted 결과 1024 PNG + TXT Export
- review / rejected 원본 보존 Export
- decisions / quality_scores / vision_reviews / dataset_summary 로그
- Musubi Tuner Identity LoRA Training 탭 (Control 불필요)
- Trigger 자동 삽입 학습 workspace 생성
- Qwen-Image base용 dataset.toml / cache / train BAT 생성
- ComfyUI models 폴더에서 Qwen-Image DiT / VAE / Text Encoder 자동 검색
- Musubi Tuner v0.3.5 + uv managed Python 3.11 + cu128 전용 환경 자동 설치
- Latent cache / Text Encoder cache / LoRA Train 3단계 실시간 상태와 Progress Bar
- Training stdout/stderr 실시간 로그, Step/Loss 표시, 중지 버튼
- GitHub Actions 자동 검사
- Windows EXE 자동 빌드 artifact

## Vision 2차 검수
기본 endpoint는 다음과 같다.

~~~text
http://127.0.0.1:11434/api/chat
~~~

Ollama에 설치된 Vision 모델 이름을 UI의 Model 칸에 입력하면 Vision 검수 버튼이 활성화된다.

Vision 검수는 모든 이미지를 무조건 다시 처리하지 않는다. REVIEW 상태, 품질 점수가 낮은 항목, crop edge 접촉, 다중 얼굴 등 애매한 후보를 우선 검수한다.

## 아직 미구현
- 중복/유사 프레임 자동 그룹화
- 더 정교한 head pose(yaw/pitch/roll)
- 다양성 기반 20~40장 자동 선별
- Vision provider 추가(OpenAI-compatible 등)
- WAN Trainer 연동
- Qwen Edit 2511 직접 Edit-LoRA 모드(control/source pair) 별도 추가

## 로컬 실행

~~~bash
pip install -e .
loradatasetmaker
~~~

## GitHub Actions
main 브랜치에 push하거나 Actions 탭에서 수동 실행하면:
1. Python 3.11 환경에서 패키지 설치
2. Python 소스 compile 검사
3. 핵심 모듈 import smoke test
4. Windows EXE 빌드
5. LoRADatasetMaker-windows artifact 업로드

자세한 설계는 docs/SPEC.md, docs/PIPELINE.md 참고.


## Identity LoRA Training

Training 탭의 기본 모드는 accepted 이미지 + TXT만 사용하는 Control 없는 Identity LoRA 모드다.

Musubi Tuner에서 Qwen-Image-Edit-2511의 직접 학습은 control/source 이미지를 사용하는 구조이므로, 0014의 Identity 모드는 `model_version=original`인 표준 Qwen-Image base 학습으로 분리했다. 따라서 DiT에는 `qwen_image_bf16.safetensors` 같은 Qwen-Image base 가중치를 지정해야 하며, Edit-2511 DiT를 잘못 지정하면 준비 단계에서 중단한다.

학습 준비 시 원본 accepted 폴더를 수정하지 않고 workspace로 복사하며 TXT 앞에 Trigger token을 자동 삽입한다. workspace에는 dataset.toml, training_manifest.json 및 scripts/01_cache_latents.bat, 02_cache_text.bat, 03_train.bat, run_all.bat을 생성한다.

이 Identity LoRA를 Edit-2511에서 사용하는 부분은 직접 Edit-2511을 control 없이 학습한다는 의미가 아니며, 실제 ComfyUI 호환성/재현 결과는 생성 후 테스트 대상으로 둔다.


## ComfyUI 모델 자동 검색

Training 탭에서 ComfyUI 루트를 지정하고 `모델 자동 검색`을 누르면 `models/diffusion_models`, `models/unet`, `models/checkpoints`, `models/vae`, `models/text_encoders`, `models/clip` 순으로 필요한 Qwen-Image 학습 모델을 찾는다.

Identity LoRA 모드에서는 Qwen-Image-Edit-2511 DiT를 자동 선택하지 않으며, `qwen_image_bf16.safetensors`, `qwen_image_vae.safetensors`, `qwen_2.5_vl_7b.safetensors` 같은 표준 Qwen-Image 학습용 파일을 우선한다.


## Musubi Tuner 자동 설치

Training 탭의 `자동 설치` 버튼은 프로그램의 `tools` 폴더 아래에 Musubi Tuner stable release v0.3.5와 독립 Python 환경을 준비한다. Git 설치는 필요하지 않으며, uv를 사용해 Python 3.11과 `.venv`를 관리한다.

ComfyUI의 Python 환경은 수정하거나 재사용하지 않는다. Musubi의 `cu128` extra를 사용해 CUDA 12.8용 PyTorch/torchvision 및 필요한 의존성을 설치하며, 설치 완료 후 Musubi 경로와 전용 `python.exe`가 Training 탭에 자동 입력된다.
