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
- Qwen/WAN Trainer 연동

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
