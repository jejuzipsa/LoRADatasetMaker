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

## 1차 목표
사진을 대량 투입하면 자동으로 동일 인물 후보를 분류하고, Head/Portrait 학습용 크롭·품질 검사·중복 판정·얼굴 방향 분석·캡션 생성을 수행한 뒤, 사용자가 모든 채택/제외/보류 결과를 프리뷰에서 최종 검수하여 LoRA 학습용 데이터셋으로 Export한다.

## 현재 지원
- 폴더 이미지 인덱싱
- 폴더 / 다중 이미지 드래그앤드롭
- PySide6 다크모드 검수 UI
- 백그라운드 자동 분석 및 진행률 표시
- YuNet 기반 얼굴 검출
- 첫 분석 시 YuNet 모델 자동 다운로드 및 SHA256 검증
- 원본 프리뷰 얼굴 검출 박스 / confidence 표시
- Head Crop 생성
- 기준 인물 지정
- 기본 동일인물 유사도 재분류
- ACCEPTED / REVIEW / REJECTED 수동 조정
- caption 수동 수정
- accepted 결과 1024 PNG + TXT Export
- decisions.json 감사 로그
- GitHub Actions 자동 검사
- Windows EXE 자동 빌드 artifact

## 아직 미구현
- 전용 face embedding 모델 기반 고정밀 동일인물 판정
- 고급 품질 검사
- 중복/유사 프레임 제거
- 정교한 head pose
- 추가 AI 모델 자동 다운로드 체계 확장
- rejected/review 이미지 별도 export
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
5. `LoRADatasetMaker-windows` artifact 업로드

자세한 설계는 `docs/SPEC.md`, `docs/PIPELINE.md` 참고.
