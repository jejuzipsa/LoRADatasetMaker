# LoRADatasetMaker 기능 명세 v0.1

## 목적
많은 사진/영상 캡처를 입력받아 특정 인물의 LoRA 학습용 데이터셋을 반자동으로 제작한다.

## 설계 방향
완전 자동 삭제기가 아니라 **자동 분석 + 사람이 최종 확정하는 검수 도구**로 설계한다.

## 입력
- JPG / JPEG / PNG / WEBP
- 폴더 단위 대량 입력
- 기준 인물(reference identity) 이미지 1장 또는 입력 이미지 중 얼굴 선택

## 자동 처리
### 얼굴 검출
- 이미지 내 모든 얼굴 검출
- 얼굴 크기, 경계 잘림 여부 기록
- 검출 실패도 결과 목록에 남김

### 동일 인물 판정
- 기준 얼굴 embedding과 각 검출 얼굴 embedding 비교
- 자동 채택/보류/제외 추천
- 타인으로 판정돼도 파일은 삭제하지 않음

### 품질 검사
- blur
- 너무 작은 얼굴
- 낮은 유효 해상도
- 심한 가림/경계 잘림
- 기타 품질 지표

### 중복/유사 프레임 판정
- 영상 연속 캡처처럼 거의 동일한 이미지 묶음 탐지
- 대표 후보 추천
- 중복 후보 전체를 UI에서 확인 가능

### Head Pose / 얼굴 방향
내부 각도값(yaw/pitch/roll)을 학습용 텍스트 태그로 변환한다.
- front view
- slightly left
- left three-quarter view
- left profile
- slightly right
- right three-quarter view
- right profile
- looking up
- looking down
- head tilted left/right

### 자동 크롭
기본값은 Head Crop.
- FACE: 얼굴 중심
- HEAD: 머리 전체 + 목 + 어깨 일부
- PORTRAIT: 머리 + 상반신 일부
- MIXED: 후속 버전

기본 출력 해상도: 1024x1024
원본 비율/경계를 고려해 padding/crop을 수행한다.

### 자동 캡션
트리거 토큰 + 방향 중심의 짧은 캡션을 기본으로 한다.
예:
`personA, woman, left three-quarter view, smiling, short brown hair`

v0.1에서는 방향 태그를 우선 신뢰하고, 표정/헤어 등은 선택 기능으로 둔다.

## 상태 모델
모든 결과는 다음 셋 중 하나를 가진다.
- ACCEPTED: 학습 포함 추천
- REVIEW: 사용자 검토 추천
- REJECTED: 학습 제외 추천

자동 판정과 최종 판정을 별도 저장한다.
예:
- auto_status = REJECTED
- auto_reason = duplicate
- final_status = ACCEPTED
- user_override = true

## 프리뷰/검수 UI
### 상단 필터
- 전체
- 채택
- 보류
- 제외
- 중복 후보
- 저품질
- 타인 후보
- 얼굴 검출 실패

### 썸네일 카드
- 크롭 프리뷰
- 파일명
- 상태
- 얼굴 방향
- 자동 제외/보류 사유
- 동일인물 유사도
- 품질 점수

### 상세 패널
- 원본 이미지
- 크롭 이미지
- 얼굴 bounding box
- 자동 상태 / 최종 상태
- 자동 판정 사유
- similarity score
- quality score
- yaw / pitch / roll
- 방향 태그
- caption

### 사용자 조작
- 채택으로 변경
- 보류로 변경
- 제외로 변경
- 캡션 수정
- 방향 태그 수정
- 여러 항목 일괄 상태 변경
- 사유별 필터

## Export
### 학습 폴더
최종 ACCEPTED만 LoRA Trainer에 바로 넣을 수 있게 출력.
예:
```
dataset/
  0001.png
  0001.txt
  0002.png
  0002.txt
```

### 감사/검수 결과
```
review_output/
  accepted/
  rejected/
  review/
  logs/
    decisions.json
    dataset_summary.json
    captions.json
```

### decisions.json 최소 필드
- source_file
- crop_file
- detected_face_index
- auto_status
- auto_reason[]
- final_status
- user_override
- identity_similarity
- quality_metrics
- yaw/pitch/roll
- direction_caption
- final_caption

## 첫 실행 모델 관리
- 프로그램 폴더 하위 `models/` 사용
- 필요한 모델이 없을 때만 다운로드 안내
- 다운로드 진행률 표시
- 다운로드 후 hash/version 확인
- 이후 오프라인 실행 가능
- 모델 업데이트는 사용자 선택형
- 배포 전 각 모델의 라이선스 확인 필수

## v0.1 범위
포함:
- 폴더 입력
- 얼굴 검출
- 기준 인물 선택
- 동일인물 분류
- 기본 품질 검사
- 유사/중복 후보 탐지
- Head Pose
- Head Crop 1024
- 기본 캡션
- 전체 결과 프리뷰
- 수동 override
- Export + 로그

제외(후속):
- Qwen/WAN LoRA 학습 실행
- 자동 epoch 평가
- 영상 직접 디코딩/프레임 추출
- 고급 VLM 캡션
- 클라우드 기능
