# Pipeline

~~~text
[원본 사진 폴더 / Drag & Drop]
        |
        v
[파일 인덱싱]
        |
        v
[YuNet 얼굴 검출]
        |--------------------> [검출 실패: 결과에 보존]
        v
[Head Crop 생성]
        |
        v
[빠른 품질 검사]
 blur / face pixel size / exposure / crop edge / confidence
        |
        v
[기준 인물 선택]
        |
        v
[SFace Identity Embedding]
        |
        v
[동일인물 유사도 계산]
        |
        +---- low -----------> [REJECTED 추천]
        +---- uncertain -----> [REVIEW 추천]
        +---- high ----------> [계속]
        |
        v
[품질 판정 재적용]
        |
        v
[애매한 후보 선별]
 REVIEW / 낮은 품질점수 / crop edge / 다중 얼굴
        |
        v
[선택적 로컬 Vision 2차 검수]
 original + detected face box
 head crop
        |
        +---- bad -----------> [REJECTED 추천]
        +---- maybe ---------> [REVIEW 추천]
        +---- good ----------> [기존 상태 유지]
        |
        v
[전체 결과 프리뷰]
 accepted / review / rejected 모두 표시
        |
        v
[사용자 최종 검수]
 상태 / caption 수정
 사용자 override가 자동 추천보다 우선
        |
        v
[EXPORT]
  |- accepted/ : 1024 crop + caption TXT
  |- review/   : 원본 보존
  |- rejected/ : 원본 보존
  |- logs/
       |- decisions.json
       |- quality_scores.json
       |- vision_reviews.json
       |- dataset_summary.json
~~~

## 다음 단계

~~~text
[중복/유사 프레임 그룹화]
        |
        v
[정교한 Head Pose]
        |
        v
[다양성 기반 20~40장 자동 선별]
        |
        v
[Dataset Maker]
   |
   +--> Qwen Image LoRA Trainer --> A_QWEN.safetensors
   |
   +--> WAN LoRA Trainer        --> A_WAN.safetensors
~~~

같은 정리된 dataset을 재사용하되, 서로 다른 베이스 모델 계열의 LoRA는 별도로 학습한다.
