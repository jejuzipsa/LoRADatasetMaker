# Pipeline

```text
[원본 사진 폴더]
        |
        v
[파일 인덱싱]
        |
        v
[얼굴 검출]
        |--------------------> [검출 실패: 결과에 보존]
        v
[기준 인물 선택 / Identity Embedding]
        |
        v
[동일인물 유사도 계산]
        |
        +---- low -----------> [REJECTED 추천]
        +---- uncertain -----> [REVIEW 추천]
        +---- high ----------> [계속]
        |
        v
[품질 검사]
 blur / size / crop edge / occlusion
        |
        v
[중복/유사 프레임 그룹화]
        |
        v
[Head Pose: yaw/pitch/roll]
        |
        v
[Head Crop / 1024x1024]
        |
        v
[자동 캡션 생성]
        |
        v
[전체 결과 프리뷰]
 accepted / review / rejected 모두 표시
        |
        v
[사용자 최종 검수]
 상태/캡션/방향 수정
        |
        v
[EXPORT]
  |- dataset/       : 최종 ACCEPTED 학습용
  `- review_output/ : 모든 판정/로그/제외본
```

## 후속 확장
```text
Dataset Maker
   |
   +--> Qwen Image LoRA Trainer --> A_QWEN.safetensors
   |
   `--> WAN LoRA Trainer        --> A_WAN.safetensors
```

같은 정리된 dataset을 재사용하되, 서로 다른 베이스 모델 계열의 LoRA는 별도로 학습한다.
