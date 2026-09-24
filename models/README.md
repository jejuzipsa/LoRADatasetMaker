# models

런타임 AI 모델은 Git 저장소에 커밋하지 않는다.
프로그램이 필요한 모델의 존재와 SHA256을 확인하고, 없거나 손상됐으면 app-local `models/` 폴더에 자동 다운로드한다.

현재 사용 모델:

- `face_detection_yunet_2023mar.onnx`
  - 역할: 얼굴 위치 / 5-point landmark 검출
  - 출처: OpenCV Zoo / YuNet
  - 라이선스: Apache License 2.0
- `face_recognition_sface_2021dec.onnx`
  - 역할: 동일인물 판정을 위한 얼굴 feature/embedding
  - 출처: OpenCV Zoo / SFace
  - 라이선스: Apache License 2.0

모델 파일은 프로그램 실행 폴더의 `models/`에 저장하며, 정상 다운로드 후에는 오프라인으로 재사용한다.
