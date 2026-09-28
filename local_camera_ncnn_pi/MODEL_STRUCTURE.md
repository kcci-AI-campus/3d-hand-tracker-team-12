# 공식 모델 구조와 코드 대응

## 원본

[공식 안내의 Latest](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
배포본을 2026-09-23에 다운로드했습니다.

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
SHA256: fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
```

이 시점의 Latest는 기존 공식 version 1 번들과 같은 해시입니다. 새 릴리스가
발표되었다는 뜻이 아니라 현재 공식 Latest 파일을 다시 받아 확인한 것입니다.
`.task` ZIP 번들에는 두 TFLite FlatBuffer가 있습니다.

## 손바닥 검출 — hand_detector

- 학습 파라미터 **1,136,248개**.
- 원본 입력 `input_1`: FLOAT32 `[1,192,192,3]`, RGB NHWC, 0..1.
- ncnn 입력 `input`: FP32 `[3,192,192]`, CHW. 배치 1은 Mat에서 생략합니다.

| TFLite 출력 | Shape | ncnn blob | 의미 |
|---|---|---|---|
| `Identity` | `[1,2016,18]` | `regressors` | 중심 XY, 너비·높이, 7개 XY 키포인트 |
| `Identity_1` | `[1,2016,1]` | `scores` | sigmoid 적용 전 palm logit |

연산자: DEQUANTIZE 133, CONV_2D 35, DEPTHWISE_CONV_2D 28, PRELU 31,
ADD 30, MAX_POOL_2D 4, PAD 3, RESIZE_BILINEAR 2, RESHAPE 4, CONCATENATION 2.

Convolution/Depthwise + PReLU 잔차 블록과 상향 해상도 결합 구조입니다.
stride 8의 24×24×2 = 1,152개와 stride 16의 12×12×6 = 864개 앵커를 합칩니다.
sigmoid → 0.5 임계값 → IoU 0.3 가중 NMS로 박스·키포인트를 합칩니다.

## 손 관절 — hand_landmarks_detector

- 학습 파라미터 **2,715,680개**.
- 원본 입력 `input_1`: FLOAT32 `[1,224,224,3]`; ncnn `input`: `[3,224,224]`.
- 깊이별 분리 합성곱·ReLU6·잔차 연결 → 공간 평균 → 네 개의 독립 FC head.

| TFLite 출력 | Shape | ncnn blob | 코드 처리 |
|---|---|---|---|
| `Identity` | `[1,63]` | `landmarks` | crop의 21×XYZ → 원본 영상 좌표 |
| `Identity_1` | `[1,1]` | `presence` | sigmoid 손 존재 확률; 0.5 미만 제거 |
| `Identity_2` | `[1,1]` | `handedness` | sigmoid 좌우 손 확률; 존재 확률과 별도 |
| `Identity_3` | `[1,63]` | `world_landmarks` | 손 중심 기준 21×XYZ 미터 값, 회전만 복원 |

연산자: DEQUANTIZE 102, CONV_2D 31, DEPTHWISE_CONV_2D 16,
ADD 9, MEAN 1, FULLY_CONNECTED 4, LOGISTIC 2.

손목 키포인트 0과 중지 밑마디 2로 회전을 구하고 palm 사각형의 Y를 -0.5만큼
이동한 다음 긴 변을 2.6배 확대합니다. 224×224 워핑 영상을 관절 모델에 넣습니다.
이미지 Z는 공식 `normalize_z=0.4`에 따라 `raw_z * crop_side / (224 * 0.4)`로
복원합니다. 월드 XYZ에는 XY 회전만 적용하며 픽셀 크기·이미지 위치를 더하지 않습니다.

출력 의미와 계수의 공식 근거:

- [Hand detector graph](https://github.com/google-ai-edge/mediapipe/blob/f769019723ef810112419a75d5e84fe8d11ae5de/mediapipe/tasks/cc/vision/hand_detector/hand_detector_graph.cc)
- [Hand landmarks graph](https://github.com/google-ai-edge/mediapipe/blob/f769019723ef810112419a75d5e84fe8d11ae5de/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarks_detector_graph.cc)

## 변환과 검증

FlatBuffer의 **실제 가중치·연산 옵션을 읽어** 같은 PyTorch 그래프를 만들었습니다.
NHWC→NCHW 배치, depthwise weight 축, SAME 비대칭 패딩, bilinear 좌표 규칙,
reshape의 NHWC 순서를 보존합니다. 지원하지 않는 연산·옵션은 실패하도록 했습니다.
PyTorch 결과를 LiteRT와 대조한 뒤 TorchScript → pnnx로 표준 ncnn 연산자를
생성했습니다. 커스텀 레이어나 재학습은 없고 네 출력 head를 모두 보존합니다.
원본 FP16 상수는 FP32로 확장하여 저장했으며 실행도 FP32로 요청합니다.

최신 공식 pnnx `20260704`로 변환하고 ncnn `20260526`으로 검증했습니다.
pnnx가 손바닥 출력의 네 Reshape에 남긴 singleton batch(`2=1`)는 ncnn에서
3차원 채널로 해석되어 후보 연결 축이 달라집니다. 변환 스크립트는 정확히 네 개의
알려진 형태만 확인한 뒤 `[864 또는 1152, 1 또는 18]`의 2차원으로 정규화합니다.
최종 출력은 원본과 같은 2016개 후보입니다. 보정 내역은 `conversion_report.json`의
`ncnn_reshape_fixes`에 기록되며, 예상과 다른 그래프는 변환을 중단합니다.

LiteRT 2.2.0과 **ncnn 20260526 C++**에 동일한 검정·흰색·난수·실제 손·90도 회전·
좌우 반전 텐서를 입력했습니다. 모든 출력 shape가 같고 `atol=0.002, rtol=0.002`
비교를 통과했습니다.

| 출력 | 6종 입력에서 최대 절대 오차 |
|---|---:|
| palm regressors | 0.0004731 |
| palm logits | 0.0001030 |
| hand image landmarks | 0.0003205 |
| hand presence | 0.0000003 |
| handedness | 0.0000011 |
| world landmarks | 0.0000004m |

이는 모델 변환의 수치 일치 검사이며 인식 정확도 평가가 아닙니다.
C++ 전체 전후처리도 이미지 9종의 예상 손 개수와 일치했습니다.
480px 이미지의 회전/반전 전후 관절 평균 차이는 90도 1.32px, 180도 0.82px,
반전 3.67px입니다. `conversion/image_checks.json`에 상세 결과가 있습니다.
Windows 호스트에서 검증했으며 ARM64의 수치 차이·카메라·속도는 실기 확인이 필요합니다.
