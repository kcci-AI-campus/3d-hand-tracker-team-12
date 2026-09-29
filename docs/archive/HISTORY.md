# 개발 이력 (삭제된 코드·문서 요약)

최종 구성에 쓰이지 않아 저장소에서 삭제한 코드와 문서를 한곳에 정리했습니다. 원본은 git 기록에 남아 있습니다(삭제 직전 커밋 `369d923`).

## 1. 카메라·통신

| 단계 | 내용 | 대체 |
|---|---|---|
| MediaPipe Python 프로토타입 (`camera/`) | Pi 3대가 MediaPipe Hand Landmarker(Python)로 320×240 영상에서 손 관절을 검출하고, 송신 Pi 2대가 JPEG + 관절 JSON을 TCP(`HND1` 헤더, 프레임마다 ACK)로 마스터에 보내 세 화면을 표시. 3D 복원은 없었음. 로컬 화면(`local_camera.py`), MediaPipe world landmarks 3D 보기도 포함 | C++ slave/master (UV2·H3D1, UDP) |
| Windows ncnn 카메라 앱 (`local_camera_ncnn/`) | `local_camera.py`를 C++ + ncnn + OpenCV로 옮긴 PC용 화면 앱 | — (시험용) |
| Pi ncnn 카메라 앱 (`local_camera_ncnn_pi/`) | 공식 `hand_landmarker.task`의 두 TFLite 모델을 ncnn으로 직접 변환해 Pi에서 C++로 실행하고 u,v를 UDP로 보냄 (UV42, 헤더 없음 336바이트) | [hand_tracker_slave](../../hand_tracker_slave/README.md): 같은 변환 모델 + UV2 헤더(카메라 번호·순번·촬영→송신 시간) |
| YOLO26n Pose 실험 (`notebooks/train_yolo26n_hand_pose.ipynb` 등 3개) | Ultralytics Hand Keypoints 데이터를 MediaPipe로 좌우 재라벨링·검수해 YOLO26n-pose를 파인튜닝하는 Colab 노트북 | 쓰지 않음: 최종 slave는 공식 MediaPipe 모델을 ncnn으로 변환해 사용 |

**TCP → UDP로 바꾼 이유:** 3D 모델은 영상이 아니라 관절 좌표만 필요하고, 늦게 온 프레임보다 최신 프레임이 중요합니다. 재전송·ACK 대기가 없는 UDP 356바이트 패킷으로 바꾸고, 시계 동기화 대신 slave가 촬영→송신 시간을 보내 마스터가 자기 시계에서 촬영 시각을 복원합니다.

## 2. 지연 측정 → 시뮬레이터 타이밍 프로필

`camera/latency_recorder.py`로 실제 Pi(C270 웹캠)→PC 경로의 촬영 간격·처리 대기·전송 시간을 기록했습니다(원본 CSV는 `latency_logs/`). 이 로그를 한 행씩 묶은 것이 데이터 생성기의 기본 타이밍 프로필 `profiles/pi_c270_timing.json`입니다.

- 초기 5초 제외 5,015행
- 촬영 간격 평균 58.54ms (약 17.1FPS)
- 송신 전 처리·대기·인코딩 평균 64.33ms
- 송신 준비 → 수신 평균 1.47ms
- 읽기 완료 → 수신 평균 65.80ms, P95 123.83ms

측정 도구는 MediaPipe Python 송신기를 재사용했으므로 함께 삭제했습니다. 프로필 사용법은 [GIGAHANDS.md](../../GIGAHANDS.md)에 있습니다.

## 3. 3D 모델

| 모델 | 구조 | 파라미터 | 결과 |
|---|---|---|---|
| 순환 보정 모델 | 카메라별 GRU 상태로 설치 오차 누적 보정 → 삼각측량·외삽 → 공간/시간 Transformer | 2,042,121 | 관측 없는 구간에도 보정값이 흘러가는 문제, 장시간 안정성 검증 불가로 폐기 |
| 윈도 기반 모델 | 단기 보정 + 기하 기준점 + causal Transformer (프레임 윈도 입력) | 1,944,909 | 비동기 카메라를 프레임 단위로 다루는 한계 |
| 이벤트 기반 모델 | 카메라 프레임 하나 = 이벤트, 임의 시각 query | 1,746,765 → 763,693 | 이벤트·슬롯·스트리밍 구조는 현재 모델로 이어짐 |
| HandLite v1 | 최신 두 프레임 외삽 + 카메라 보정 신경망 + 잔차 | 191,469 | HandLiteV3로 대체 ([HANDLITE_V1_ARCHITECTURE.md](HANDLITE_V1_ARCHITECTURE.md)) |
| **HandLiteV3** | 직선 맞춤 삼각측량 기준점 + 보정 신경망 | 133,797 | val 20.36mm |
| **HandDirect-wrist** | 신경망만으로 좌표 출력 (손목 기준 분해 + 단계적 보정) | 202,092 | val 22.8mm, 실사용에서 손 모양이 더 자연스러워 마스터 기본값 |

검토 과정(2026-09-23 리뷰 6건)에서 고친 주요 문제와, 지금 코드에 남은 규칙은 다음과 같습니다.

- **빈 이벤트 유지:** 관절을 하나도 못 본 카메라 프레임도 이벤트로 남깁니다. 버리면 그 카메라의 옛 관측이 계속 유효한 것처럼 쓰입니다.
- **늦게 온 옛 프레임 버리기:** 같은 카메라에서 더 옛 촬영이 늦게 도착하면 학습·스트림·C++ 모두 버립니다(`events.accepted_captures`).
- **입력 버퍼 소유:** 런타임은 push된 배열을 복사해 보관합니다. 호출 쪽이 버퍼를 재사용해도 이력이 바뀌지 않습니다.
- **ncnn 입력 수명 (2026-09-28):** float64 입력을 변환한 임시 배열이 먼저 해제되어 쓰레기 값을 읽던 문제를 고쳤습니다(`runtime.NcnnGraphs.run`).

## 4. 시도했지만 쓰지 않은 것

- **HandLite v1 C++ 런타임 (`cpp/hand_lite/`):** HandLiteV3 C++ 이식의 참고용으로 잠시 남겼다가, 이식이 끝나(`hand_tracker_master/src/litev3_runtime.*`) 삭제했습니다.
- **떨림 필터 (One Euro):** 마스터 출력에 넣어 봤다가 제거했습니다(모델 출력을 그대로 보냄). PC 뷰어의 `--smooth`(화면 표시용)만 남아 있습니다.
- **재투영 손실 (`--reprojection-weight`, 기본 0):** 예측 관절을 각 카메라의 명목 ray에 재투영한 각도 오차를 손실에 더하는 실험입니다. 명목 ray 기준으로는 정답 좌표의 재투영 오차(71.2mrad)가 모델 예측(약 69.5mrad)보다 오히려 커서(카메라 설치 오차 수 도·수 cm 때문), 이 손실은 정답에서 멀어지는 방향을 줍니다. 옵션만 남기고 학습에는 쓰지 않았습니다.
- **HandDirect의 참 월드 좌표 손실:** HandDirect-wrist 학습 노트북은 0.1배로 썼지만, 저장소 기본 손실에는 넣지 않았습니다(고정 배치 좌표계에서 학습).
