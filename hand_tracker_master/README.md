# Hand Tracker Master — 마스터 파이 3D 손 추정

마스터 라즈베리 파이에서 돌아가는 C++ 앱입니다. slave 파이 2대가 보낸 2D 손 관절(UV2)과 자기 카메라에서 직접 검출한 2D 관절을 합쳐 **HandDirect-wrist** 모델로 두 손 3D 관절을 추정하고, PC로 보냅니다(H3D1).

```text
slave 파이 (카메라 0) ── UV2/UDP 5001 ─┐
slave 파이 (카메라 1) ── UV2/UDP 5002 ─┼─> 마스터 파이 ── H3D1/UDP 6000 ──> PC (tools/pc_receiver.py)
마스터 파이 자기 카메라 (카메라 2) ────┘   2D→ray 특징 → HandDirect-wrist (ncnn)
```

- **입력**: slave마다 관절 u,v 84개와 촬영→송신 시간(UV2, [hand_tracker_slave](../hand_tracker_slave/README.md)), 자기 카메라는 slave와 같은 손 검출 모델(`hand_detector.hpp`)
- **모델**: `models/hand_direct_wrist/`의 ncnn 그래프 2개(encoder: 프레임마다 1번, query: 출력마다 1번). 신경망만으로 좌표를 내는 HandDirect에 손목 기준 분해·단계적 보정을 넣은 모델입니다(60 epoch, val 22.8mm).
- **출력**: 자기 카메라 프레임마다 1번(학습 데이터와 같은 출력 시점), 두 손 21관절 x,y,z(cm)
- MediaPipe·Python·PyTorch 없이 **C++ + ncnn + OpenCV**로 동작합니다.

## 설치할 때 반드시 맞출 것: 카메라 배치

모델은 **정해진 카메라 배치**(`models/rig.json`, 학습 데이터의 명목 배치)의 ray로 학습했습니다. 실제 카메라를 이 배치대로 설치해야 합니다. 좌표계는 Y가 위쪽이고, 원점은 세 카메라가 바라보는 점입니다.

| 카메라 | 장치 | 위치 (cm) | 바라보는 방향 |
|---|---|---|---|
| 0 | slave (포트 5001) | (−40, −40, 20) | 원점 (위로 올려다봄) |
| 1 | slave (포트 5002) | (40, −40, 20) | 원점 (위로 올려다봄) |
| 2 | 마스터 파이 자기 카메라 | (0, 40, 40) | 원점 (아래로 내려다봄) |

- 세 카메라 모두 320×240, 대각 화각 55°(초점거리 384.2px), 롤 0°(화면 수평)입니다. 다른 카메라·해상도·화각이면 ray가 달라져 정확도가 떨어집니다.
- 설치 오차(수 cm, 수 도)는 학습 데이터에 이미 넣어 두었지만, 그보다 크면 재학습이 필요합니다.
- slave를 실행하면 카메라 번호(0 또는 1)를 묻습니다. 위 표의 설치 위치와 맞게 입력하세요. 마스터는 포트와 카메라 번호가 다르면 그 패킷을 버리고 상태 줄의 `wrong-id`에 셉니다.
- 왼손/오른손: slave와 마스터 모두 비미러 영상을 기준으로 `[왼손, 오른손]` 순서로 넣고, 학습 데이터의 손 0/1도 같은 순서입니다(손 모양으로 확인). 카메라가 미러 영상을 내면 양쪽 모두 `--mirrored`를 쓰세요.

## 파이에서 빌드·실행

64비트 Raspberry Pi OS, Pi 4/5, USB/V4L2 웹캠 기준입니다. **`hand_tracker_slave` 폴더를 이 폴더 옆에 함께 복사하세요.** 마스터는 그 폴더의 손 검출 모델과 코드를 씁니다. slave와 같이 ncnn 20260526을 소스로 빌드하고 OpenCV는 apt 패키지를 씁니다.

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build libopencv-dev curl ca-certificates
cd hand_tracker_master
bash scripts/build_pi.sh          # 빌드 + 테스트 → dist/
bash run_master.sh --check-model  # 모델만 확인 (카메라·네트워크 없이)
bash run_master.sh                # PC IP와 포트를 입력하면 시작
```

```text
PC IPv4: 192.168.0.20
PC UDP port [6000]:
Slaves: cam0 UDP 5001, cam1 UDP 5002  ->  PC 192.168.0.20:6000 (H3D1, 532 bytes)
OUT 11.7 fps  latency 62ms  query 9.8ms  own infer 58ms  seen L/R 3/2 | cam0 11.6fps age 91ms | cam1 11.5fps age 88ms | cam2 11.7fps age 60ms | TX 1203 stale 0
```

상태 줄(약 1초마다):
- `OUT`: PC로 보낸 출력 속도(= 자기 카메라 처리 속도), `latency`: 출력 시각 − 자기 카메라 촬영 시각, `query`: 모델 실행 시간, `own infer`: 자기 카메라 손 검출 시간
- `seen L/R`: 최근 0.5초 안에 그 손을 5관절 이상 본 카메라 수. 0이면 그 손의 좌표는 모델의 추측이니 PC에서 무시하세요.
- `camN`: 카메라별 수신 속도와 촬영→도착 시간. `(no data)`는 1초 넘게 끊긴 상태입니다. slave는 `bad`(형식 오류·중복), `wrong-id`, `lost`(프레임 번호 건너뜀)도 표시합니다.
- `stale`: 같은 카메라의 더 옛 촬영이 늦게 도착해 버린 수

옵션 (`--help`):

| 옵션 | 기본값 | 내용 |
|---|---|---|
| `--camera N` | 0 | 자기 USB 카메라 번호 |
| `--fps N` | 15 | 자기 카메라 처리 상한 (= 출력 속도 상한) |
| `--hands 1\|2`, `--threads N` | 2, 2 | 손 검출 최대 손 수, 검출 스레드 |
| `--model-threads N`, `--no-fp16` | 1, fp16 켬 | HandDirect ncnn 스레드, fp16 끄기 |
| `--slave-ports P0,P1` | 5001,5002 | 카메라 0·1 slave 수신 포트 |
| `--net-latency-ms MS` | 1 | slave→마스터 네트워크 지연 추정 (촬영 시각 복원에 더함) |
| `--mirrored` | 끔 | 자기 카메라가 미러 영상일 때 |
| `--models DIR`, `--model DIR`, `--rig FILE` | `dist/models/...` | 모델·배치 파일 위치 |
| `--image FILE` | — | 카메라 대신 이미지를 반복 입력 (시험용) |
| `--check-model` | — | 손 검출 모델과 HandDirect를 한 번 실행해 확인하고 종료 |

## 시간 처리

- 두 파이의 시계를 맞출 필요가 없습니다. slave는 절대 시각 대신 **촬영→송신 시간**(`capture_to_send_us`)을 보내고, 마스터는 `촬영 시각 = 수신 시각 − 촬영→송신 시간 − --net-latency-ms`로 자기 시계에서 복원합니다. 유선/같은 공유기라면 네트워크 지연은 보통 1ms 안팎입니다.
- 촬영 시각은 양쪽 모두 "카메라가 프레임을 넘겨준 순간"입니다(실제 노출은 조금 더 이릅니다). 모든 카메라가 같은 기준이라 서로 일관됩니다.
- 모델 입력은 학습 데이터와 같습니다: 관절별 u,v, 명목 카메라 중심, 명목 pinhole ray, 촬영→도착 지연. 출력 시점은 자기 카메라 키포인트가 준비된 순간으로, 학습 데이터(`query_sync_camera3`)와 같습니다.
- 카메라마다 최근 0.5초 안의 최신 8프레임만 씁니다. slave가 끊기면 0.5초 뒤부터 그 카메라 없이 추정합니다.

## PC에서 받기

```bash
python hand_tracker_master/tools/pc_receiver.py            # 1초마다 요약 출력
python hand_tracker_master/tools/pc_receiver.py --plot     # 3D 실시간 보기 (matplotlib)
python hand_tracker_master/tools/pc_receiver.py --save run.npz
```

H3D1 패킷(UDP, big-endian, 532바이트, [src/pc_packet.hpp](src/pc_packet.hpp)):

| 오프셋 | 크기 | 내용 |
|---|---|---|
| 0 | 4 | `"H3D1"` |
| 4 | 1 | 버전 1 |
| 5, 6 | 1, 2 | flags, 예약 (0) |
| 8 | 4 | 출력 순번 (uint32) |
| 12 | 8 | 출력 시각 (float64, 초, 마스터 시계: 차이만 의미 있음) |
| 20 | 4 | 지연 (float32, ms): 출력 시각 − 자기 카메라 촬영 시각 |
| 24 | 2 | 왼손·오른손을 본 카메라 수 (uint8 각각; 0 = 안 보임) |
| 26 | 2 | 예약 (0) |
| 28 | 504 | float32 126개: `[왼손 21][오른손 21]` 관절 x,y,z (cm, 배치 좌표계, Y 위쪽) |

관절 순서는 MediaPipe 21관절(손목 0, 엄지 1–4, 검지 5–8, 중지 9–12, 약지 13–16, 소지 17–20)입니다.

## slave 없이 시험하기

PC에서 검증 클립의 카메라 0·1 프레임을 실제 slave처럼 UV2로 재생합니다(저장소 루트에서, 데이터셋 필요).

```bash
python hand_tracker_master/tools/fake_slaves.py --master 192.168.0.10
```

마스터의 자기 카메라는 실제 카메라나 `--image`를 쓰므로 재생한 장면과 내용은 다르지만, 수신·시간 복원·모델·PC 송신 전체 경로를 확인할 수 있습니다.

## 검증

`ctest`(빌드 스크립트가 실행)는 네 가지를 검사합니다.

| 테스트 | 내용 | 결과 (Windows MinGW 빌드, 2026-09-28) |
|---|---|---|
| `runtime_golden` | C++ 런타임이 실제 검증 클립 이벤트(push 120, query 41, 중복 촬영·0.5초 끊김 포함)에서 Python 기준과 같은 push 결과·손별 카메라 수·좌표를 내는지 | 최대 차이 2.1e-5 단위 (0.009mm) |
| `rig_features` | slave 형식 u,v로 만든 특징이 데이터셋 특징과 같은지 (`models/rig.json`) | 최대 차이 6e-8 |
| `packets` | H3D1·UV2 바이트 배치와 왕복 | 통과 |
| `model_smoke` | `--check-model` | 통과, query 3.1ms (PC) |

- 기준 데이터는 `tools/make_golden.py`가 만듭니다. 그 안의 Python 기준 런타임은 모델을 학습한 노트북(`gigahands_colab_wrist_refine.ipynb`)의 `DirectRuntime`과 같은 값을 냅니다(차이 0, ncnn 같은 설정).
- PC에서 가짜 slave 2대 + 마스터(`--image`) + `pc_receiver.py`로 전체 경로를 돌려, 출력 약 10.7fps·손실 0을 확인했습니다.
- **실제 파이 ARM 빌드·카메라·장치 간 통신은 아직 검증하지 않았습니다.** ARM에서 fp16을 쓰면 좌표가 최대 0.8mm(2e-3 단위) 정도 달라질 수 있습니다(`--no-fp16`으로 끔).

## 모델·배치 파일 다시 만들기 (개발 PC)

```bash
# 1) 체크포인트를 ncnn으로: 모델을 학습한 노트북의 코드로 내보냅니다 (encoder/query + direct.json)
python -m training.export --checkpoint checkpoints/hand_direct_wrist/last.pt --output hand_tracker_master/models/hand_direct_wrist --formats ncnn --check-input exports/gigahands_pi3_overlap/val/clip_00001.npz
# 2) 배치: 학습 데이터의 명목 배치
python hand_tracker_master/tools/export_rig.py --data exports/gigahands_pi3_overlap
# 3) C++ 테스트 기준 데이터
python hand_tracker_master/tools/make_golden.py
```

`training.export`는 새 폴더에만 쓰므로 기존 `models/hand_direct_wrist`를 지우고 실행하세요. 1번의 `training.export`는 HandDirect-wrist 코드가 있는 노트북 소스에서 실행해야 합니다(이 저장소의 `hand_tracking`에는 아직 wrist 구조가 없습니다). 그 노트북 코드의 ncnn 실행부에는 float64 입력 메모리 버그가 있으니, 내보내기 검사를 하기 전에 저장소 `hand_tracking/runtime.py`의 `NcnnGraphs.run`처럼 고쳐서 쓰세요. 모델 입출력 형식이 같으면 C++ 코드는 바꿀 필요가 없습니다.

## 코드 구성

| 파일 | 내용 |
|---|---|
| `src/main.cpp` | 스레드 구성: slave 수신 2개, 자기 카메라 1개, 런타임·PC 송신 |
| `src/direct_runtime.*` | HandDirect 런타임 (Python `DirectRuntime` 이식, 표준 라이브러리만) |
| `src/ncnn_graphs.*` | ncnn 그래프 실행 |
| `src/rig.*` | `rig.json` 읽기, u,v → ray 특징 |
| `src/pc_packet.hpp`, `src/udp.hpp`, `src/json.hpp` | H3D1 패킷, UDP 소켓, 설정 파일 읽기 |
| `../hand_tracker_slave/hand_detector.hpp`, `uv_sender.hpp`, `cli.hpp` | 손 검출, UV2 프로토콜, 주소 입력 (slave와 공유) |
