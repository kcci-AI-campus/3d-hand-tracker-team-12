# Hand Tracker Slave — 라즈베리 파이 UV 송신기

Google의 공식 **`hand_landmarker.task` Latest (full)**를 2026-09-23에 받아,
내부의 두 TFLite 모델을 직접 ncnn으로 변환했습니다.
기존 커뮤니티 palm-lite / hand-lite 파일은 이 프로젝트에서 삭제했습니다.

앱은 **C++ 소스 + 작은 UDP 헤더 + ncnn + 시스템 OpenCV**로 동작합니다.
MediaPipe 런타임, Bazel, TensorFlow, Python 실행 환경은 파이에 필요 없습니다.
공식 원본·모델 구조·재변환 스크립트·원본과의 수치 비교 결과를 포함했습니다.

## 파이에서 설치·빌드·실행

64비트 Raspberry Pi OS, Pi 4/5, USB/V4L2 웹캠 기준입니다.
GUI 없이 SSH/화면 없는 환경에서 실행합니다. HighGUI 의존성을 제거했습니다.
CMake 3.24 이상이 필요합니다.

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build libopencv-dev curl ca-certificates
cd hand_tracker_slave
bash scripts/build_pi.sh
bash run_slave.sh
```

**OpenCV는 apt 패키지로 사용하고 ncnn만 소스 빌드합니다.** ncnn은 `20260526`으로
고정했으며 다운로드 해시를 검사합니다. 변환된 모델 파일이 포함되어 있어
파이에서 변환 도구를 설치하거나 모델을 다운로드할 필요 없습니다.

기본 빌드 병렬도는 메모리를 아끼기 위해 1입니다.
메모리가 충분하면 `JOBS=2 bash scripts/build_pi.sh`를 사용하세요.
로그는 `build_pi.log`에 기록됩니다. `sudo`로 빌드하지 마세요.
빌드와 CTest가 성공하면 `dist/hand_tracker_slave`, `dist/models/`가 생성됩니다.
이전 빌드에 남아 있는 lite 모델 4개도 빌드 스크립트가 해당 배포 폴더에서 제거합니다.

```bash
bash run_slave.sh --camera 0 --hands 2 --fps 15 --threads 2
bash run_slave.sh --hands 1 --fps 10
bash run_slave.sh --fps 15
bash run_slave.sh --check-model
bash run_slave.sh --image testdata/hand.jpg --expect-hands 1
```

기본값: 카메라 0, 320×240, 최대 2개 손, 처리 상한 15 FPS, 추론 스레드 2개.
터미널에 실제 처리 FPS·손 개수·추론 시간을 표시합니다. Ctrl+C로 종료합니다.
`--check-model`과 `--image`는 카메라·화면 없이 실행합니다.
모델 위치는 실행 파일 옆 `models/`이며 `--models 경로`로 변경할 수 있습니다.

창 생성·화면 표시·관절 그리기·이미지 저장 기능은 제거했습니다. 터미널에 약 1초마다
`FPS: 14.9  Hands: 1  Infer: 45ms` 형태로 출력합니다. Ctrl+C로 종료합니다.
이전 `--headless`, `--print-fps` 옵션은 호환용으로 허용하지만 항상 같은 동작입니다.
FPS는 캡처·추론·대기를 포함한 최근 처리 속도이며, `Infer`는 마지막 프레임의
전처리·모델 추론·후처리 시간입니다. `--fps` 값은 목표 상한으로 실제 속도와 다릅니다.

## 실행 시 CLI로 송신 대상 설정

```bash
bash run_slave.sh
```

프로그램 실행 후 IP, 포트, 이 카메라의 rig 카메라 번호를 입력합니다. 명령줄의 `--master`/`--port` 옵션은 없습니다.

```text
Master IPv4: 192.168.0.10
Master UDP port [5001]: 5001
Rig camera id (0 or 1): 0
```

포트에서 Enter만 누르면 5001을 사용합니다. 마스터(`hand_tracker_master`) 기본값은 **카메라 0 = 포트 5001,
카메라 1 = 포트 5002**입니다. 카메라 번호는 실제 설치 위치와 맞아야 합니다(위치 표는
[../hand_tracker_master/README.md](../hand_tracker_master/README.md)). 기본값이 없으니 반드시 입력합니다.
잘못된 값은 다시 입력받습니다. 입력이 종료되면 송신을 시작하지 않고 종료합니다. 종료는 Ctrl+C입니다.

실행 중 약 1초마다 상태를 출력하고 즉시 flush하므로 SSH에서도 확인할 수 있습니다.

```text
cam0 -> 192.168.0.10:5001  FPS: 14.9  Hands: 1  Infer: 45ms  Age: 47ms  TX: 150  Failed: 0  Bytes: 53400  Last socket error: 0
```

`Age`는 마지막 프레임의 촬영→송신 시간입니다(패킷에 담아 보냄).

`TX`는 운영체제가 전송을 받아들인 누적 패킷 수, `Failed`는 로컬 전송 실패 수입니다.
UDP 응답을 받지 않으므로 **마스터 수신 성공/접속 상태를 의미하지 않습니다.**
`Bytes`는 성공한 로컬 전송의 payload 누적 바이트이며 마지막 소켓 오류는 성공 시 0으로
돌아갑니다. 마스터가 꺼져 있어도 TX가 증가할 수 있습니다.

`--check-model`과 `--image` 검사는 IP를 묻지 않습니다. 이미지에서 추론한 UV 한 번만
보내며 검증하려면 `--image testdata/hand.jpg --send-uv`를 사용하면 됩니다.
이 경우에도 IP/포트는 CLI로 입력하며 **이미지는 전송하지 않습니다.**

수신하는 마스터는 [hand_tracker_master](../hand_tracker_master/README.md)입니다. 기존 `camera/master.py`의
JPEG/TCP 수신기, 이전 UV42(336바이트, 헤더 없음) 형식과는 호환되지 않습니다.

- 매 처리 프레임마다 UDP 데이터그램 **356바이트(UV2)**, 모두 **big-endian**:

  | 오프셋 | 크기 | 내용 |
  |---|---|---|
  | 0 | 4 | `"HUV2"` |
  | 4 | 1 | 버전 2 |
  | 5 | 1 | rig 카메라 번호 (0 또는 1; 마스터 자기 카메라가 2) |
  | 6 | 2 | flags (0) |
  | 8 | 4 | 프레임 순번 (uint32, 프레임마다 +1) |
  | 12 | 4 | 촬영→송신 시간 (uint32, µs, slave 시계로 측정) |
  | 16 | 4 | 추론 시간 (uint32, µs, 진단용) |
  | 20 | 336 | IEEE754 float32 84개 (아래 배열) |
- 배열 순서: `[왼손 관절 0의 u,v, …, 왼손 20의 u,v, 오른손 0의 u,v, …, 오른손 20의 u,v]`.
- 관절 순서는 MediaPipe 21관절 순서이며 `u=x/영상폭`, `v=y/영상높이`입니다.
- 미검출 손, 화면 밖 또는 비정상 관절은 해당 UV를 `(-1,-1)`로 보냅니다.
  손이 전혀 없어도 모든 값이 -1인 프레임을 전송합니다.
- 기본 USB 영상은 비미러로 간주하며 모델의 셀피 좌우 분류를 뒤집어 슬롯에 배치합니다.
  카메라 자체가 미러 영상을 출력하면 `--mirrored`를 지정하세요. 이는 좌표를 반전하는
  옵션이 아니라 좌우 슬롯 해석 옵션입니다. 설치 후 실제 왼손/오른손으로 확인하세요.
- 같은 쪽으로 두 손이 분류되면 존재확률×좌우확률이 높은 손을 선택합니다.
- **이미지, Z, 월드 좌표, 카메라 위치, 절대 시각은 전송하지 않습니다.** ray 생성·시간 처리는 마스터 담당입니다.
- **시간**: 촬영 시각은 카메라가 프레임을 넘겨준 순간입니다. 두 파이의 시계가 달라도 되도록 절대 시각 대신
  촬영→송신 시간을 보내고, 마스터가 `수신 시각 − 촬영→송신 시간 − 네트워크 지연`으로 자기 시계에서 복원합니다.
- UDP는 누락/순서 변경이 가능합니다. 마스터는 프레임 순번으로 누락·중복·역순을 판별하고, 오래된
  프레임은 0.5초 뒤 모델 입력에서 빠집니다.

Python으로 형식만 확인하는 예:

```python
import socket, struct
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5001))
while True:
    data, source = sock.recvfrom(65535)
    if len(data) != 356 or data[:4] != b"HUV2":
        continue
    _, version, camera, _, sequence, age_us, infer_us = struct.unpack("!4sBBHIII", data[:20])
    values = struct.unpack("!84f", data[20:])
    uv = list(zip(values[0::2], values[1::2]))  # 42쌍, invalid=(-1,-1)
    print(camera, sequence, age_us / 1000, "ms", uv[0])
```

손 검출 코드는 `hand_detector.hpp`에 있고, 마스터가 자기 카메라에 같은 코드를 씁니다.

## 모델과 코드 변경

| ncnn 파일 | 입력 | 출력 |
|---|---|---|
| `hand_detector.param`, `.bin` | RGB 192×192, 0..1 | 2,016개 박스·7개 손바닥 키포인트, 검출 logit |
| `hand_landmarks_detector.param`, `.bin` | 회전 보정 RGB 224×224, 0..1 | 관절 XYZ, 손 존재 확률, 좌우 손 분류, 월드 XYZ |

공식 그래프의 가중 NMS(IoU 0.3), palm 임계값 0.5, hand presence 임계값 0.5,
회전·영역 확대, Z 보정 계수 0.4를 적용했습니다.
이미지 모드에서는 `Presence`, 모델의 오른손 분류 확률, 이미지 좌표와
회전 보정한 월드 좌표를 터미널에 출력합니다.

공식 번들의 FP16 가중치를 FP32로 정확히 확장했고 추가 양자화·재학습은 하지
않았습니다. 런타임도 FP32입니다. 이전 변환본의 packing 우회는 제거했고,
새 `InnerProduct` 출력층은 채널 packing을 켠 상태로 검증했습니다.

상세 출력명·shape·연산자 수·검증 오차는 [MODEL_STRUCTURE.md](MODEL_STRUCTURE.md)를 보세요.

## 검증과 제한

- GUI 없이 Windows 호스트에서 빌드하고 실제 모델의 한 손·두 손·빈 화면·좌우 슬롯을
  UDP loopback으로 수신해 84개 float 값을 검증했습니다(UV42 시절, `conversion/uv_checks.json`).
  `uv_protocol` CTest는 미검출·좌우 순서·중복 분류·비정상 좌표·UV2 헤더와 바이트 순서·
  디코딩을 검사합니다. UV2 변경 후 main.cpp는 Windows에서 컴파일만 확인했습니다.
- 원본 LiteRT와 **ncnn 20260526 C++**의 모든 출력을 6종 입력에서 비교했습니다.
  관절 좌표 최대 절대 차이는 약 0.00032, 월드 좌표는 약 0.0000004m였습니다.
- 수정된 C++ 소스를 Windows 호스트에서 컴파일하고 명시적 모델 경로로 이미지 모드를
  실행했습니다. 회전·반전·두 손·가로/세로 여백·빈 화면 검사 9종을 통과했습니다.
- **실제 파이 ARM64 빌드·카메라·장치 간 UDP는 미검증입니다. ARM 실행 파일은 포함하지 않습니다.**
- 공식 모델을 사용하지만 MediaPipe VIDEO 모드의 프레임 간 추적·검출 생략·스무딩
  전체를 재현한 것은 아닙니다. 매 프레임 검출하며 OpenCV 샘플링도 MediaPipe와
  완전히 같지는 않습니다. 전체 Tasks API 결과와의 동일성을 보장하지 않습니다.
- 월드 좌표는 모델이 추정한 손 중심 기준 미터 값입니다. 카메라 기준 절대 위치나
  두 손 사이 실제 거리가 아닙니다. 좌우 분류는 셀피 미러 관례의 원시 확률이며,
  비미러 영상의 물리적 좌우로 자동 변환하지 않습니다.
- USB/V4L2 전용이며 CSI/Picamera2는 포함하지 않습니다. 단일 캡처·추론 루프이므로
  최신 프레임 보장·장치 읽기 타임아웃은 없습니다. `--fps`는 속도 보장이 아닙니다.

## 모델 검사·재변환

```bash
bash scripts/verify_models.sh
```

`models/SHA256SUMS`와 다르면 빌드를 중단합니다. 변환된 ncnn 파일은 Google이
직접 배포하는 파일이 아니므로 다른 커뮤니티 모델로 대체 다운로드하지 않습니다.
손상 시 ZIP의 `models/`를 복구하거나 [개발 PC 재변환 절차](conversion/README.md)를 따르세요.

- `conversion/source/hand_landmarker.task`: 공식 Latest 원본
- `conversion/model_structure.json`: 원본 전체 텐서·연산자 구조
- `conversion/parity_report.json`: 원본/ncnn 수치 비교 및 검증 파일 해시
- `models/model_manifest.json`: 출처·변환 도구 버전·모델 해시
- [라이선스 및 출처](THIRD_PARTY.md)
