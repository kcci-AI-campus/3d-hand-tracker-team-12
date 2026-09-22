# Raspberry Pi 3-camera Hand Landmarker

라즈베리 파이 **총 3대**를 사용합니다. 각 파이에 카메라를 1개씩 연결합니다.
- **Pi 1: 로컬 카메라 추론 + 마스터**. `master.py` 하나로 실행합니다.
- **Pi 2, Pi 3: 카메라 추론 + LAN 송신**. 각각 `sender.py`를 실행합니다.

3대 모두 카메라 영상을 **너비 320 × 높이 240**으로 맞춰 MediaPipe Hand Landmarker로 추론합니다.
마스터는 로컬 추론 영상 1개와 LAN으로 받은 JPEG/랜드마크 2개를 OpenCV `imshow`로 표시합니다.
기본은 960×240 창에 3개 영상을 나란히 표시하며, 독립된 3개 창도 지원합니다.

## 구현 언어

C++ 우선 요청을 검토했으나, MediaPipe C++의 ARM 빌드 및 Bazel/네이티브 의존성 구성이 배포 부담이 됩니다.
이번 구현은 사용자가 허용한 Python 대안입니다. C++ 빌드 실패를 실제 장치에서 재현한 것은 아닙니다.
Google의 [공식 Raspberry Pi 예제](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/hand_landmarker/raspberry_pi)도 Python을 사용합니다.
Python API가 호출하는 MediaPipe 추론과 OpenCV의 주요 영상 연산은 네이티브 구현입니다.
대상 보드에서 측정한 FPS/지연 보장은 없으며, C++ 변경만으로 추론 속도가 크게 개선된다고 가정하지 않습니다.

## 설치

64비트 Raspberry Pi OS 및 유선 LAN을 전제로 합니다. 마스터는 GUI 데스크톱 세션 또는 아래의 X11 전달 설정이 필요합니다.
아래 명령은 프로젝트 폴더에서 실행합니다. USB 카메라가 기본이며 CSI 카메라는 Picamera2 옵션을 사용합니다.

모든 파이:

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-numpy
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

마스터를 포함한 **3대 모두** 추가 (기존 파일명 `requirements-sender.txt`를 공통 사용):

```bash
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements-sender.txt
python download_model.py
python -c "import cv2, mediapipe; from mediapipe.tasks.python import vision; print(mediapipe.__version__)"
python -m pip check
```

MediaPipe wheel은 OS 아키텍처와 Python 버전에 따라 제공 여부가 다릅니다. `No matching distribution`이면
`uname -m`과 `python --version`을 확인하고, 해당 버전의 Linux aarch64 wheel을 제공하는 Python 환경을 사용해야 합니다.
32비트 OS 또는 새 Python 버전에서의 설치를 보장하지 않습니다. 범위 지정은 설치 후보이며 모든 조합을 검증한 고정 환경이 아닙니다.
pip 설치 후 OS OpenCV와 NumPy ABI 충돌이 발생하면 NumPy/OpenCV/MediaPipe가 호환되는 환경을 구성해야 합니다.
USB 전용 환경에서는 system-site-packages 없이 venv를 만들고 GUI 지원 `opencv-contrib-python` wheel을 함께 설치하는 대안이 있습니다.
마스터에 `opencv-python-headless`만 설치하면 `imshow`가 동작하지 않습니다.
동작 확인 후 `python -m pip freeze > requirements-tested.txt`로 장치별 실제 버전을 기록하세요.

CSI 카메라 사용 시 해당 파이(마스터 포함)에 추가:

```bash
sudo apt install -y python3-picamera2
```

Picamera2는 위의 `--system-site-packages` 환경에서 사용합니다. NumPy를 pip로 교체하면 OS Picamera2 의존성과도 호환성을 확인해야 합니다.

## 실행

마스터 IP를 예를 들어 `192.168.1.100`으로 고정/DHCP 예약하고, 송신기에서 마스터 TCP **5001, 5002** 접근을 허용합니다.
Pi 1의 영상은 메모리로 직접 전달하므로 로컬 TCP 연결이나 JPEG 압축이 필요 없습니다.

Pi 1(추론 + 마스터)의 **로컬 데스크톱 터미널**:

```bash
source .venv/bin/activate
python master.py
# 각각 독립된 창 3개가 필요하면:
python master.py --separate-windows
```

Pi 1에서는 `sender.py`를 추가 실행하지 않습니다. 나머지 두 파이에서 각각 실행합니다:

```bash
# 송신 Pi 2
python sender.py --master 192.168.1.100 --id 2
# 송신 Pi 3
python sender.py --master 192.168.1.100 --id 3
```

마스터 CSI 예시: `python master.py --backend picamera2`

송신기 CSI 예시: `python sender.py --master 192.168.1.100 --id 2 --backend picamera2`

마스터 종료: Q, ESC, 창 닫기, Ctrl+C. 송신기 종료: Ctrl+C.
일반 SSH 연결만으로는 `imshow` 창을 볼 수 없습니다. 로컬 데스크톱에서 실행하거나 아래처럼 X11 전달을 설정하세요.
`--base-port`를 바꾸면 양쪽에 같은 값을 지정합니다. Pi 2/3은 각각 base+1/base+2 포트를 사용합니다.
Pi 1은 로컬 전용이며 base 포트(기본 5000)는 열지 않습니다. 송신기 `--id`는 2 또는 3만 허용합니다.

## VS Code에서 실행하고 VcXsrv로 화면 표시

Windows의 VS Code 터미널에서 WSL의 SSH 클라이언트로 Pi 1에 접속하고, Pi에서 실행한 OpenCV 창을 Windows의 VcXsrv에 표시합니다.
**카메라와 추론 프로그램은 Raspberry Pi에서 실행**합니다. Windows/WSL로 카메라를 옮길 필요는 없습니다.
위 설치 과정을 각 Pi에서 먼저 완료하세요.

### 1. Windows: VcXsrv 시작

[VcXsrv](https://sourceforge.net/projects/vcxsrv/)를 설치한 뒤 XLaunch를 실행합니다.

- `Multiple windows`, Display number `0` 선택
- `Start no client` 선택
- 신뢰하는 로컬 개발 환경에서 간단히 연결하려면 `Disable access control` 체크
- Windows 방화벽에서 WSL에서 오는 VcXsrv 연결을 허용

접근 제어를 끄면 접속 가능한 다른 클라이언트도 화면에 접근할 수 있으므로 방화벽 허용 범위를 WSL 주소로 제한하세요.
Display `0`의 X11 TCP 포트는 `6000`입니다. 인터넷에 공개하지 않습니다.

### 2. Pi 1: X11 전달 준비

Pi 1의 터미널에서 실행합니다.

```bash
sudo apt update
sudo apt install -y openssh-server xauth x11-apps
sudo nano /etc/ssh/sshd_config
```

SSH 서버 설정에 다음 항목이 활성화되어 있는지 확인합니다.

```text
X11Forwarding yes
```

설정을 변경했다면 검증 후 서비스를 다시 불러옵니다.

```bash
sudo sshd -t && sudo systemctl reload ssh
```

### 3. VS Code: WSL 터미널에서 Pi 1 접속

Windows에 WSL/Ubuntu를 준비하고 VS Code 터미널 프로필에서 **Ubuntu (WSL)**를 선택합니다.
PowerShell 터미널이라면 먼저 `wsl`을 입력합니다. 아래 명령은 WSL에서 실행합니다.

```bash
sudo apt update
sudo apt install -y openssh-client xauth x11-apps

# 기본 WSL2 NAT 모드: Windows 호스트 주소 사용
export DISPLAY="$(ip route show default | awk '{print $3; exit}'):0"

# VcXsrv 연결 확인: Windows에 시계 창이 뜨면 닫고 진행
xclock

# 사용자 이름과 IP를 실제 Pi 1 정보로 변경
ssh -Y <PI_USER>@192.168.1.100
```

WSL 미러링 네트워크 모드를 사용한다면 위 `export DISPLAY=...` 대신 `export DISPLAY=localhost:0`을 사용합니다.
`-Y`는 신뢰하는 Pi에만 사용합니다. 접속 후 Pi의 `DISPLAY`는 SSH가 자동 설정하므로 Windows IP로 덮어쓰지 마세요.

### 4. 접속한 Pi 1에서 마스터 실행

```bash
echo "$DISPLAY"
xclock
```

`DISPLAY`는 보통 `localhost:10.0` 형태이며, 두 번째 `xclock`도 Windows에 표시되어야 합니다.
시계 창을 닫은 뒤 실제 프로젝트 경로로 이동합니다.

```bash
cd ~/3D-Hand-Tracker
source .venv/bin/activate
python master.py
# 개별 창 3개로 표시하려면 위 명령 대신:
# python master.py --separate-windows
# CSI 카메라라면:
# python master.py --backend picamera2
```

Pi 2와 Pi 3에서는 앞의 실행 절차대로 `sender.py`를 실행합니다. `--master`에는 **Pi 1의 LAN IP**를 지정합니다.
VS Code의 실행 버튼 대신 **`ssh -Y`로 접속한 같은 터미널**에서 실행해야 해당 X11 환경을 사용합니다.
VS Code Remote-SSH 연결만으로 X11 전달이 설정되었다고 가정하지 마세요.

### 화면이 표시되지 않을 때

| 증상 | 확인 사항 |
|---|---|
| WSL에서 `xclock` 실패 | XLaunch 실행 상태, WSL 모드에 맞는 `DISPLAY`, Windows 방화벽의 WSL 접근 허용 확인 |
| Pi의 `DISPLAY`가 비어 있음 | Pi의 `xauth` 설치와 `X11Forwarding yes` 확인 후 WSL에서 `ssh -Y`로 재접속 |
| Pi에서 `xclock`은 되지만 OpenCV 창 실패 | GUI 지원 OpenCV 설치 여부 확인. `opencv-python-headless`만으로는 표시 불가 |
| 창은 뜨지만 로컬 영상이 없음 | Pi 1 카메라 연결, `--backend` 설정 및 터미널 오류 로그 확인 |
| Pi 2/3 화면이 `waiting` 또는 `STALE` | 송신기 실행, 마스터 IP, Pi 1의 TCP 5001/5002 접근 확인 |
| 화면 표시가 느림 | 유선 LAN 사용. X11 전달 자체의 표시 지연도 확인 |

이 절차는 실제 Raspberry Pi 및 VcXsrv 환경에서 별도 검증이 필요합니다.
참고: [WSL 네트워크](https://learn.microsoft.com/en-us/windows/wsl/networking),
[OpenSSH X11 전달 및 -Y 옵션](https://man.openbsd.org/ssh).

## 반영한 문제점과 개선

| 문제 | 구현 |
|---|---|
| 원본 BGR 전송 대역폭 | 원격 2대만 JPEG 품질 75로 전송. 무압축 2대×320×240×3×15fps는 약 55Mbps(프로토콜 오버헤드 제외) |
| 마스터의 추론 + 표시 부하 | 로컬 추론 전용 스레드, 최신 프레임만 공유. 로컬 JPEG 인코딩/디코딩 생략 |
| 추론/네트워크 지연이 누적 | 카메라 수집 전용 스레드가 최신 1프레임만 유지. 네트워크는 프레임마다 ACK 후 다음 추론 |
| 영상과 랜드마크 불일치 | 동일 프레임의 JPEG와 좌표를 하나의 길이 지정 패킷으로 전송 |
| 한 파이 장애로 전체 화면 정지 | 포트별 독립 수신 스레드, GUI는 메인 스레드에서 실행 |
| LAN 단절/마스터 재시작 | 송신기 자동 재접속. 수신 2초 경과 시 검은 STALE 화면으로 전환 |
| TCP 분할 수신/비정상 길이 | 정확한 길이까지 수신하고 헤더/메타데이터/이미지 크기 제한 |
| 반복 손 검출 비용 | VIDEO 모드의 tracking 활용, strictly increasing monotonic timestamp 사용 |
| CSI 카메라 접근 | OpenCV USB와 Picamera2 CSI 백엔드 분리 |

ACK는 **수신/디코딩 완료**를 뜻하며 화면 표시 완료 확인은 아닙니다. GUI도 최신 프레임만 가져옵니다.
TCP 재전송 때문에 네트워크 장애 중 지연이 생길 수 있으며 하드 실시간 시스템은 아닙니다.
수신/송신 socket timeout은 5초이고 재접속 대기는 1초입니다. timeout은 개별 I/O에 적용되며 전체 패킷의 절대 마감은 아닙니다.
원격 송신기의 카메라 고장/추론 오류는 로그와 함께 프로세스를 종료합니다.
마스터의 로컬 카메라/추론 오류는 해당 작업만 종료하고 로그를 남기며 원격 2개 화면은 계속 표시합니다.
로컬 화면은 수신 이력이 있으면 STALE, 없으면 waiting 상태로 남습니다. 로컬 복구 후 마스터를 재시작하세요.
장시간 무인 운영에는 장치 상태 감시와 프로세스 재시작 정책이 추가로 필요합니다.

`--fps 15`는 처리/전송 상한이며 실측 FPS 보장이 아닙니다. 느린 보드는 `--hands 1 --fps 10`으로 줄일 수 있습니다.
마스터도 `python master.py --hands 1 --fps 10`으로 로컬 추론 부하를 줄일 수 있습니다.
마스터가 추론과 3개 화면 표시를 함께 수행하므로 실제 장치에서 CPU 부하/발열과 표시 지연을 확인하세요.
JPEG 대역폭은 장면에 따라 달라집니다. 망이 혼잡하면 `--quality 60`을 사용하세요.
320×240은 작은 손/먼 손 검출 정확도를 제한합니다. 입력 크기는 요청대로 고정되어 있습니다.
카메라 실제 해상도가 다르면 320×240으로 resize하므로 4:3이 아닌 입력은 종횡비 왜곡이 생길 수 있습니다.

## 프로토콜과 3D 확장

프레임: network byte order `!4sII` 헤더(magic `HND1`, JSON 길이, JPEG 길이), UTF-8 JSON, JPEG.
최대 JSON 16KiB, JPEG 256KiB. ACK는 ASCII `K` 한 바이트입니다. JSON은 ID, 수집 순번,
호스트 수집 시각 `capture_unix_ns`, 추론 시간, normalized/world landmarks 및 handedness를 포함합니다.
카메라 수집 순번이 건너뛰는 것은 최신 프레임 정책상 정상입니다. 마스터는 normalized 좌표로 오버레이를 그립니다.
인증/암호화가 없는 신뢰하는 사설 LAN 전용 구현입니다. 인터넷에 포트를 공개하지 마세요.

**현재 프로그램은 3시점 화면 표시이며, 다중 카메라 3D 삼각측량은 하지 않습니다.**
각 카메라의 촬영 순간은 동기화되지 않습니다. 수집 시각도 센서 노출 시각이 아닌 호스트의 capture 반환 시각입니다.
추후 3D 복원에는 카메라 내부/외부 파라미터 보정, 좌표 대응 및 시간 동기화(요구 정확도에 따라 하드웨어 트리거)가 필요합니다.
NTP/chrony는 호스트 시계 차이를 줄일 수 있지만 노출 동기화를 보장하지 않습니다.
MediaPipe world landmarks는 각 손에 대한 좌표이며 3대 카메라의 공통 월드 좌표가 아닙니다.
표시되는 `infer ms`는 송신 측 추론 시간이며 전체 네트워크 지연이나 FPS가 아닙니다.

## 검증

외부 패키지 없이 TCP 분할 수신, 연속 프레임/ACK, 중간 연결 종료, 길이 제한 및 좌표 검증 테스트:

```bash
python -m unittest discover -s tests -v
python -m compileall -q sender.py master.py protocol.py download_model.py
```

실장치 확인: 3대 연결 → 각 영상에 손 오버레이 확인 → Pi 2 LAN 분리 → 나머지 2화면 유지와 STALE 확인
→ LAN 재연결 및 복구 → 마스터 재시작 후 송신기 자동 재접속 확인.
본 개발 환경에서는 라즈베리 파이/카메라/GUI가 연결되지 않아 실제 추론 성능과 3대 통합 동작은 검증하지 못했습니다.

공식 자료: [Hand Landmarker Python 가이드](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python),
[Raspberry Pi 예제](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/hand_landmarker/raspberry_pi).
