# 3D-Hand-Tracker

Raspberry Pi 3대와 MediaPipe Hand Landmarker를 이용한 다중 시점 손 랜드마크 추적 프로젝트입니다. 각 장치에서 카메라 영상을 추론하고, 마스터에서 세 카메라의 영상과 손 랜드마크를 함께 표시합니다.

현재 구현은 **3개 시점의 손 추적 및 시각화**를 지원합니다. 카메라 보정, 촬영 동기화 및 삼각측량을 통한 통합 3D 좌표 복원은 구현되어 있지 않습니다.

## 주요 기능

- 각 Raspberry Pi에서 320×240 영상의 손 랜드마크 추론
- 로컬 카메라 1대와 원격 카메라 2대의 영상을 마스터에서 표시
- 960×240 통합 창 또는 카메라별 독립 창 지원
- TCP를 통한 JPEG 영상 및 랜드마크 전송
- 최신 프레임 우선 처리, 송신기 자동 재접속 및 오래된 영상의 `STALE` 표시
- USB 카메라와 Picamera2 기반 CSI 카메라 지원

## 시스템 구성

| 장치 | 역할 | 실행 파일 |
|---|---|---|
| Pi 1 | 로컬 추론, 원격 영상 수신 및 화면 표시 | `master.py` |
| Pi 2 | 카메라 추론 및 Pi 1로 전송 | `sender.py --id 2` |
| Pi 3 | 카메라 추론 및 Pi 1로 전송 | `sender.py --id 3` |

각 Pi에 카메라 1대를 연결합니다. Pi 2와 Pi 3은 각각 Pi 1의 TCP 포트 **5001**, **5002**에 접속합니다. Pi 1에서는 `sender.py`를 별도로 실행하지 않습니다.

## 요구 환경

- Raspberry Pi 3대와 카메라 3대
- 64비트 Raspberry Pi OS 및 해당 Python 버전용 MediaPipe wheel
- 장치 간 통신이 가능한 사설 LAN — 유선 연결 권장
- 마스터의 GUI 데스크톱 또는 X11 전달 환경

## 설치

프로젝트를 각 Pi에 내려받고 프로젝트 폴더에서 다음 명령을 실행합니다.

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-numpy

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements-sender.txt
python download_model.py
python -m pip check
```

`requirements-sender.txt`는 마스터를 포함한 **3대 모두**에서 사용합니다. 모델은 `models/hand_landmarker.task`에 다운로드됩니다.

CSI 카메라를 사용하는 Pi에는 Picamera2를 추가 설치합니다.

```bash
sudo apt install -y python3-picamera2
```

MediaPipe 설치 가능 여부는 OS 아키텍처와 Python 버전에 따라 달라집니다. 설치 오류 및 NumPy/OpenCV 호환성에 관한 설명은 [상세 가이드](GUIDE.md#설치)를 참고하세요. 화면 표시에는 GUI를 지원하는 OpenCV가 필요합니다.

## 실행

아래 예시는 Pi 1의 LAN IP가 `192.168.1.100`인 경우입니다. 실제 주소로 변경하세요. 각 장치의 프로젝트 폴더에서 실행합니다.

### Pi 1 — 마스터

GUI 데스크톱 터미널 또는 X11 전달이 설정된 터미널에서 실행합니다.

```bash
source .venv/bin/activate
python master.py
```

### Pi 2 — 송신기

```bash
source .venv/bin/activate
python sender.py --master 192.168.1.100 --id 2
```

### Pi 3 — 송신기

```bash
source .venv/bin/activate
python sender.py --master 192.168.1.100 --id 3
```

마스터는 `Q`, `Esc`, 창 닫기 또는 `Ctrl+C`로 종료합니다. 송신기는 `Ctrl+C`로 종료합니다.

### 실행 옵션

| 옵션 | 적용 대상 | 설명 |
|---|---|---|
| `--separate-windows` | 마스터 | 카메라별 창 3개로 표시 |
| `--backend picamera2` | 공통 | CSI 카메라 사용 |
| `--camera 1` | 공통 | USB 카메라 인덱스 지정, 기본값 `0` |
| `--hands 1` | 공통 | 검출할 손의 최대 개수를 1개로 제한 |
| `--fps 10` | 공통 | 처리 속도 상한 지정, 기본값 `15` |
| `--quality 60` | 송신기 | JPEG 품질 지정, 기본값 `75` |
| `--base-port 7000` | 공통 | 기본 포트 변경. 송신 연결은 base+1, base+2 사용 |

포트를 변경할 때는 마스터와 두 송신기에 같은 `--base-port` 값을 지정합니다. 전체 옵션은 `python master.py --help` 및 `python sender.py --help`로 확인할 수 있습니다.

Windows에서 VS Code로 실행하고 VcXsrv로 화면을 표시하는 절차는 [VS Code·VcXsrv 가이드](GUIDE.md#vs-code에서-실행하고-vcxsrv로-화면-표시)를 참고하세요.

## 프로젝트 구조

```text
3D-Hand-Tracker/
├── master.py                # 로컬 추론, 원격 영상 수신 및 화면 표시
├── sender.py                # 카메라 추론 및 TCP 송신
├── protocol.py              # 프레임 패킷 인코딩 및 수신
├── download_model.py        # Hand Landmarker 모델 다운로드
├── requirements-sender.txt  # 공통 Python 의존성
├── tests/                   # 프로토콜 및 로컬 마스터 테스트
├── GUIDE.md                 # 상세 설정, VcXsrv 및 구현 설명
└── README.md
```

## 테스트

```bash
python -m unittest discover -s tests -v
python -m compileall -q sender.py master.py protocol.py download_model.py
```

단위 테스트와 별도로 실제 Pi, 카메라 및 GUI 환경에서 통합 동작을 확인해야 합니다. 실제 장치의 FPS와 지연은 측정되지 않았으며, `--fps` 값은 성능 보장이 아닌 처리 상한입니다.

## 제한 사항

- 카메라 촬영 시각이 동기화되지 않으며, MediaPipe world landmarks는 카메라 간 공통 좌표가 아닙니다.
- 인증과 암호화가 없는 TCP 통신을 사용하므로 신뢰하는 사설 LAN에서 실행합니다.
- 마스터의 로컬 카메라 오류 이후에는 원인을 해결하고 마스터를 재시작해야 합니다.

프로토콜, 장애 처리 및 3D 확장에 관한 설명은 [상세 가이드](GUIDE.md)를 참고하세요.
