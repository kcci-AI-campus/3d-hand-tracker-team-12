# 3D-Hand-Tracker

## 두 손 3D 포즈 모델

비동기로 도착하는 카메라 3대의 2D 손 관절로 원하는 시각의 두 손 3D 관절을 예측하는 모델(기본 HandLiteV3: 직선 맞춤 삼각측량 기준점 + 신경망 보정, 비교용 HandDirect: 신경망만으로 좌표 직접 출력)과 학습/검증/추론/배포 코드를 제공합니다. 구조와 실행 방법은 [docs/MODELS.md](docs/MODELS.md)를 참고하세요. 본학습은 별도 실행하며 실시간 수신기와의 자동 연결은 아직 포함하지 않습니다.

## GigaHands 학습 데이터 생성 GUI

`python gigahands_sim.py`로 3대 카메라 투영, ray·타임스탬프 생성, 보정 오차·통신 지연 시뮬레이션, 3D 애니메이션 재생 및 모델 학습용 NPZ 저장을 실행할 수 있습니다. 설치·입력 형식·설정·학습 데이터 계약은 [GIGAHANDS.md](GIGAHANDS.md)를 참고하세요.

Raspberry Pi 3대와 MediaPipe Hand Landmarker를 이용한 다중 시점 손 랜드마크 추적 프로젝트입니다. 각 장치에서 카메라 영상을 추론하고, 마스터에서 세 카메라의 영상과 손 랜드마크를 함께 표시합니다.

실시간 수신 화면은 **3개 시점의 손 추적 및 시각화**를 지원합니다. 카메라 보정, 촬영 동기화 및 삼각측량을 통한 통합 3D 좌표 복원은 아직 실시간 수신기에 연결되어 있지 않습니다. 모델의 오프라인 학습·추론은 위 학습 코드로 실행합니다.

## 주요 기능

- 각 Raspberry Pi에서 320×240 영상의 손 랜드마크 추론
- 로컬 카메라 1대와 원격 카메라 2대의 영상을 마스터에서 표시
- 960×240 통합 창 또는 카메라별 독립 창 지원
- TCP를 통한 JPEG 영상 및 랜드마크 전송
- 최신 프레임 우선 처리, 송신기 자동 재접속 및 오래된 영상의 `STALE` 표시
- USB 카메라와 Picamera2 기반 CSI 카메라 지원
- 카메라 1대만으로 실행하는 독립형 로컬 화면 지원 (`camera/local_camera.py`)

## 시스템 구성

| 장치 | 역할 | 실행 파일 |
|---|---|---|
| Pi 1 | 로컬 추론, 원격 영상 수신 및 화면 표시 | `camera/master.py` |
| Pi 2 | 카메라 추론 및 Pi 1로 전송 | `camera/sender.py --id 2` |
| Pi 3 | 카메라 추론 및 Pi 1로 전송 | `camera/sender.py --id 3` |

각 Pi에 카메라 1대를 연결합니다. Pi 2와 Pi 3은 각각 Pi 1의 TCP 포트 **5001**, **5002**에 접속합니다. Pi 1에서는 `camera/sender.py`를 별도로 실행하지 않습니다.

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
python camera/download_model.py
python -m pip check
```

`requirements-sender.txt`는 마스터를 포함한 **3대 모두**에서 사용합니다. 모델은 `camera/models/hand_landmarker.task`에 다운로드됩니다.

CSI 카메라를 사용하는 Pi에는 Picamera2를 추가 설치합니다.

```bash
sudo apt install -y python3-picamera2
```

MediaPipe 설치 가능 여부는 OS 아키텍처와 Python 버전에 따라 달라집니다. 설치 오류 및 NumPy/OpenCV 호환성에 관한 설명은 [상세 가이드](docs/GUIDE.md#설치)를 참고하세요. 화면 표시에는 GUI를 지원하는 OpenCV가 필요합니다.

## 실행

### 자신의 카메라만 사용하기

위 설치를 마친 장치 한 대에서 다음 명령을 실행하면, 해당 장치의 카메라에 Hand Landmarker를 실행하고 영상 위에 손 관절 21개와 연결선, 검출된 손 개수 및 추론 시간을 표시합니다. 다른 Pi나 네트워크 연결은 필요하지 않습니다. GUI를 지원하는 OpenCV와 데스크톱 또는 X11 전달 환경이 필요합니다.

```bash
python camera/local_camera.py
```

기본값은 USB 카메라 `0`, 해상도 320×240, 최대 2개 손, 처리 상한 15 FPS입니다. 다른 USB 카메라 또는 CSI 카메라는 다음과 같이 선택합니다.

```bash
python camera/local_camera.py --camera 1
python camera/local_camera.py --backend picamera2
python camera/local_camera.py --hands 1 --fps 10
```

`Q`, `Esc`, 창 닫기 또는 `Ctrl+C`로 종료합니다. 모델 경로는 `--model`로 지정할 수 있으며, 전체 옵션은 `python camera/local_camera.py --help`로 확인합니다.

### World Landmarks 3D 와이어프레임 보기

로컬 웹캠에서 추론한 `hand_world_landmarks`를 3D 와이어프레임으로 표시합니다. 기존 MediaPipe·NumPy·GUI 지원 OpenCV 환경을 사용하며 추가 렌더링 패키지는 필요하지 않습니다.

```bash
python camera/download_model.py
python camera/world_landmarks_viewer.py
```

Windows에서 패키지가 없다면 `python -m pip install -r requirements-sender.txt opencv-contrib-python numpy`로 설치합니다. `opencv-python-headless` 환경에서는 창을 표시할 수 없습니다.

- 마우스 왼쪽 버튼 드래그: 시점 회전
- 마우스 휠 또는 `+` / `-`: 확대·축소
- `Space`: 현재 영상과 관절 정지·재개 (정지 상태에서도 회전 가능)
- `R`: 시점 초기화, `I`: 관절 번호 표시
- `Q` / `Esc` 또는 창 닫기: 종료

`--camera 1`, `--hands 1`, `--fps 10`, `--backend picamera2`, `--model 경로` 옵션을 지원합니다. 왼쪽에는 카메라 영상, 오른쪽에는 손별 3D 패널을 표시합니다. 두 패널의 시점은 함께 회전합니다. 검출 순서가 바뀌면 손의 패널 위치도 바뀔 수 있습니다.

좌표는 모델이 추정한 미터 단위 값을 그대로 사용하며 격자 간격은 2cm입니다. 각 손의 원점은 손의 기하학적 중심 부근입니다. 두 손 사이의 실제 거리나 카메라 기준 절대 위치를 표현하는 좌표가 아니므로 손별로 분리해서 표시합니다. 기본 시점은 이미지와 같이 +X가 오른쪽, +Y가 아래쪽이며 축 표시도 함께 회전합니다.

### 세 장치로 실행하기

아래 예시는 Pi 1의 LAN IP가 `192.168.1.100`인 경우입니다. 실제 주소로 변경하세요. 각 장치의 프로젝트 폴더에서 실행합니다.

### Pi 1 — 마스터

GUI 데스크톱 터미널 또는 X11 전달이 설정된 터미널에서 실행합니다.

```bash
source .venv/bin/activate
python camera/master.py
```

### Pi 2 — 송신기

```bash
source .venv/bin/activate
python camera/sender.py --master 192.168.1.100 --id 2
```

### Pi 3 — 송신기

```bash
source .venv/bin/activate
python camera/sender.py --master 192.168.1.100 --id 3
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

포트를 변경할 때는 마스터와 두 송신기에 같은 `--base-port` 값을 지정합니다. 전체 옵션은 `python camera/master.py --help` 및 `python camera/sender.py --help`로 확인할 수 있습니다.

Windows에서 VS Code로 실행하고 VcXsrv로 화면을 표시하는 절차는 [VS Code·VcXsrv 가이드](docs/GUIDE.md#vs-code에서-실행하고-vcxsrv로-화면-표시)를 참고하세요.

## 프로젝트 구조

```text
3D-Hand-Tracker/
├── camera/                  # MediaPipe 카메라·네트워크 (루트에서 python camera/<파일>.py)
│   ├── master.py            #   로컬 추론, 원격 영상 수신 및 화면 표시
│   ├── sender.py            #   카메라 추론 및 TCP 송신
│   ├── protocol.py          #   프레임 패킷 인코딩 및 수신
│   ├── local_camera.py      #   자신의 카메라만 추론하고 화면 표시 (네트워크 없음)
│   ├── world_landmarks_viewer.py
│   ├── latency_recorder.py, run_latency.ps1   # 지연 측정 (docs/LATENCY.md)
│   └── download_model.py    #   Hand Landmarker 모델 → camera/models/
├── hand_tracking/           # 손 포즈 모델 라이브러리: HandLiteV3, HandDirect (docs/MODELS.md)
├── training/                # 학습·평가·추론·내보내기 (python -m training.<이름>)
│   ├── train.py, evaluate.py, predict.py
│   ├── export.py            #   ncnn 내보내기 + 런타임 검증
│   └── build_colab_notebook.py
├── gigahands_*.py, motion_archive.py, npz_reader.py, start_gigahands.ps1
│                            # GigaHands 시뮬레이터·데이터셋 생성 (GIGAHANDS.md)
├── cpp/hand_lite/           # HandLite v1 C++ 런타임 (보관용, HandLiteV3 이식 참고)
├── local_camera_ncnn/       # Windows ncnn 카메라 앱 (실행 파일은 dist/)
├── local_camera_ncnn_pi/    # 라즈베리 파이 ncnn 카메라 앱
├── hand_tracker_slave/      # slave 파이: 카메라 → 2D 손 관절 → UV2/UDP (마스터로)
├── hand_tracker_master/     # 마스터 파이: slave 2대 + 자기 카메라 → HandDirect-wrist 또는 HandLiteV3 → 3D 관절 H3D1/UDP (PC로)
├── notebooks/               # Colab 노트북
├── tests/                   # python -m unittest discover -s tests
├── docs/                    # GUIDE, MODELS, HANDLITEV3·HANDDIRECT_ARCHITECTURE, LATENCY, reviews/, archive/
├── requirements-*.txt
├── exports/, samples/, profiles/, latency_logs/, runs/, dist/   # 데이터·결과 (대부분 git 제외)
├── GIGAHANDS.md
└── README.md
```

## YOLO26n Pose 학습 (Colab)

재라벨링 노트북의 입력은 **Ultralytics Hand Keypoints 데이터셋**입니다. 노트북 내부에서 [Ultralytics 공식 배포 ZIP](https://github.com/ultralytics/assets/releases/download/v0.0.0/hand-keypoints.zip)을 직접 다운로드하므로 별도 입력 파일을 준비할 필요가 없습니다. 이 데이터셋은 MediaPipe로 자동 주석된 hand 1클래스·21개 관절 데이터이며, Google의 MediaPipe 학습 원본 데이터셋과는 다릅니다.

기존 관절·visibility를 유지하면서 좌우 클래스를 추가하려면 [MediaPipe 재라벨링 노트북](notebooks/relabel_mediapipe_hand_pose.ipynb)을 실행합니다. 원본 손과 새 MediaPipe 예측을 일대일 대응시킨 뒤, `v>0` 관절의 픽셀 차이가 원본 bbox 대각선 대비 평균 5% 이하·최대 15% 이하인 이미지만 채택합니다. 기준은 설정 셀에서 조정할 수 있습니다. 원본 이미지 바이트와 라벨의 bbox·키포인트·visibility 토큰을 보존하고 클래스만 `left_hand/right_hand`로 교체합니다. 손 개수 차이·모호한 매칭·좌우 점수 미달 등 제외 사유와 비교 거리를 기록합니다. 생성 ZIP은 아래 학습 노트북의 `DATA_SOURCE="custom_handedness"`와 `DATA_ZIP`에 지정합니다.

handedness는 자동 생성한 의사 라벨입니다. 기존 visibility의 0/1/2를 그대로 보존하며, 관절별 confidence를 생성하지 않습니다. 미리보기는 원본 관절(초록)과 새 추론(자홍)을 겹쳐 표시합니다. 두 결과가 일치해도 정답을 보장하지 않으므로 검수해야 합니다. 검출 실패나 어느 손이든 불일치한 이미지는 배경으로 간주하지 않고 전체 제외합니다.

[Colab용 학습 노트북](notebooks/train_yolo26n_hand_pose.ipynb)을 Colab에 업로드하고 GPU 런타임에서 실행합니다. 기본 설정은 **MediaPipe로 라벨링된 공개 Hand Keypoints 데이터셋**을 자동 다운로드하여 `yolo26n-pose.pt`를 `hand` 1클래스, 손 관절 21개로 파인튜닝합니다. Google의 MediaPipe 모델 학습 원본 데이터셋을 의미하지 않습니다.

- 기본 실행에는 데이터 업로드가 필요하지 않습니다. [데이터 출처 및 이용 조건](https://docs.ultralytics.com/datasets/pose/hand-keypoints/)을 확인하세요.
- 공개 데이터에는 좌우 구분이 없어 기본 결과의 `handedness`는 `null`입니다. `DATA_SOURCE="custom_handedness"`를 선택하면 `0=left_hand`, `1=right_hand`로 직접 라벨링한 ZIP으로 2클래스 학습이 가능합니다.
- 라벨 검사, 정답 미리보기, 학습, 검증, 관절별 confidence 시각화 및 결과 ZIP 다운로드를 포함합니다.
- 좌우 반전 증강은 끄며, 선택적으로 Google Drive에 체크포인트를 저장합니다.
- 관절 confidence는 실제 visibility 확률과 다릅니다. 현재 카메라 프로그램은 MediaPipe를 사용하며, 학습한 YOLO 가중치의 실시간 연결은 별도 구현이 필요합니다.

## 재라벨링 데이터 검수

한 장씩 보면서 삭제하려면 Google Drive의 같은 폴더에 [이미지·라벨 삭제 노트북](notebooks/curate_hand_keypoints.ipynb)과 데이터 ZIP을 놓고 Colab에서 실행하세요. Drive를 마운트하고 `NOTEBOOK_DIR`에 그 폴더 경로를 지정합니다. ZIP 하나는 자동 선택하며, 여러 개면 `ZIP_NAME`에 파일명만 입력합니다. **현재 보는 이미지와 라벨 한 쌍만 Colab 임시 폴더에 압축 해제**하고, 다른 이미지로 이동하면 이전 캐시를 지웁니다. 삭제 버튼은 현재 캐시를 지우고 Drive의 `review_state/ZIP이름/review_state.json`에 삭제 목록을 저장해 즉시 목록에서 제외합니다. 마지막에 남은 파일을 원본 ZIP에서 하나씩 읽어 새 ZIP에 저장하므로 전체 압축 해제는 하지 않습니다. 원본 ZIP은 검수 중 수정하지 않습니다.

## 테스트

```bash
python -m unittest discover -s tests -v
python -m compileall -q camera training hand_tracking
```

단위 테스트와 별도로 실제 Pi, 카메라 및 GUI 환경에서 통합 동작을 확인해야 합니다. 실제 장치의 FPS와 지연은 측정되지 않았으며, `--fps` 값은 성능 보장이 아닌 처리 상한입니다.

## 제한 사항

- 카메라 촬영 시각이 동기화되지 않으며, MediaPipe world landmarks는 카메라 간 공통 좌표가 아닙니다.
- 인증과 암호화가 없는 TCP 통신을 사용하므로 신뢰하는 사설 LAN에서 실행합니다.
- 마스터의 로컬 카메라 오류 이후에는 원인을 해결하고 마스터를 재시작해야 합니다.

프로토콜, 장애 처리 및 3D 확장에 관한 설명은 [상세 가이드](docs/GUIDE.md)를 참고하세요.
