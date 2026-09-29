# 3D-Hand-Tracker

라즈베리 파이 3대의 카메라로 **두 손의 3D 관절 42개를 실시간 추정**하는 프로젝트입니다. 각 파이가 2D 손 관절을 검출하고, 마스터 파이가 서로 다른 시각에 도착한 세 카메라의 관측을 신경망으로 합쳐 3D 좌표를 만든 뒤 PC로 보냅니다. 학습 데이터는 GigaHands 3D 손 동작을 같은 카메라 배치로 투영해 만들었습니다.

```text
slave 파이 (카메라 0) ─ 2D 관절 UV2/UDP ─┐
slave 파이 (카메라 1) ─ 2D 관절 UV2/UDP ─┼─> 마스터 파이 (카메라 2) ── 3D 관절 H3D1/UDP ──> PC 3D 뷰어
                                        │   2D → ray 특징 → HandDirect-wrist / HandLiteV3 (ncnn)
```

| 단계 | 내용 | 폴더 |
|---|---|---|
| 2D 검출 | 공식 MediaPipe Hand Landmarker(`hand_landmarker.task`)를 ncnn으로 변환해 C++로 실행, 320×240 | [hand_tracker_slave](hand_tracker_slave/README.md) |
| 3D 추정 | slave 2대 + 자기 카메라의 최근 0.5초 관측 → 두 손 3D 관절. 촬영 시각 동기화 없이 촬영→송신 시간으로 복원 | [hand_tracker_master](hand_tracker_master/README.md) |
| 표시 | H3D1 패킷 수신, 손 3D 렌더링 | [hand_tracker_master/tools](hand_tracker_master/tools) |
| 모델 | 학습·평가·ncnn 내보내기 (PyTorch) | [hand_tracking](hand_tracking), [training](training), [docs/MODELS.md](docs/MODELS.md) |
| 학습 데이터 | GigaHands → 3카메라 비동기 투영, 설치 오차·지연·미검출·오검출 시뮬레이션 | [GIGAHANDS.md](GIGAHANDS.md) |

## 모델과 결과

| | **HandDirect-wrist** (마스터 기본) | **HandLiteV3** |
|---|---|---|
| 방식 | 신경망만으로 좌표 직접 출력 (손목 기준 분해 + 단계적 보정) | 직선 맞춤 삼각측량 기준점 + 보정 신경망 |
| 파라미터 | 202,092 | 133,797 |
| 검증 MPJPE | 22.8mm (GPU 60 epoch) | 20.36mm (TPU 85 epoch) |
| 추가 출력 | — | 관절별 예상 오차, 손별 '시야 안' 확률 |
| 파이 query 시간 | 5–15ms | — |

- 수치는 HandLiteV3가 앞서지만(장편 클립에서는 차이 0.7mm 이하), 실제 장치에서는 HandDirect-wrist가 손 모양을 더 자연스럽게 유지해 마스터 기본값으로 씁니다. `--arch litev3`로 바꿀 수 있습니다.
- 구조: [HANDDIRECT_ARCHITECTURE.md](docs/HANDDIRECT_ARCHITECTURE.md), [HANDLITEV3_ARCHITECTURE.md](docs/HANDLITEV3_ARCHITECTURE.md)

## 실행

**카메라 배치**: 모델은 정해진 배치로 학습했습니다. 카메라 0 (−40, −40, 20)cm, 카메라 1 (40, −40, 20)cm, 카메라 2 (0, 40, 40)cm, 모두 원점을 바라봅니다(Y 위쪽). 자세한 내용은 [마스터 README](hand_tracker_master/README.md#설치할-때-반드시-맞출-것-카메라-배치)를 보세요.

```bash
# slave 파이 2대 (실행하면 마스터 IP와 카메라 번호 0/1을 묻습니다)
cd hand_tracker_slave && bash scripts/build_pi.sh && bash run_slave.sh

# 마스터 파이 (hand_tracker_slave 폴더를 옆에 함께 복사; 실행하면 PC IP를 묻습니다)
cd hand_tracker_master && bash scripts/build_pi.sh && bash run_master.sh
```

```bash
# PC: 3D 보기
pip install numpy pyqtgraph PyQt6 PyOpenGL
python hand_tracker_master/tools/hand_viewer.py
```

## 학습

```powershell
python -m venv .venv-transformer
.\.venv-transformer\Scripts\python.exe -m pip install -r requirements-transformer.txt
.\.venv-transformer\Scripts\python.exe -m training.train --arch direct --data exports/gigahands_pi3_overlap --output runs/direct --cache-dir runs/cache
.\.venv-transformer\Scripts\python.exe -m training.export --checkpoint runs/direct/best.pt --output runs/direct/deploy --check-input exports/gigahands_pi3_overlap/val/clip_00001.npz
```

- Colab(TPU)은 [notebooks/gigahands_colab.ipynb](notebooks/gigahands_colab.ipynb)를 씁니다(현재 소스를 내장, `python -m training.build_colab_notebook`으로 다시 생성).
- 학습 데이터 생성: `python gigahands_sim.py` (GUI), 일괄 생성은 [GIGAHANDS.md](GIGAHANDS.md)
- 옵션·지표·배포는 [docs/MODELS.md](docs/MODELS.md)

## 프로젝트 구조

```text
3D-Hand-Tracker/
├── hand_tracker_slave/      # slave 파이: 카메라 → 2D 손 관절 → UV2/UDP (C++, ncnn)
├── hand_tracker_master/     # 마스터 파이: slave 2대 + 자기 카메라 → 3D 관절 → H3D1/UDP (C++, ncnn)
│   └── tools/               #   PC 수신기·3D 뷰어, 가짜 slave, 모델·배치·golden 데이터 생성
├── hand_tracking/           # 모델 라이브러리: HandDirect, HandLiteV3, 데이터, 런타임 (PyTorch/NumPy)
├── training/                # 학습·평가·추론·내보내기·Colab 노트북 생성 (python -m training.<이름>)
├── notebooks/               # Colab 학습 노트북
├── gigahands_*.py, motion_archive.py, npz_reader.py, dataset_quality.py, start_gigahands.ps1
│                            # GigaHands 시뮬레이터·학습 데이터 생성 (GIGAHANDS.md)
├── profiles/, samples/      # 실측 타이밍 프로필, 생성 설정·예시
├── tests/                   # python -m unittest discover -s tests
├── docs/                    # MODELS, 모델 구조 문서, archive/ (개발 이력)
└── requirements-*.txt
```

## 테스트

```bash
python -m unittest discover -s tests          # Python (모델·데이터·시뮬레이터)
ctest --test-dir hand_tracker_master/build    # C++ (build_pi.sh가 실행): golden 비교, 패킷, 특징
```

## 개발 이력

MediaPipe Python 프로토타입(JPEG/TCP 3화면 표시), 지연 측정, YOLO 2D 검출 실험, 이전 3D 모델들(순환 보정 모델 → 이벤트 모델 → HandLite v1)과 쓰지 않은 방법은 [docs/archive/HISTORY.md](docs/archive/HISTORY.md)에 정리했습니다.
