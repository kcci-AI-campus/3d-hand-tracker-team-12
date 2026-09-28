# 두 손 3D 포즈 모델

비동기로 도착하는 카메라 3대의 2D 손 관절(이벤트)을 받아, 원하는 시각의 두 손 3D 관절 42개를 출력합니다. 모델은 두 가지입니다.

| | **HandLiteV3** (`--arch litev3`, 기본) | HandDirect (`--arch direct`, 비교용) |
|---|---|---|
| 방식 | 직선 맞춤 삼각측량 기준점 + 신경망 보정 | 신경망만으로 좌표를 **직접** 출력 (손목 기준 분해 + 단계적 보정) |
| 기하 계산 | ray 직선 맞춤, 삼각측량, 직전 위치로 실패 관절 기준점 | 없음 |
| 신경망 | encoder, corrector 2개 | encoder, query(fusion + decoder) 2개 |
| 추가 출력 | 관절별 예상 오차(mm), 손별 '시야 안' 확률, 기준점 종류 | 없음 |
| 파라미터 (폭 64) | 133,797 | 202,092 |
| 필요 이력 | 1.45초 | 0.75초 |
| 런타임 | `runtime.py` (NumPy + ncnn), C++ `hand_tracker_master` | `direct_runtime.py` (NumPy + ncnn), C++ `hand_tracker_master` |
| 자세한 구조 | [HANDLITEV3_ARCHITECTURE.md](HANDLITEV3_ARCHITECTURE.md) | [HANDDIRECT_ARCHITECTURE.md](HANDDIRECT_ARCHITECTURE.md) |

**결과 (`gigahands_pi3_overlap`, 검증 데이터):** HandLiteV3는 TPU v6e-1에서 85 epoch 학습해 val MPJPE 20.36mm(평가 20.21mm, PCK@20mm 69.1%, 손목 기준 9.09mm, 시야 판정 99.0%)입니다(실행 `event_litev3_state_tpu_e85_v3`, 2026-09-28). HandDirect-wrist(손목 기준 분해 + 단계적 보정, fusion 2)는 GPU에서 60 epoch 학습해 best val 22.8mm입니다(`gigahands_colab_wrist_refine.ipynb`, 시야 밖 손 포함, 참 월드 좌표 손실 0.1배). 같은 검증 창·관절에서 비교하면 짧은 클립은 HandLiteV3가 약 2.3mm 앞서고, 창 300개 이상 장편 클립에서는 차이가 0.7mm로 줄어듭니다. 마스터 파이 앱([hand_tracker_master](../hand_tracker_master/README.md))은 HandDirect-wrist를 씁니다.

HandLite v1(최신 두 프레임 외삽 + 카메라 보정 신경망)은 HandLiteV3로 대체되어 삭제했습니다. 그 C++ 런타임 `cpp/hand_lite`와 문서 [archive/HANDLITE_V1_ARCHITECTURE.md](archive/HANDLITE_V1_ARCHITECTURE.md)는 HandLiteV3 C++ 이식의 참고용으로 남겼습니다.

## 공통: 입력과 데이터

- **이벤트** = 카메라 한 대의 프레임 하나. 관절 특징 `[2손, 21관절, 14채널]`, 검출 여부 `[2,21]`, 카메라 번호, 촬영·도착 시각(초)입니다. 채널: u,v(0–1) 0–1, ray 원점 2–4, ray 방향 5–7, 촬영→도착 지연 10. 나머지 채널은 모델이 쓰지 않습니다.
- ray는 2D 관절을 카메라 내부 파라미터와 **정해 둔 배치(공칭 외부 파라미터)**로 바꾼 것입니다. 지금은 시뮬레이터가 계산해 NPZ에 넣습니다. 실제 카메라(MediaPipe)와의 연결은 아직 없습니다.
- 같은 카메라에서 더 옛 촬영 프레임이 늦게 도착하면 버립니다(`events.accepted_captures`).
- **슬롯**: query 시각 t까지 도착했고 촬영이 t − 0.5초 이후인 이벤트 중 카메라마다 최근 8개(`events.select_slots`). 두 모델 모두 이 슬롯의 손가락 토큰 240개를 신경망 입력으로 씁니다. 데이터의 카메라 속도는 약 11.7fps라 0.5초 안에 카메라당 약 6프레임입니다.
- **학습 윈도**(`data.make_sample`): 연속 query 16개와, 첫 query보다 `context_s` 전부터 마지막 query까지 도착한 이벤트(최대 `--max-events`개). 시각은 마지막 query 기준 상대 초입니다. `--train-queries N`(기본 4)이면 학습은 윈도의 마지막 N개 query만 씁니다(검증은 전부).
- **증강 없음**: 카메라 배치가 고정이므로, 그 배치의 월드 좌표를 모델이 그대로 배우는 편이 유리합니다. 배치의 작은 흔들림(설치 오차)은 데이터 생성기가 클립마다 따로 넣습니다.
- **정답 좌표계**(`--target-frame`, 기본 `rig`): 카메라 영상으로는 카메라끼리의 상대 배치만 알 수 있어, 클립마다 "정해 둔 배치를 실제로 놓인 대로" 본 좌표계로 정답을 옮깁니다(`data.rig_frame`). 참 월드 좌표 오차는 `mpjpe_world_mm`으로 따로 기록합니다(손실에는 없음).
- **시야 밖 손**(`data.hands_in_view`): 정답 3D 관절을 실제 카메라 자세로 투영해, 관절의 절반 이상이 어느 한 카메라 화면 안에 들어오면 '시야 안'입니다. 시야 밖 손은 위치 정답이 없고(`--keep-out-of-view`로 끔), HandLiteV3는 대신 '시야 안' 여부를 학습합니다. 위치 지표는 시야 안 손 기준입니다.
- 참가자·원본 클립이 split 사이에 겹치지 않는지 시작할 때 검사합니다.

## 학습

```powershell
.\.venv-transformer\Scripts\python.exe -m training.train --data exports/gigahands_pi3_overlap --output runs/litev3 --cache-dir runs/cache
.\.venv-transformer\Scripts\python.exe -m training.train --arch direct --data exports/gigahands_pi3_overlap --output runs/direct --cache-dir runs/cache
```

- 기본값은 TPU 본학습 설정입니다: 85 epoch, 배치 128, lr 8.5e-4, `--train-queries 4`. GPU·CPU에서 배치를 줄이면 lr을 배치의 제곱근에 비례해 줄이세요(배치 64에 6e-4).
- `--device xla`는 TPU(PyTorch/XLA)입니다. float32로 학습하고, 배치 shape를 고정하며(`--max-events`로 패딩, 학습 epoch의 마지막 불완전 배치는 버림), 첫 배치는 컴파일로 느립니다(`hand_tracking/accelerator.py`). CUDA에서는 mixed precision을 쓰고 fp16 오버플로 스텝은 건너뜁니다(50번 연속이면 멈춤).
- Colab(TPU)은 [gigahands_colab.ipynb](../notebooks/gigahands_colab.ipynb)를 씁니다(`ARCH='litev3'` 기본). 모델 코드를 바꾼 뒤에는 `python -m training.build_colab_notebook --dataset <데이터셋>`으로 다시 만듭니다. 소스가 바뀌면 이전 실행은 재개되지 않으니 새 실행 이름을 쓰세요.
- 손실: 관절 위치 SmoothL1(미터 단위, β=6mm) + 0.1 × 뼈 길이, 손목 기준 상대 좌표 × `--relative-weight`(1.0). HandLiteV3는 예상 오차 × `--error-weight`(0.1)와 시야 판정 × `--presence-weight`(0.1)가 더해지며, 두 head는 토큰 gradient를 끊어 포즈 학습을 바꾸지 않습니다.
- 모델 설정의 모든 필드가 같은 이름의 옵션입니다(목록은 `--arch <이름> --help`).
- 데이터 준비가 학습 속도를 정하므로 `--workers`와 `--cache-dir`(첫 epoch에 전처리한 클립을 압축 없이 저장, 데이터의 약 2배 용량)를 쓰세요.
- `best.pt`는 검증 MPJPE가 가장 낮은 epoch, `last.pt`는 마지막 완료 epoch입니다(CPU 텐서로 저장). 같은 명령에 `--resume`을 붙이면 이어갑니다(속도 설정 `--workers`, `--threads`, `--cache-dir`, `--prefetch`는 달라도 됨).

## 평가·추론·배포

```powershell
.\.venv-transformer\Scripts\python.exe -m training.evaluate --checkpoint runs/litev3/best.pt --data exports/gigahands_pi3_overlap --output runs/litev3/val.json
.\.venv-transformer\Scripts\python.exe -m training.predict --checkpoint runs/litev3/best.pt --input exports/gigahands_pi3_overlap/val/<clip>.npz --output runs/litev3/prediction.npz
.\.venv-transformer\Scripts\python.exe -m training.export --checkpoint runs/litev3/best.pt --output runs/litev3/deploy --check-input exports/gigahands_pi3_overlap/val/<clip>.npz
```

- 지표: 윈도 **마지막 query**의 MPJPE(mm), PCK@20mm, 손목 기준 MPJPE, 참 월드 좌표 MPJPE. HandLiteV3는 예상 오차의 빗나감(`error_miss_mm`), 시야 판정 정확도, 기준점 종류별 비율·오차(보정 전 기준점만의 오차 포함)도 기록합니다.
- `training.export`는 체크포인트의 모델에 맞춰 신경망을 **ncnn**으로 내보내고(pnnx), 검증 클립에서 NumPy 런타임과 PyTorch 스트림의 차이를 확인합니다(`--tolerance` 기본 2e-3). 결과 폴더를 장치로 복사하면 `numpy`와 `ncnn`만으로 추론합니다.

```python
from hand_tracking.runtime import LiteV3Runtime
runtime = LiteV3Runtime('deploy')
runtime.push(camera, features, valid, capture_time, arrival_time)   # 카메라 프레임 도착마다
pose = runtime.query(time)                                          # [2,21,3] 월드 단위
runtime.error_mm, runtime.in_view_probability, runtime.anchor_kind  # 관절별 예상 오차, 손별 시야 안 확률, 기준점 종류
```

HandDirect는 `hand_tracking.direct_runtime.DirectRuntime`을 같은 방식으로 씁니다(추가 출력 없음).

## 코드 구성 (`hand_tracking/`)

| 모듈 | 내용 |
|---|---|
| `constants.py`, `config.py`, `contracts.py` | 입력 채널·단위, 모델 설정(`LiteV3Config`, `DirectConfig`, `ARCHITECTURES`), 텐서 계약 |
| `data.py` | NPZ에서 이벤트 복원, 학습 윈도, rig 좌표계, 시야 판정, 전처리 캐시 |
| `events.py` | 이벤트 수락, 슬롯 선택 |
| `geometry.py` | 최소제곱 삼각측량, ray와 점의 각도 잔차 |
| `layers.py`, `networks.py` | 공통 신경망 부품: attention 블록, 손가락 토큰 encoder, encoder 입력 |
| `model.py`, `deploy.py`, `runtime.py` | HandLiteV3 모델·스트림, ncnn 내보내기, NumPy 런타임 |
| `direct.py`, `direct_export.py`, `direct_runtime.py` | HandDirect 모델·스트림, ncnn 내보내기, NumPy 런타임 |
| `stream.py` | 이벤트별 스트리밍 공통 부분(순서 검사, 오래된 이벤트 정리, 인코딩 캐시) |
| `objectives.py`, `engine.py`, `checkpoints.py`, `accelerator.py` | 손실·지표, 학습/검증 루프, 체크포인트, TPU(XLA) 지원 |

실행 명령은 `training/`(`train`, `evaluate`, `predict`, `export`, `build_colab_notebook`)에 있고, 저장소 루트에서 `python -m training.<이름>`으로 실행합니다. 테스트는 `python -m unittest discover -s tests`입니다.

## 상태

- **HandLiteV3 스트림·런타임의 관절 상태**: 배포 런타임과 `LiteV3Stream`은 관절마다 마지막 삼각측량을 기억하지만, 과거 창 탐색(1.0초)이 직전 위치를 못 찾은 관절에만 씁니다(2026-09-28 변경). 그래서 매 프레임 query해도 입력이 학습(`forward()`)과 같습니다. 그 전에는 상태가 더 최근이면 항상 써서, 배포의 직전 위치 경과 시간(약 0.09초)이 학습에서 본 범위(약 0.2초 이상) 밖이었습니다. 이 변경 뒤 새로 학습한 결과는 아직 없습니다.
- ncnn 런타임에 float64 입력을 넘길 때 변환된 임시 배열이 먼저 해제되어 신경망이 쓰레기 값을 읽던 버그를 2026-09-28에 고쳤습니다(`runtime.NcnnGraphs.run`). 그 전에 내보낸 모델도 다시 내보낼 필요는 없고 런타임 코드만 바꾸면 됩니다.
- 두 모델의 C++ 런타임이 [hand_tracker_master](../hand_tracker_master/README.md)에 있습니다(`--arch direct|litev3`). Python 런타임과 golden 테스트로 비교합니다(HandLiteV3: 기준점 종류 모두 같음, 좌표 차이 4e-7 단위).
- 저장소의 HandDirect 학습에는 참 월드 좌표 손실이 없습니다(HandLiteV3와 같은 손실 구성). HandDirect-wrist를 학습한 노트북은 0.1배로 썼습니다.
- 실제 카메라에 쓰기 전에는 실제 3카메라 관측에서 손 좌우 대응, 관절 오차, 지연 분포를 따로 확인해야 합니다.
