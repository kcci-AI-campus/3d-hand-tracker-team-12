# HandLite 전체 검토 — 2026-09-23

검토 대상: `2dd6fe3` 기반 현재 작업 트리의 HandLite, LiteStream, NumPy 런타임, ONNX/ncnn 내보내기, 공용 데이터·학습·평가·체크포인트 코드, Colab 노트북 및 관련 테스트. 검토 중 다른 작업에서 추가된 상수 통합·빈 슬롯 처리·테스트 변경도 확인했다. 모델 구현은 수정하지 않았다.

## 판단

기하 기반 anchor 위에 작은 학습 잔차를 얹는 구조는 경량 배포 목적에 맞는다. 기본 파라미터 수 191,469개를 직접 확인했고, 학습·추론·변환의 기본 경로는 테스트를 통과했다. 다만 아래 입력 소유권 및 내보내기 검증 결함을 수정하기 전에는 실시간 배포의 신뢰성을 보장하기 어렵다. 기존 테스트 통과는 정확도 개선이나 Raspberry Pi 성능을 입증하지 않는다.

## 확인된 결함

### 1. [P2] 재사용하는 valid 배열이 과거 이벤트를 변경한다

위치: `hand_tracking/lite_runtime.py:231` (`valid = np.asarray(valid, bool)`), `_Event`에 저장하고 다음 push에서 `event.valid`를 사용하는 부분.

입력이 이미 bool NumPy 배열이면 복사가 발생하지 않는다. 카메라 루프가 같은 마스크 버퍼를 다음 프레임에 재사용하면 보관 중인 이전 카메라의 검출 상태가 바뀐다. raw는 np.where로 복사했지만 valid는 외부 배열을 계속 참조하므로, 기존 토큰과 이후 삼각측량에 사용하는 마스크가 불일치한다. PyTorch EventStream은 clone하므로 배포 런타임과도 동작이 달라진다.

재현: camera 0의 valid를 모두 True로 push한 뒤 원본 배열을 False로 바꾸고 camera 1을 push했다. 입력을 복사한 대조군과 비교하면 최신 표본의 유효 ray 수가 84에서 42로 줄고 pose 최대 차이가 0.13475994 월드 단위 발생했다. 임의의 비영 잔차 가중치를 사용한 합성 시험이며 실제 정확도 수치는 아니다.

권장 수정: `np.array(valid, dtype=bool, copy=True)`로 보관 입력을 소유하고, NumPy 재사용 버퍼 회귀 시험을 LiteRuntime에도 추가한다.

### 2. [P2] 배포 검증이 NaN 출력을 오차 0으로 통과시킨다

위치: `export_lite.py:49` (`worst[name] = max(worst[name], ...)`). `tests/test_lite.py`의 런타임 비교도 같은 누적 패턴을 사용한다.

배포 출력이나 PyTorch 기준 출력에 NaN이 있으면 차이의 max도 NaN이다. Python의 `max(0., nan)`은 0을 반환하므로 최종 tolerance 검사가 실패하지 않는다. 즉 잘못 변환된 모델도 정상 검증 결과처럼 보고될 수 있다.

재현: LiteRuntime을 모든 관절에 NaN을 반환하는 시험용 객체로 교체하고 실제 compare_on_clip 경로를 실행했더니 `{'onnx': 0.0}`을 반환했다. 이는 현재 정상 백엔드가 NaN을 출력했다는 뜻이 아니라 검증기의 결함을 주입 시험으로 확인한 것이다.

권장 수정: 기준 출력·배포 출력·오차에 명시적인 isfinite 검사를 하고 즉시 실패시킨다. 실제 비교 query 수가 0인 경우도 실패시켜야 한다. `--queries 0` 역시 현재 빈 비교로 0 오차를 반환할 수 있으므로 양수 검증이 필요하다.

### 3. [P2] 긴 입력 중단 후 패딩 슬롯이 attention에 다시 참여한다

위치: `hand_tracking/lite.py:234` 및 Decoder의 시간 bias와 token bias를 합하는 부분(현재 140행).

없는 슬롯은 인덱스 0으로 대체되어 첫 이벤트 토큰과 capture를 가져온다. 유효하지 않아도 gap은 그대로 시간 MLP에 입력되고, gap 상한도 없다. 학습된 시간 bias가 고정 마스크 -10000보다 커지면 제외되어야 할 토큰이 attention에 참여한다. NumPy 런타임은 빈 슬롯을 zero token·gap 0으로 만들기 때문에 PyTorch와도 달라진다.

재현: seed 42, 폭 32·1 block·보정 head 없음, 비영 잔차 head인 모델로 동일 이벤트를 입력했다. query 1/1000/10000초에서는 두 구현 차이가 0이었으나 100000초에서는 최대 0.09773460 월드 단위였다. 장애 임계 시간은 가중치에 의존한다. 일반적인 짧은 중단에서 재현된 문제는 아니며, 장시간 실행 후 카메라 단절에 대한 경계 조건이다.

권장 수정: 미사용 슬롯의 token과 gap을 PyTorch에서도 명시적으로 0으로 만들고, 시간 bias를 적용한 뒤 미사용 key의 점수를 고정 마스크 값으로 덮어쓴다. ONNX/ncnn 대응과 장시간 단절 후 재접속 회귀 시험을 함께 확인한다.

### 4. [P3] HandLite 평가 결과가 transformer로 기록된다

위치: `evaluate_hand_transformer.py:61`.

체크포인트가 HandLite여도 보고서에 `method='transformer'`를 하드코딩한다. MPJPE 자체는 계산되지만, 결과를 method별 집계하면 두 구조를 잘못 분류하게 된다. 현재 Lite CLI 시험은 MPJPE 유한성만 검사해 이를 놓친다.

권장 수정: `architecture_of(checkpoint.model)` 또는 체크포인트 architecture를 사용하고 Lite 평가 JSON의 method도 검증한다.

## 구조·학습에 대한 평가

| 영역 | 확인 내용 | 판단 및 후속 확인 |
|---|---|---|
| 이벤트 인코딩 | 21관절 × 14특징을 손별 64차원 토큰으로 압축, 카메라·손 임베딩 사용 | 모델 크기를 줄이는 명확한 설계. 개별 관절의 미세 정보를 압축하므로 가림·손가락 끝 성능을 별도 평가할 필요가 있다. |
| 슬롯 | 카메라별 최근 8개, capture age 0.5초 이내 | 카메라별 할당은 유리하다. 다만 fps가 높아지면 실제 시간 범위가 짧아진다. 8개는 60fps에서 대략 0.12~0.13초이므로 시간 단위 설정만으로 fps 변화에 완전히 불변하지 않다. |
| 기하 | 최신 ray, 보정 후 삼각측량, 최근 표본 선형 외삽, hold와 손 평균 fallback | 신경망이 절대 위치 전체를 처음부터 학습할 필요가 없다. 비동기 관측과 빠른 비선형 운동의 오차는 여전히 데이터별 검증이 필요하다. |
| 카메라 보정 | 슬롯 토큰을 카메라별 pooling, query당 3×6 보정, 기하 증거 없으면 nominal | 장기 누적 drift가 없는 구조다. calibrator에는 슬롯의 명시적 시간 gap이 없으므로 촬영 간격을 활용하는 능력은 decoder보다 제한적이다. 이는 설계 제약이며 재현된 구현 결함으로 분류하지 않았다. |
| 손실·학습 | pose SmoothL1 + bone loss + 선택적 calibration 보조 손실, 잔차 head 0 초기화 | 초기에는 anchor로 시작하며 보정 head gradient와 학습 경로를 기존 시험이 검증한다. 긴 본학습 수렴성과 실제 장치의 입력 분포는 이번 검토로 검증되지 않았다. |
| 데이터 | 참여자·원본 train/val 분리 검사, 프레임 이벤트 복원, 최대 128 이벤트 | query-grid NPZ에서 두 query 사이에 대체된 프레임은 복원할 수 없다는 제약이 코드에 명시돼 있다. 실시간 원본 이벤트와 분포 차이를 별도 측정해야 한다. |
| 배포 | 고정 shape 신경망 3개 + Python/NumPy 기하 | 네트워크 경량화와 전체 런타임 경량화는 별개다. 슬롯 선택·캐시·기하까지 포함한 Pi 실측이 필요하다. |

## 검증 결과와 한계

- `.venv-transformer/Scripts/python.exe -m unittest discover -s tests -v`: **72개 모두 통과**, 51.768초. 합성 데이터 학습·재개·평가·예측 CLI, batch/stream, geometry, ONNX 및 ncnn 런타임 비교 포함.
- 별도 보정 head 비활성화 모델의 ONNX/ncnn 내보내기·로드·1회 추론은 두 백엔드 모두 유한한 결과를 반환했다.
- 추가 재현 코드: `.tools/review_handlite.py`. 결과: `.tools/review_handlite.log`. 원본 모델 파일을 변경하지 않고 버퍼 재사용, 긴 단절, NaN 검증 결함을 확인했다.
- 노트북의 내장 소스 21개를 현재 파일과 대조: 차이 없음(비교 시점, 바깥쪽 공백 제외).
- 문서의 예비 MPJPE HandLite 63.93mm / Transformer 62.86mm / baseline 64.42mm는 이번에 본학습으로 재측정하지 않았다. 문서 자체도 짧은 개발용 학습으로 한정한다. 수치만으로 정확도 우위를 주장할 수 없다.
- Raspberry Pi·CUDA mixed precision·실제 카메라 입력을 이번에 실행하지 않았다. CPU에서 통과한 결과를 해당 환경의 성능 보장으로 확장해서는 안 된다.

## 권장 순서

1. 입력 버퍼 소유권과 NaN 검증을 먼저 고친다.
2. 패딩 슬롯 시간 처리와 평가 method를 고치고, 관련 회귀 시험 및 노트북을 갱신한다.
3. 같은 데이터·seed·학습 예산으로 baseline/Transformer/Lite를 비교한다. 빠른 운동, 손별 가림, 카메라 누락, fps·지연 변화별 지표도 분리한다.
4. Pi에서 push+query 전체 지연의 p50/p95/p99, 메모리, 장시간 단절·재접속, fp16 정확도를 측정한다.

최신 동시 변경 반영 후 재확인: HandLite 전용 테스트 7개 모두 통과(9.074초). 로그: .tools/review_handlite_tests.log.
