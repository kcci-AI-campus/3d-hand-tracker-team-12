# hand_lite — HandLite C++ 런타임

HandLite 배포용 C++17 라이브러리입니다. `training/export.py`가 HandLite 체크포인트로 만든 신경망 3개(encoder·calibrator·decoder)를 **ncnn C++ API**로 실행하고, 그 주변의 이벤트·기하 로직을 C++로 구현했습니다. 원본은 [`hand_tracking/lite_runtime.py`](../../hand_tracking/lite_runtime.py)이고, 같은 결과를 내는지 기준 데이터(golden)로 검증합니다.

- **라이브러리 본체**(`src/geometry.cpp`, `src/runtime.cpp`)는 표준 라이브러리만 씁니다.
- **ncnn 연결부**(`NcnnGraphs`)만 ncnn이 필요합니다. `-DHAND_LITE_WITH_NCNN=ON`일 때 빌드됩니다.
- 신경망 실행부는 `hand_lite::Graphs` 인터페이스로 분리되어 있어, 다른 추론 엔진으로 바꿀 수 있습니다.

## 사용

```cpp
#include "hand_lite/hand_lite.hpp"
#include "hand_lite/ncnn_graphs.hpp"

auto graphs = std::make_shared<hand_lite::NcnnGraphs>("deploy");   // training.export 출력 폴더
hand_lite::Runtime runtime(hand_lite::load_config("deploy"), graphs);

// 카메라 프레임이 도착할 때마다: features [2][21][14] float, valid [2][21] (0/1)
runtime.push(camera, features, valid, capture_time, arrival_time);

// 원하는 시각에 (최신 도착 이후): [2][21][3] 월드 단위 (현재 데이터 1 = 40cm)
std::array<double, 126> pose = runtime.query(time);
```

- **입력**은 학습 데이터의 이벤트 특징과 같습니다. 채널은 u,v(0–1), ray 원점(2–4), 방향(5–7), 촬영→도착 지연 초(10)이고 나머지는 쓰지 않습니다. 시각은 절대 초(double)입니다.
- **예외**:
  - 카메라 번호가 0–2 밖이면 `std::invalid_argument`
  - 도착 순서가 거꾸로이거나, 최신 도착보다 이른 시각에 query하면 `std::logic_error`
  - 같은 카메라에서 더 옛 촬영이 늦게 도착하면 예외 없이 `push`가 `false`를 돌려주고 무시합니다.
- **threads/fp16**: `NcnnGraphs(dir, threads=1, fp16=true)`입니다. 그래프가 작아 1스레드가 가장 빨랐습니다. ARM에서 fp16 정확도가 문제면 `fp16=false`로 끄세요.
- 스레드 안전하지 않습니다. `Runtime` 하나는 한 스레드에서만 쓰세요.

## 빌드 (라즈베리 파이 등 Linux)

```bash
# 1) ncnn (한 번): Vulkan 끄고 설치
git clone --depth 1 https://github.com/Tencent/ncnn && cd ncnn
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_BENCHMARK=OFF -DCMAKE_INSTALL_PREFIX=$HOME/ncnn-install
cmake --build build -j4 && cmake --install build

# 2) hand_lite + 검증
cmake -S cpp/hand_lite -B build-hl -DCMAKE_BUILD_TYPE=Release \
      -DHAND_LITE_WITH_NCNN=ON -Dncnn_DIR=$HOME/ncnn-install/lib/cmake/ncnn
cmake --build build-hl -j4
ctest --test-dir build-hl --output-on-failure
```

`ctest`는 두 가지를 검사합니다.

| 테스트 | 내용 | 허용 오차 |
|---|---|---|
| `golden_replay` | 기록된 신경망 출력을 되돌려주며, C++가 만든 **모든 신경망 입력**, push 결과, 예외 종류, pose, 보정값을 Python과 비교 | pose 1e-6, 입력 1e-4 |
| `golden_ncnn` | 같은 시나리오를 **실제 ncnn 그래프**로 실행해 pose 비교 | 2e-3 (ncnn fp16/GELU 근사) |

ncnn 없이 `golden_replay`만 빌드하려면 `-DHAND_LITE_WITH_NCNN=OFF`(기본)로 두면 됩니다.

## 기준 데이터 (`tests/data`)

`tools/make_golden.py`가 만듭니다. 폭 16인 작은 무작위 HandLite를 ncnn으로 내보내고, Python `LiteRuntime`에 다음 사건이 담긴 시나리오를 넣으면서 모든 호출을 기록합니다.

- 3대 카메라, 17.1fps, 무작위 전송 지연
- 관절 누락, 아무것도 검출되지 않은 프레임
- 늦게 도착한 옛 프레임
- 잘못된 카메라 번호, 도착 순서 위반, 최신 도착보다 이른 query
- 카메라 2대가 0.4초 끊김(표본 유지 규칙이 쓰이는 구간)
- 1.5초 전체 끊김(슬롯이 없는 query) 후 복귀

```bash
python cpp/hand_lite/tools/make_golden.py     # 저장소 루트에서, requirements-transformer/export 필요
```

기준 모델은 현재 기본 구조(손가락 토큰·간격 임베딩·ray 깊이 anchor, export 형식 4)입니다. `load_config`는 형식 2·3 export도 읽고, 없는 키(`finger_tokens`, `anchor_ray_depth`)는 꺼진 것으로 봅니다. 시나리오의 카메라 끊김·관절 누락 때문에 decoder 호출 53번 중 40번에 ray anchor가 들어 있습니다. 카메라 1이 0.15초 동안 손 좌우를 바꿔 보고하는 구간도 있고, 기준 모델은 기본으로 꺼진 이상치 ray 제거(`ray_outlier_ratio=6`)를 켜서 제거·모호 판정 경로를 거칩니다. `ray_outlier_ratio` 키가 없는 export는 필터를 끈 것으로 읽습니다.

Python 쪽 기하·런타임을 바꾸면 기준 데이터를 다시 만들어야 합니다. 저장소 테스트 `tests/test_cpp_golden.py`는 기준 데이터가 현재 Python 런타임과 여전히 일치하는지 검사합니다. 일치하지 않으면 실패하므로 재생성이 필요한 때를 알 수 있습니다.

검증 강도 확인: 반복 표본 중복 제거, 보정 증거 판정, 빈 검출 프레임 처리, 표본 유지 시간, 슬롯 순서를 각각 일부러 틀리게 바꾸면 `golden_replay`가 모두 실패합니다. 이벤트 정리 시점 변경은 출력에 영향이 없어 통과하는 것이 맞습니다.
