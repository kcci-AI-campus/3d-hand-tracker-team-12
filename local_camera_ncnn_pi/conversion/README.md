# 공식 모델 변환 재현 — 개발 PC 전용

**파이에서 앱을 실행할 때는 이 절차가 필요 없습니다.** 검증된 ncnn 모델은
이미 `../models/`에 있습니다. 변환은 별도 개발 PC에서 수행합니다.
검증 환경: Windows x64, Python 3.12, PyTorch 2.14.0 CPU, pnnx 20260704,
LiteRT 2.2.0, ncnn 20260526 C++.

## 1. 공식 원본과 모델 구조

프로젝트 루트에서 실행합니다. 아래 명령의 Python은 새 가상환경 것을 사용하세요.

```bash
python -m venv .venv-convert
# Windows: .venv-convert\Scripts\activate
# Linux: source .venv-convert/bin/activate
python -m pip install -r conversion/requirements.txt
python conversion/download_official.py
python conversion/inspect_models.py
```

`download_official.py`는 공식 `latest` URL만 사용합니다. 다운로드 날짜, HTTP ETag,
SHA-256을 `official_source.json`에 기록합니다. 이후 Latest가 다른 파일로 바뀌면
기존에 검토한 해시와 달라 실패합니다. 구조·전후처리를 검토하고 해시를 갱신한 뒤
전체 검증을 다시 해야 합니다. 모델이 달라졌는데 조용히 덮어쓰지 않습니다.

## 2. ncnn 변환

공식 [pnnx 20260704 릴리스](https://github.com/pnnx/pnnx/releases/tag/20260704)에서
개발 PC에 맞는 ZIP을 받아 압축을 풉니다. Windows에서 검증한 ZIP과 실행 파일의
SHA-256은 `toolchain_releases.json`에 있습니다. 이 버전의 PyPI 패키지는 철회되어
GitHub 릴리스 실행 파일을 직접 사용합니다. Python requirements에 pnnx는 포함하지 않습니다.

```bash
python conversion/convert_models.py --pnnx /absolute/path/to/pnnx
```

Windows에서는 `--pnnx C:/tools/pnnx-20260704-windows/pnnx.exe`처럼 지정합니다.

`source/*.tflite`의 FlatBuffer에서 상수와 옵션을 직접 읽고 같은 PyTorch 그래프를
만듭니다. 먼저 LiteRT 원본과 PyTorch 출력을 비교하며, 통과한 그래프만 TorchScript와
pnnx로 변환합니다. `converted/`에 ncnn 파일 4개, `work/`에 중간 파일과 로그가
생깁니다. `--model hand_detector` 또는 `--model hand_landmarks_detector`로 한 모델만
다시 변환할 수도 있지만 최종 배포 전에는 두 모델을 모두 검증해야 합니다.

pnnx 20260704의 손바닥 출력 Reshape 네 개는 ncnn 20260526의 배치 없는
2차원 표현에 맞게 정규화합니다. [구조 설명](../MODEL_STRUCTURE.md)에 원인과
처리 방식이 있으며, 보정 내역도 변환 보고서에 기록합니다.

## 3. 실제 배포 버전의 ncnn C++로 검증

Python ncnn 패키지의 다른 버전에 의존하지 않도록 별도 작은 C++ probe를 사용합니다.
Visual Studio C++ 개발 도구와 CMake가 설치된 Windows 예:

```powershell
cmake -S conversion -B conversion/probe-build -A x64
cmake --build conversion/probe-build --config Release --target ncnn_probe --parallel 4
python conversion/validate_conversion.py --probe conversion/probe-build/bin/ncnn_probe.exe
python conversion/publish_models.py
```

Linux에서는 `-A x64` 대신 `-G Ninja -DCMAKE_BUILD_TYPE=Release`를 사용하고 실행 파일의
`.exe`를 생략합니다. 파이 프로젝트 루트에서도 `cmake --build build --target ncnn_probe`
로 같은 probe를 추가 빌드할 수 있습니다.

검증은 검정·흰색·난수·실제 손·회전·반전의 6종 입력에서 **모든 출력 텐서**를
원본 LiteRT와 비교합니다. `parity_report.json`에 비교 오차와 검증한 파일 해시를
저장합니다. `publish_models.py`는 검증 이후 모델이 변경되지 않았음을 확인한 뒤
`../models/`로 복사하고 SHA256SUMS·manifest를 생성하며 이전 lite 모델을 삭제합니다.

원본과 ncnn의 출력 허용 오차는 `atol=0.002, rtol=0.002`입니다. 전체 Tasks API의
추적·영상 샘플링까지 동일하다는 검사는 아니며 모델의 수치 동등성을 검사합니다.

## 4. 앱 전후처리 검사

앱을 빌드한 환경에서 실행합니다. Python의 OpenCV/NumPy는 검사 도구에만 필요합니다.

```bash
python conversion/validate_pipeline.py --app build/bin/local_camera
python conversion/validate_uv.py --app build/bin/local_camera
```

손 이미지의 회전·반전·두 손 합성·여백·빈 화면을 처리하고 예상 손 개수 및
변환 전후 관절 좌표 일관성을 확인합니다. 결과는 `image_checks.json`에 기록합니다.

## 포함된 근거 파일

- `source/`: 공식 `.task`와 추출한 두 원본 TFLite
- `official_source.json`: 다운로드 출처·날짜·해시
- `model_structure.json`: 실제 텐서·연산자 전체 목록
- `conversion_report.json`: TFLite ↔ PyTorch 비교
- `parity_report.json`: TFLite ↔ ncnn C++ 비교
- `image_checks.json`: C++ 앱 전후처리 검사
- `uv_checks.json`: 실제 추론 결과와 336바이트 UDP 좌표 비교 (좌우·두 손·미검출)
- `toolchain_releases.json`: ncnn/pnnx 공식 릴리스 URL과 다운로드 파일 해시

`work/`, `converted/`, `probe-build/`, 변환용 가상환경은 배포 ZIP에 포함하지 않습니다.
