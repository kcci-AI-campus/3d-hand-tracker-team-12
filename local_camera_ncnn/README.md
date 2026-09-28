# local_camera — Windows / ncnn CPU

`camera/local_camera.py`의 로컬 웹캠 화면을 **C++ + ncnn + OpenCV**로 구현했습니다.
앱 소스는 `local_camera.cpp` 하나입니다. **MediaPipe 런타임, Bazel, TensorFlow,
Python 실행 환경은 필요 없습니다.** ncnn은 정적으로 링크하며, OpenCV가 카메라와
영상 처리를 담당합니다.

기본값: USB 카메라 0, 320×240, 최대 2개 손, 처리 상한 15 FPS.
관절 21개와 연결선, 실제 갱신 FPS, 손 개수, 검출·관절 추론을 합친 시간을 표시합니다.
Q / Esc / 창 닫기 / Ctrl+C로 종료합니다.

## 실행

제공된 ZIP을 풀고 `run_camera.cmd`를 더블클릭하면 됩니다.
Windows x64 실행 파일과 모델은 `dist/`에 들어 있습니다.
이 실행 파일은 ncnn·OpenCV·런타임을 정적으로 링크해 별도 DLL 설치가 필요 없습니다.

```powershell
.\dist\local_camera.exe
.\dist\local_camera.exe --camera 1 --hands 1 --fps 10
.\dist\local_camera.exe --check-model
.\dist\local_camera.exe --image .\testdata\hand.jpg --output result.png
```

`--check-model`은 카메라/화면 없이 **두 모델 모두 실제 추론**합니다.
`--image`도 화면을 열지 않고, 관절 좌표를 출력하고 선택적으로 결과 이미지를 저장합니다.
모델 위치는 작업 디렉터리가 아닌 실행 파일 옆을 기준으로 찾습니다.
별도 위치는 `--models 경로`로 지정합니다.

## Windows에서 소스 빌드

Visual Studio 2022의 **Desktop development with C++** 도구와 CMake 3.24 이상을
설치한 뒤 Developer PowerShell에서 실행합니다. 빌드 도구 설치는 자동으로 수행하지 않습니다.

### OpenCV가 설치되어 있는 경우

[공식 OpenCV Windows 배포판](https://opencv.org/releases/)을 풀어 사용합니다.

```powershell
cd local_camera_ncnn
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1 -OpenCV_DIR C:\opencv\build
.\build-windows\bin\local_camera.exe
```

이 경로는 **ncnn만 소스 빌드**하고 OpenCV는 이미 빌드된 라이브러리를 사용합니다.
스크립트가 공식 OpenCV 배포판의 x64 DLL을 실행 폴더로 복사합니다.
사용자 지정 OpenCV 설치라면 필요한 DLL을 PATH 또는 실행 폴더에 직접 배치하세요.

### OpenCV 설치 없이 필요한 모듈만 같이 빌드

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1 -BundledOpenCV
.\build-windows\bin\local_camera.exe
```

ncnn 20250503과 OpenCV 4.10.0 소스를 해시 검증 후 받아 빌드합니다.
OpenCV는 `core`, `imgproc`, `imgcodecs`, `highgui`, `videoio`만 빌드합니다.
첫 빌드는 시간이 걸리지만 재빌드는 캐시를 사용합니다. 전체 MediaPipe를 받거나
컴파일하는 과정은 없습니다. 모델 파일은 포함되어 있으며 빌드 시 해시를 확인합니다.
빠진 모델은 `scripts/download_models.ps1`이 고정된 원본에서 복구합니다.

기존 ncnn 설치도 사용할 수 있습니다:

```powershell
cmake -S . -B build-custom -A x64 -DUSE_SYSTEM_NCNN=ON -Dncnn_DIR=C:\ncnn\lib\cmake\ncnn -DOpenCV_DIR=C:\opencv\build
cmake --build build-custom --config Release --target local_camera --parallel 4
ctest --test-dir build-custom -C Release --output-on-failure
```

## 모델과 처리 방식

1. 192×192 RGB / 0..1, 종횡비 유지·검은 여백 → palm-lite 모델
2. 2,016개 SSD 앵커 디코딩, 손바닥 신뢰도 0.5, IoU 0.3 NMS
3. 손목·중지 밑마디 방향으로 회전 보정, 손 영역 확대 → 224×224 crop
4. hand-lite 모델의 63개 값 → 원본 영상의 21개 관절 좌표로 복원

모델은 커뮤니티에서 **이미 ncnn으로 변환한 MediaPipe 계열 lite 모델**입니다.
이전 `hand_landmarker.task` 파일을 여기서 변환한 것이 아니며, 이전 full 모델과
같은 결과나 정확도를 보장하지 않습니다. 출처·버전·라이선스는 [THIRD_PARTY.md](THIRD_PARTY.md)에 있습니다.

**이 모델의 `score`는 손 존재 확률이 아니라 좌우 손 분류값입니다.** 이를 검출
신뢰도로 사용하지 않습니다. 이 변환본에는 별도 hand-presence 출력이 없으므로
손바닥 검출의 오검출을 관절 모델의 존재 확률로 다시 거르는 기능은 없습니다.

## 원본과 차이·제한

- 매 프레임 손바닥부터 검출합니다. MediaPipe VIDEO 모드의 프레임 간 추적·검출 생략·
  스무딩을 재현하지 않으므로 속도와 흔들림 특성이 다릅니다.
- 캡처 → 검출 → 관절 → 표시의 단일 루프입니다. 카메라 버퍼 크기 1은 드라이버에
  대한 요청이며, 최신 프레임 보장이나 읽기 타임아웃을 제공하지 않습니다.
- `--fps`는 처리 상한입니다. 표시된 추론 시간에는 전후처리도 포함됩니다.
- X/Y는 이미지 픽셀 좌표, Z는 모델이 추정한 상대 깊이를 이미지 픽셀 척도로 환산한
  값입니다. **미터 단위 world landmarks가 아닙니다.** 좌우 손 분류는 표시하지 않습니다.
- Windows 카메라는 DirectShow USB 웹캠을 사용합니다. CSI/Picamera2는 대상이 아닙니다.
- `--threads`는 ncnn이 OpenMP 지원으로 빌드되었을 때 CPU 병렬도에 적용됩니다.
- 카메라 종료가 장치 읽기 완료까지 지연될 수 있습니다.

## 검증

`ctest`는 빈 화면의 손바닥 추론, 직접 호출한 관절 추론, 실제 손 이미지에서
1개 손 검출, 잘못된 FPS 입력 거부를 검사합니다. 실제 웹캠의 인식 품질과 FPS는
PC와 카메라 환경에서 별도로 확인해야 합니다.

**Windows x64 빌드와 이미지 추론은 확인했습니다. 실제 웹캠·GUI 실행은 미검증입니다.**
회전·반전·두 손 등 추가 검사 결과는 [VALIDATION.md](VALIDATION.md)에 기록했습니다.
