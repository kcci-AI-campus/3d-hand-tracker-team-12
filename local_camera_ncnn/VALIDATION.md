# Validation — 2026-09-23

## Build

- Windows x64, w64devkit 2.10.0 (GCC 16.2.0), CMake 3.31.6, Ninja 1.12.1.
- ncnn 20250503, OpenCV 4.10.0; CPU only, static libraries, OpenMP enabled.
- SSE/AVX runtime dispatch enabled; AVX-512, XOP and newer AVX-VNNI variants disabled.
- OpenCV built with only core/imgproc/imgcodecs/highgui/videoio; DirectShow + Win32 UI.
- PE import inspection: Windows system DLLs only. No ncnn/OpenCV/GCC runtime DLL import.
- PowerShell script parsing and all four model SHA-256 checks passed.

Build uses the CMake project with `BUILD_BUNDLED_OPENCV=ON` and a Ninja/MinGW generator.
Visual Studio build instructions are provided, but that compiler was not installed
on the validation machine and was not tested.

## Automated runtime checks

All 3 CTest tests passed:

1. Blank-frame palm inference plus a direct call to the landmark model.
2. `testdata/hand.jpg`: one hand detected, 21 finite XYZ values.
3. Invalid `--fps 0` rejected with a nonzero exit code.

Additional integration checks used the same image resized to 480×480:

| Input | Expected / observed hands | Mean XY difference from transformed original |
|---|---:|---:|
| Original | 1 / 1 | — |
| 90° clockwise | 1 / 1 | 2.98 px |
| 180° | 1 / 1 | 1.27 px |
| Horizontal reflection | 1 / 1 | 4.10 px |
| Original + reflected copy side by side | 2 / 2 | — |
| Portrait padding | 1 / 1 | — |
| Landscape padding | 1 / 1 | — |
| Black frame | 0 / 0 | — |
| White frame | 0 / 0 | — |

These are transformation consistency checks, not accuracy measurements against
human-labelled ground truth. Rendered normal and two-hand results were visually
inspected. `testdata/result_two_hands.png` is the inspected example.

An incompatibility was found and corrected during testing: the converted hand
model's Squeeze/Gemm head produced a 63×8 output with packed x86 activations.
Disabling channel packing for **the landmark net only** restored the expected 63
scalars. The app validates output shapes so this cannot silently draw bad data.
The model's `score` output represents handedness and is deliberately not used as
a hand-presence threshold.

## Not validated

- Physical webcam capture, interactive GUI lifecycle, real-time latency/FPS.
- Robustness on a broad dataset, occlusions, complex backgrounds or fast motion.
- Equivalence to the earlier MediaPipe Tasks full model or its temporal tracker.
- Raspberry Pi/ARM, GPU/Vulkan and Visual Studio compilation.
