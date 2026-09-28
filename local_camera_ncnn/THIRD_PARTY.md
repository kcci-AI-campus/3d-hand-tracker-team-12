# Third-party sources

- **ncnn 20250503**, Tencent, BSD 3-Clause and bundled dependency notices:
  https://github.com/Tencent/ncnn/tree/20250503 (`licenses/ncnn-LICENSE.txt`).
- **OpenCV 4.10.0**, Apache 2.0 and bundled codec licenses:
  https://github.com/opencv/opencv/tree/4.10.0 (`licenses/opencv-LICENSE`).
- **MediaPipe-derived palm-lite / hand-lite ncnn model files**, obtained from
  https://github.com/FeiGeChuanShu/ncnn-Android-mediapipe_hand/tree/65b1bcac06c5b9a1a492be3afebc9e4d52680f5c/app/src/main/assets
  Model conversion is supplied by that community repository; this project does
  not claim that these files were exported from the previous `.task` bundle or
  that they reproduce its outputs exactly. Original MediaPipe project:
  https://github.com/google-ai-edge/mediapipe (Apache 2.0, `licenses/mediapipe-LICENSE`).
  The conversion repository provides no separate model-specific license file.
- **Reference preprocessing / output contract**: `hand.cpp` and `landmark.cpp`
  in the above conversion repository. Those files carry the Tencent 2021
  BSD 3-Clause notice, retained in `licenses/reference-BSD-3-Clause.txt`.
  The implementation here simplifies the pipeline and validates shapes/errors;
  it does not include the Android camera or application code.
- **testdata/hand.jpg**: Qengineering/Hand-Pose-ncnn-Raspberry-Pi-4,
  commit `f220df2b12d6b54d2cc5fdcbddde816c092d12ef`, repository BSD 3-Clause
  (`licenses/sample-image-LICENSE`).

Models are distributed as `.param` + `.bin` pairs. SHA-256 values and the pinned
download origin are recorded in `scripts/download_models.ps1`.

The supplied Windows executable also statically links the MinGW-w64 / GCC
runtime and OpenCV's zlib, libpng, and libjpeg-turbo codecs. Their notices are in
`licenses/`. This software is based in part on the work of the Independent JPEG
Group. GCC runtime components are covered by GPLv3 with the GCC Runtime Library
Exception; those texts are included. The compiler/toolchain itself is not part
of the application distribution.
