# Sources and licenses

- **Models**: Google's official Hand Landmarker (full), downloaded from
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  on 2026-09-23. Original bundle and TFLite models are in `conversion/source/`.
  MediaPipe Apache 2.0 notice: `licenses/mediapipe-LICENSE`.
  No community-converted lite model files are used in this project anymore.
- **Conversion**: the included script reads the official FlatBuffers and copies
  the actual constants into an equivalent PyTorch graph, then uses Tencent
  pnnx to emit ncnn models. No training or replacement model is involved.
  Tool versions, input/output hashes and test results are recorded in the manifests.
- **ncnn 20260526**: https://github.com/Tencent/ncnn/tree/20260526
  BSD 3-Clause and dependency notices in `licenses/ncnn-LICENSE.txt`.
- **pnnx 20260704**: https://github.com/pnnx/pnnx/releases/tag/20260704
  Official GitHub executable used for conversion; download hashes in
  `conversion/toolchain_releases.json`. Not required on the Raspberry Pi.
- **OpenCV**: linked from the Raspberry Pi OS package. Reference Apache 2.0
  notice in `licenses/opencv-LICENSE`; installed package supplies its applicable notices.
- **Earlier preprocessing reference**: the original application implementation
  consulted Tencent-BSD-labelled `hand.cpp` / `landmark.cpp` from
  FeiGeChuanShu/ncnn-Android-mediapipe_hand. Its notice is retained in
  `licenses/reference-BSD-3-Clause.txt`. It is not the source of the new models.
- **testdata/hand.jpg**: Qengineering/Hand-Pose-ncnn-Raspberry-Pi-4, commit
  `f220df2b12d6b54d2cc5fdcbddde816c092d12ef`, repository BSD 3-Clause notice in
  `licenses/sample-image-LICENSE`. The validation preview is derived from this image.

Conversion tools are development dependencies only and are not distributed as
executables in this source package. No Windows validation executable or prebuilt
ARM binary is included.
