# 손 쥐기·펴기 애니메이션 샘플

Blender 4.5.14 LTS에서 제작한 **손 펴기 → 주먹 쥐기 → 유지 → 다시 펴기** 반복 애니메이션입니다.

## 결과 파일

- `hand_open_close.blend`: 손 메시, 손톱, 21개 뼈, 관절 제한, 편집 가능한 키프레임, 미리보기 카메라.
- `hand_open_close.glb`: 같은 모델과 `Hand_Open_Close_Loop` 애니메이션 1개. 관절 제한 결과를 프레임별로 베이크했으며 길이는 미터 단위입니다.
- `hand_open_close.mp4`: 720×720, 30 FPS, 120프레임 / 4초 미리보기.
- `preview_open.png`, `preview_fist.png`: 시작 자세와 주먹 자세 렌더.
- `verification.json`: Blender 파일 및 GLB 재임포트 검증 결과.

## Blender에서 재생·편집

1. Blender 4.5 이상에서 `hand_open_close.blend`를 엽니다.
2. 타임라인에서 **Space**를 누르면 1–120프레임이 반복 재생됩니다.
3. `Rig`를 선택하고 Pose Mode에서 손가락 뼈를 편집할 수 있습니다.
4. 키프레임은 1, 16, 49, 67, 105, 121프레임에 있습니다. 121프레임은 1프레임과 동일한 루프 경계이며 재생 범위에서는 제외합니다.
5. 오버레이는 미리보기용으로 꺼두었습니다. 뼈를 보려면 뷰포트 우측 상단의 **Show Overlays** 버튼을 켜세요.

원본의 외부 피부 텍스처 대신 자체 포함된 단색 피부·손톱 재질을 적용했습니다. 추가 애드온이나 텍스처 다운로드는 필요 없습니다. 이 동작은 수작업 키프레임 샘플이며 카메라 손 추적 데이터에서 생성한 모션은 아닙니다. 다른 모델에 적용할 때는 뼈 구조에 맞춘 리타기팅이 필요합니다.

## 출처와 라이선스

**3D Rigged Hand Model © 2026 Emma L. D. Lieker**

- 원본: https://github.com/emmalieker/anatomical-hand-model
- 원본 커밋: `f27f19f55f8270b3108ea0b53ce5aa81c543ff7c`
- 원본 파일: `blender/hand_model.blend`
- 원본 SHA-256: `D2A52121CF30AA17014036729EF6C964299F90653B8BE1487EAF3524A04C7D08`
- 라이선스: **CC BY-NC 4.0 — 출처 표기 필수, 비상업적 용도만 허용**.
- 이용 조건: https://creativecommons.org/licenses/by-nc/4.0/
- 저작자 제공 라이선스: `source/LICENSE.txt`
- 변경 사항: 새 손 쥐기·펴기 키프레임, 자체 포함 단색 재질, 카메라와 렌더 설정, GLB 베이크 및 단위 변환, MP4 렌더.

기존 프로젝트의 LICENSE와 별개로 이 모델과 모델을 포함하는 파생 결과에는 위 조건이 적용됩니다.

## 재생성

프로젝트 루트의 PowerShell에서:

```powershell
& .tools/blender-4.5.14-windows-x64/blender.exe --background --factory-startup --disable-autoexec --python samples/hand_animation/create_sample.py
& .tools/blender-4.5.14-windows-x64/blender.exe --background --factory-startup --disable-autoexec --python samples/hand_animation/verify_sample.py
```

휴대용 Blender는 공식 https://download.blender.org/release/Blender4.5/ 에서 받은 `.tools` 폴더에 있습니다. `.tools`는 Git 추적에서 제외했습니다. 재생성에는 `source/hand_model.blend`가 필요합니다.
