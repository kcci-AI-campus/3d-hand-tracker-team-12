# GigaHands 다중 카메라 학습 데이터 생성기

## Roll 없는 카메라 설치 오차

카메라 화면의 roll은 월드 Y-up 기준 **0°로 고정**합니다. 위치는 X/Y/Z 각 축 표준편차 3cm, 방향은 yaw(월드 Y축 회전)·pitch(고도각) 각각 표준편차 5°로 뽑습니다. 회전벡터 Z값만 0으로 놓는 대신, 방향을 뽑고 수평인 카메라 좌표계를 구성하므로 합성 회전에서도 roll이 생기지 않습니다. 위치·방향 오차는 클립 동안 고정이며 상한은 없습니다.

NPZ에는 `sampled_yaw_pitch_errors_deg`(카메라별 yaw/pitch 추출값), `actual_roll_deg`(실제 카메라 roll), metadata의 `camera_roll_enabled=false`를 추가했습니다. 기존 데이터셋은 자동으로 변경되지 않습니다.

전체 원본 클립을 용량 제한 없이 생성하려면:

`python gigahands_balanced.py --input "C:\Users\kccistc\Downloads\keypoints_3d_mano_align.tar.gz" --output exports/gigahands_balanced_no_roll_full --all-clips --gui`

## 일괄 생성 / Batch

참가자를 고르게 담고 학습/검증 참가자를 분리하려면:

`python gigahands_balanced.py --input "C:\Users\kccistc\Downloads\keypoints_3d_mano_align.tar.gz" --output exports/balanced_run --max-gb 5 --gui`

전체 참가자를 번갈아 선택하고, 참가자 안에서는 동작 폴더를 번갈아 선택합니다. 각 동작 폴더의 클립 순서는 seed로 섞습니다. 클립당 카메라 증강은 1회이며, 참가자 약 80%는 train, 20%는 val에만 들어갑니다. 따라서 파일 개수 비율이 정확히 80:20이라는 뜻은 아닙니다. `split.json`에 참가자 배정, `selection_plan.json`에 전체 선택 순서, `manifest.jsonl`에 실제 생성 목록을 기록합니다. 끝난 참가자는 선택 대상에서 빠집니다. 5GB에 도달하면 마지막 파일까지 저장하고 중단합니다. 원본 전체를 한 번 스트리밍해 임시 캐시에 담으므로 생성 중 디스크 사용량은 최종 NPZ 한도보다 크며, 종료 시 임시 캐시를 자동 정리합니다. 새 출력 폴더가 필요하며 이 균등 생성 명령은 재개를 지원하지 않습니다.

아래 명령은 압축 내 순서대로 처리하는 기존 방식입니다:

`python gigahands_batch.py --input "C:\Users\kccistc\Downloads\keypoints_3d_mano_align.tar.gz" --output exports/batch_run --variants 3 --max-gb 5 --gui`

중단된 실행을 이어서 처리하려면 동일 명령에 `--resume`을 추가하세요. 기존 설정이 일치하는지와 manifest에 기록된 파일 크기를 검사하고, 이미 완료된 변형은 건너뜁니다. `--max-gb`를 늘리면 기존 결과를 포함한 새 용량 한도까지 이어갈 수 있습니다.

새 출력 폴더에 클립당 3개 NPZ를 생성합니다. 같은 클립의 세 파일은 시간·검출 노이즈 seed를 공유하고 카메라 설치 오차 seed만 바꿉니다. 클립이 바뀌면 두 seed 모두 변경됩니다. 원본 목록을 한 번 검사한 뒤 TAR 순서대로 스트리밍하여 반복 압축 해제를 피합니다. 5GB는 5,000,000,000바이트이며 현재 클립의 세 파일을 모두 저장한 뒤 한도를 확인하므로 약간 초과할 수 있습니다. 진행 창과 CLI, `progress.json`에 진행률·용량을 표시합니다. `manifest.jsonl`에는 원본 경로·난수·프레임 수·학습 window 수를 기록합니다. Stop은 현재 클립까지 저장 후 중단하며, 완료 파일은 유지합니다. 출력 폴더가 이미 존재하면 덮어쓰지 않고 거부합니다. 학습/검증 분리 시 같은 원본 클립의 증강본을 반드시 같은 split에 배치하세요.

**Browse Clips**으로 마지막으로 연 tar.gz의 목록을 다시 열 수 있습니다. GUI 실행 중에는 목록을 재사용하며, 파일 크기·수정 시각이 바뀌면 다시 읽습니다. 최근 동작은 최대 8개/약 128MiB까지 메모리에 보관합니다(현재 동작 한 개가 한도를 넘으면 해당 동작은 유지). 처음 여는 동작이나 캐시에서 빠진 동작은 압축 스트림을 다시 읽습니다. 앱을 종료하면 메모리 캐시는 사라집니다.

**Randomize Pose**는 현재 동작·재생 위치·뷰를 유지하고 위치·각도 오차를 다시 추출합니다. 통신 지연과 검출 노이즈 난수는 유지하며, 투영과 관측 가능 여부는 새 카메라 자세에 맞춰 다시 계산합니다. 별도 `pose_seed`를 설정 JSON과 데이터 metadata에 기록해 같은 결과를 재현할 수 있습니다. 입력칸에 수정 중인 설정도 함께 적용됩니다.

3D 씬의 회색 점선 카메라는 명목 보정값, 분홍 실선 카메라는 오차 적용 후 실제 위치·회전·내부 파라미터입니다. 시야 사각뿔, 광축, 위쪽 표식으로 yaw·pitch 방향 차이를 표시하며 오차를 시각적으로 과장하지 않습니다. 상단에는 카메라마다 실제 이동 거리(cm)와 상대 회전각(°)을 표시합니다. 손은 ground truth이며 손목 ray는 명목 보정값으로 계산한 모델 입력입니다. 약한 오차는 두 카메라 표시가 겹쳐 보일 수 있습니다.

GUI는 카메라 배치(cm), 설치 오차 범위, 학습 데이터, 프레임 지연을 표시합니다. 위치 표준편차는 기본 3cm/축, 각도 표준편차는 5°/yaw·pitch이며 상한 없이 정규분포로 추출합니다. GUI 최소·최대는 각각 3/3cm, 5/5°로 시작하므로 표준편차는 고정이고 실제 오차만 무작위입니다. 설정 변경 후 **Apply**을 누르세요. 같은 난수 번호와 설정은 같은 오차를 재현하며, 위치·각도 오차는 클립 동안 고정됩니다. 시계·렌즈·누락 등 세부값은 설정 JSON에 보관하며, 프레임 지연은 별도 탭에서 설정합니다.

## 프레임 지연 확인

**프레임 지연** 탭에서 실측 재생/수동 설정을 선택한 뒤 **Apply**로 반영합니다. 실측 재생에서는 프로파일의 촬영 간격·처리·전송 지연을 함께 적용하며, 적용되지 않는 수동 입력칸은 비활성화합니다. 적용된 프로파일의 측정 총 지연 평균/P95와 전송 구간 평균을 표시합니다. 이는 측정 원본의 통계이며 현재 생성 클립의 평균은 아닙니다.

수동 설정에서는 카메라 FPS, 송신 전 처리 시간, 전송 지연 평균·표준편차, 촬영 시각 지터를 설정합니다. 해당 값이 숨은 랜덤 범위로 덮어써지지 않으며, 총 지연은 처리 시간 + 0 이상으로 제한한 정규분포 전송 지연입니다. 수동 모드의 rolling shutter 추가 지연은 0입니다. 관측 유효 시간은 두 모드에서 모두 적용됩니다.

카메라 미리보기 아래에 현재 선택된 입력의 **촬영→도착 지연**, **관측 나이**, 촬영·도착 시각을 표시합니다. 촬영→도착은 처리·대기·전송을 포함하며, 관측 나이는 현재 예측 시각−촬영 시각이므로 두 값이 다릅니다. 표시에는 시계 오차가 섞이지 않은 physical_capture_time을 사용합니다. 프레임이 아직 도착하지 않았거나 유효 시간이 지나면 사용 가능한 프레임 없음으로 표시합니다.

## 실행

현재 Windows 작업 환경에서는 `powershell -ExecutionPolicy Bypass -File .\start_gigahands.ps1`로 실행할 수 있습니다. 일반 Python이 없으면 이 컴퓨터의 Codex 번들 Python과 로컬 Tcl 복사본을 사용합니다.

Python 3.10 이상과 Tkinter가 필요합니다. Windows python.org 설치에는 Tkinter가 포함됩니다.

```powershell
python -m pip install -r requirements-simulator.txt
python gigahands_sim.py
```

GUI는 합성 손 데모로 시작합니다. **Load Motion**으로 시퀀스를 선택하고 설정을 수정한 다음 **Apply**를 누르세요. 재생/일시정지, 타임라인 이동, 드래그 회전, 휠 확대, 3대 카메라의 2D 관측을 지원합니다. **Export NPZ**는 마지막으로 적용한 설정의 결과를 저장합니다. 설정은 JSON으로 저장/복원할 수 있습니다. 실제 GigaHands 파일과 MANO 모델은 포함하거나 자동 다운로드하지 않습니다.

```powershell
python gigahands_sim.py --input path/to/sequence.json --config settings.json --output train_clip.npz
python gigahands_sim.py --output demo_dataset.npz
python -m unittest discover -s tests -v
```

## 저장된 NPZ 읽기

생성 GUI의 **NPZ Reader** 버튼 또는 아래 명령으로 읽기 전용 리더를 엽니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\start_gigahands.ps1 --inspect
python npz_reader.py path/to/dataset.npz
python npz_reader.py path/to/dataset.npz --summary
```

배열을 선택한 후 `30,0,0,0`처럼 인덱스를 입력하거나 `0:3,0,0`처럼 슬라이스를 입력하세요. `metadata`는 생성 설정을 JSON으로 보여 줍니다. 프레임·카메라·손·관절을 선택하면 input 14개 특성, 같은 예측 시점의 3D 정답, 마스크와 타임스탬프를 비교할 수 있습니다. 목록은 헤더만 읽지만 값 조회는 해당 압축 배열 전체를 메모리에 로드합니다.

## 원본 데이터와 좌표

### tar.gz 직접 열기

**Load Motion**에서 `.tar.gz`, `.tgz`, `.tar`를 선택하면 압축 내부 시퀀스 목록을 검색하고 한 클립을 선택할 수 있습니다. `keypoints_3d_mano_align`도 지원합니다. 전체 파일을 디스크에 풀지 않고 선택한 JSON/NPY/NPZ만 메모리에서 읽습니다. 목록 생성은 gzip 스트림 전체를 검사하므로 큰 파일에서 오래 걸릴 수 있으며 GUI는 백그라운드에서 처리합니다. 선택한 클립은 메모리에 캐시하여 오차 설정만 바꿀 때 다시 압축을 읽지 않습니다. 파일별 최대 크기는 512MiB입니다.

```powershell
python gigahands_sim.py --input keypoints_3d_mano_align.tar.gz --list-sequences
python gigahands_sim.py --input keypoints_3d_mano_align.tar.gz --member p001-folder/keypoints_3d_mano_align/000.json --output clip_000.npz
```

저장 metadata의 source에는 `압축경로::내부시퀀스경로`가 기록됩니다. RGB 영상 또는 MANO params 전용 압축은 3D 키포인트 입력이 아니므로 별도 키포인트 압축을 사용하세요. 구조가 다른 JSON 후보는 선택 후 shape 검사에서 오류를 표시합니다.

[GigaHands 공식 로더](https://github.com/brown-ivl/GigaHands/blob/main/dataset/dataset_hands.py)의 `keypoints_3d_mano/<sequence>.json` 배열을 지원합니다. 지원 shape는 `[T,126]`, `[T,42,3]`, `[T,2,21,3]`입니다. 일반 NPY, `points`와 선택적 `timestamps` 배열이 있는 NPZ/JSON도 지원합니다. MANO pose 파라미터 파일이나 2D 관측 파일은 입력할 수 없습니다. 폴더 전체 일괄 변환은 CLI를 시퀀스별로 호출하세요.

타임스탬프 단위는 초이며 원본 시퀀스 시작을 0으로 이동합니다. 타임스탬프가 없으면 `source_fps`로 생성합니다. 원본 FPS는 실제 데이터에 맞게 지정해야 합니다. NaN 관절은 결측으로 처리하며 0 좌표는 유효한 좌표입니다.

원본 손/관절 순서를 기본적으로 보존합니다. `swap_hands`와 `joint_order`(출력 관절 인덱스에 대응하는 원본 인덱스 21개)로 변경합니다. 뷰어 뼈대는 손목 0, 엄지 1–4, 검지 5–8, 중지 9–12, 약지 13–16, 소지 17–20 순서이므로 원본 joint convention을 확인해 설정하세요. hand 0/1을 확인 없이 left/right로 가정하지 않습니다.

`world_unit_cm=30`이므로 **1월드=30cm**이며 (-1,1) 공간은 각 축 -30~30cm, 전체 60×60×60cm입니다. 카메라 위치는 (-30,-30,15), (30,-30,15), (0,30,30)cm이며 원점까지 거리는 각각 45, 45, 약 42.4cm입니다. 저장된 XYZ와 ray 원점에 30을 곱하면 cm가 됩니다. 단위 길이는 `metadata.units`에 저장하며 ray 방향은 무차원 단위벡터입니다.

`axis_order`, `axis_sign`으로 축 변환 후 시퀀스 전체에 **하나의** 중심 이동/스케일을 적용합니다. 프레임별 중심 이동으로 손 움직임을 제거하지 않습니다. 기본 `center=true`, `fit_extent=0.65`는 모든 점을 원점 중심 ±0.65월드(±19.5cm)에 맞춥니다. 자동 맞춤은 원본 손 크기를 바꿉니다. 물리적 크기를 유지하려면 `fit_extent=0`으로 하고 원본 mm는 `scale=1/300`(JSON에는 0.0033333333333333335), cm는 1/30, m는 1/0.3을 사용하세요. 원본 단위는 자동 추정하지 않습니다. 최종 좌표가 (-1,1)을 벗어나면 실패하며 자르지 않습니다. 변환은 metadata에 저장합니다.

## 카메라와 오차

### 기본값: 실측 시간 재생 + 약한 랜덤 오차

새로 실행하면 실측 시간 프로파일과 낮은 검출 오차, 확대된 설치 오차 범위를 기본 적용합니다. 설정 파일은 `profiles/measured_mild.json`이며 **Load Config**로 열 수 있습니다. 기존에 저장한 설정 파일을 불러오면 그 파일의 값이 유지됩니다.

`timing_profile=profiles/pi_c270_timing.json`이 기본입니다. 실제 Pi→PC 측정에서 초기 5초를 제외한 5,015개의 촬영 간격·송신 전 처리/대기 시간·전송 구간 시간을 한 행으로 저장했습니다. 한 행의 시간 관계를 유지하며 연속으로 재생하고, 카메라마다 랜덤 시작 행을 선택합니다. 로그 끝에서는 순환합니다. 평균 촬영 간격은 58.54ms(약 17.1FPS), 읽기 완료→수신 추정 지연은 평균 65.80ms/P95 123.83ms, 송신 준비→수신은 평균 1.47ms입니다. 송신 전 처리·대기·인코딩 합계는 평균 64.33ms입니다.

실측 모드에서는 `camera_fps`, `capture_jitter_ms`, `processing_ms`, `latency_ms`, `latency_jitter_ms`, `rolling_shutter_ms`를 추가 적용하지 않습니다. 따라서 전체 지연에 추론 시간을 다시 더하는 이중 계산을 피합니다. 기존 raw 시계 차이 약 1.27초는 사용하지 않으며 clock offset/drift 증강은 보정 후 남을 수 있는 작은 잔여 오차 가정입니다. 센서 노출/드라이버 버퍼 지연은 측정 범위 밖입니다.

한 대에서 측정한 동일 시간 프로필을 3대의 독립 시작점으로 사용하는 모델이며, 실제 3대 동시 실행 부하를 측정한 것은 아닙니다. 시작 seed가 같으면 같은 결과입니다. 원본 `source_fps=30`과 정답 `output_fps=30`은 별도 설정으로 유지합니다. 30FPS 센서의 건너뛴 호스트 캡처 효과는 실측 관측 간격에 이미 포함됩니다.

`timing_profile`을 빈 문자열로 지정하면 기존 수식 모드로 돌아갑니다. 수식 모드는 고정 처리 시간과 Gaussian 통신 지연을 사용하므로 실측의 긴 꼬리와 처리 시간 변동을 완전히 재현하지 못합니다. 프로필 적용 여부·우선순위·SHA256·카메라별 시작 행은 `metadata.timing_replay`에 기록됩니다.

GUI **설치 오차**에서 위치·각도 표준편차의 최소·최대값을 설정합니다. 각 시퀀스마다 범위 내에서 균등하게 표준편차를 뽑고, 그 값으로 카메라별 고정 설치 오차를 생성합니다. **학습 데이터 → 난수 번호**를 변경하면 다른 증강본을 만듭니다. 다른 노이즈 범위는 JSON의 `error_ranges`에 보관합니다. 수동 오차 설정 파일을 GUI로 열면 기존 강도를 최소=최대인 범위로 변환합니다.

해상도와 명목 대각 화각은 설정값으로 고정됩니다(기본 320×240, 55°). 이 모드에서는 `intrinsics` 대신 화각에서 계산한 내부 파라미터를 사용하며 `focal_std_pct`, `principal_std_px`는 0으로 고정합니다. 지정한 카메라 기준 위치, 좌표계, 원본 움직임, FPS, 윈도 설정은 유지됩니다. 왜곡 k1/k2는 랜덤이므로 가장자리 영상 위치는 바뀔 수 있습니다.

| 랜덤 변수 | 기본 샘플링 범위 |
|---|---|
| 위치 오차 표준편차 | 0.1 월드 단위/축 = 3cm/축 (고정) |
| 각도 오차 표준편차 | 5°/yaw·pitch (고정) |
| k1 / k2 | -0.01~0.01 / -0.003~0.003 |
| 픽셀 노이즈 표준편차 | 0.15~0.6px |
| 이상치 확률 / 크기 표준편차 | 0~0.1% / 1~3px |
| 관절 누락률 / 추가 프레임 손실률 | 0~0.5% / 0~0.1% |
| 통신 지연 평균 / 표준편차 (수식 모드만) | 1.2~1.8ms / 0.5~1.2ms |
| 처리 시간 / 촬영 지터 표준편차 (수식 모드만) | 60~68ms / 0~1ms |
| 잔여 시계 offset 표준편차 / drift 표준편차 | 0~0.3ms / 0~2ppm |
| 추가 롤링셔터 읽기 시간 | 기본 0ms |

설치 오차는 실측 사양이 아닌 증강용 가정입니다. 각 축의 위치 표준편차 3cm, yaw와 pitch 각각의 표준편차 5°로 상한 없는 Gaussian 표본을 뽑습니다. 실제 이동 거리나 상대 회전각은 이 값보다 클 수 있습니다. 카메라마다 독립적으로 한 번 뽑아 클립 동안 고정합니다. 기본 JSON의 `position_limit_cm`, `angle_limit_deg`는 `null`이며 GUI에서는 예전 설정 파일을 불러와도 상한을 제거합니다. CLI는 이전 설정 파일의 숫자 상한을 계속 지원하며 0은 오차를 끕니다. 입력은 과거 관측만 선택하고 정답 자세는 바꾸지 않습니다.

실제로 적용된 오차는 `actual_position_errors_cm`, `actual_angle_errors_deg` 배열에 저장됩니다. NPZ Reader의 관측 비교 화면에서도 확인할 수 있습니다. 이 두 배열은 진단 전용이며 모델 입력에 포함하지 마세요. 기존 NPZ에 단위 정보가 없으면 리더는 임의로 30cm를 적용하지 않습니다.

`metadata.config`에는 실제 적용된 값, `metadata.randomization`에는 요청 설정과 샘플링 결과가 저장됩니다. 수동 오차값을 사용하려면 `randomize_errors=false`로 설정하세요. 기존 설정 JSON에 이 키가 없어도 기본 랜덤 모드가 적용됩니다. 기존 NPZ는 자동 변경되지 않으므로 새로 생성해야 합니다.

기본 카메라 위치는 `[-1,-1,0.5]`, `[1,-1,0.5]`, `[0,1,1]`; 세 카메라 모두 원점을 바라봅니다. 월드 Y가 위쪽이고 영상 u는 오른쪽, v는 아래쪽입니다. `targets`로 방향을 변경합니다. 320×240, 정사각 픽셀, 중앙 주점, pinhole 모델이 기본입니다.

[Logitech 제품 페이지](https://www.logitech.com/en-us/shop/p/c270-hd-webcam)의 대각 화각 55°로 초점 거리를 추정합니다. 이는 해당 장치의 320×240 캘리브레이션 값이 아닙니다. 모드별 크롭/리사이즈에 따라 달라지므로 실제 촬영 모드에서 보정한 `intrinsics=[[fx,fy,cx,cy], ...]`를 사용하세요. 비어 있으면 화각을 사용합니다.

오차 설정:

| 변수 | 의미 |
|---|---|
| position_std / angle_std_deg | 카메라별 고정 위치/회전 보정 오차 표준편차 |
| focal_std_pct / principal_std_px | 카메라별 고정 초점/주점 오차 |
| k1 / k2 | 실제 투영의 방사 왜곡; ray는 명목 pinhole로 복원하여 잔여 왜곡 반영 |
| pixel_std | 관절별 독립 Gaussian 검출 노이즈(px) |
| outlier_prob / outlier_std_px | 큰 검출 오류 확률/크기 |
| missing_prob / packet_loss | 관절 누락/프레임 통신 손실 확률 |
| latency_ms / latency_jitter_ms | 0 이상으로 제한한 Gaussian 통신 지연 평균 파라미터/표준편차 |
| processing_ms | 고정 추론·처리 시간 |
| capture_jitter_ms | 촬영 시점 흔들림; 카메라마다 독립 시작 위상 |
| clock_offset_std_ms / clock_drift_std_ppm | 카메라 시계의 고정 offset/drift |
| rolling_shutter_ms | 첫 행부터 마지막 행까지 읽는 시간; 행별 손 위치 보간 |
| max_age_ms | 관측 만료 기준 |

시간 프로필은 실측 기반이고, 공간·검출·추가 누락 오차는 완화한 증강 가정입니다. 누락률을 실측으로 추정한 것은 아닙니다. 수식 모드의 롤링셔터는 첫 투영으로 행을 추정하는 근사입니다. 물체/손 표면에 의한 실제 가림, 조명, 압축, 모션블러, MediaPipe 추론 편향은 렌더링하지 않습니다. 독립 누락 노이즈이므로 긴 연속 가림을 재현하지는 않습니다. 영상 범위 밖의 점이나 원본 결측은 랜덤 누락률을 0으로 해도 입력에서 제외됩니다.

## 시간 정렬 및 학습 계약

각 카메라 촬영 시점의 3D 자세를 선형 보간하여 2D로 투영합니다. 실측 모드의 도착 시간은 촬영 + 실측 송신 전 구간 + 실측 전송 구간입니다. 수식 모드는 촬영 + 롤링셔터 읽기 + 처리 + 통신 지연입니다. 각 query 시점까지 도착한 프레임 중 가장 최근 촬영 프레임만 사용하며, 늦게 도착한 옛 프레임으로 되돌아가지 않습니다. 아직 도착하지 않았거나 만료된 관측은 마스크가 false입니다. 정답은 **query 시점**의 두 손 3D 좌표입니다. 따라서 지연된 관측으로 현재 자세를 예측하는 문제입니다.

| NPZ key | Shape / 의미 |
|---|---|
| features | `[T,3,2,21,14]` float32 |
| input_mask | `[T,3,2,21]` bool; true=유효 |
| in_frame_mask | `[T,3,2,21]`; 선택된 관측이 원본 유효·카메라 앞·영상 범위 안인지. true인데 input_mask=false이면 추가 랜덤 관절 누락 |
| target_xyz / target_mask | `[T,2,21,3]` / `[T,2,21]` |
| pixels / ray_directions | `[T,3,2,21,2]` / `[T,3,2,21,3]` |
| query_time | `[T]` 초 |
| capture_time / arrival_time | `[T,3]` 카메라 보고 촬영 시각 / 서버 도착 시각 |
| physical_capture_time | `[T,3]` 오차 없는 실제 촬영 시각; 진단 전용 |
| selected_frame | `[T,3]` 원본 카메라 이벤트 번호; -1=없음 |
| window_starts | 연속 학습 윈도 시작 인덱스; 짧은 시퀀스는 빈 배열 |
| nominal_* / actual_* | 명목 보정 / 실제 투영 파라미터; actual은 진단 전용 |
| metadata | JSON 문자열: 설정, 변환, feature 이름, source |

14개 feature는 정규화 u,v; 명목 ray 원점 xyz; 월드 단위 ray 방향 xyz; 보고 촬영시각−query; 도착시각−query; 도착−보고 촬영시각; camera/hand/joint ID 순서입니다. 시계 오차가 있으면 관측 지연값이 음수일 수 있습니다. 실제 통신 시간과 동일하다고 가정하지 마세요. 누락 feature는 모두 0이며 **반드시 mask와 함께** 사용하세요. physical/actual 배열과 정답은 입력에 넣지 마세요.

```python
import numpy as np
from gigahands_sim import training_window

with np.load('train_clip.npz', allow_pickle=False) as data:
    x, valid, y, target_valid = training_window(data, 0)
    # x: [window*3*2*21, 14], y: [2,21,3]
    # PyTorch Transformer key_padding_mask = ~valid
    # target_valid인 관절만 loss에 사용. 전부 padding인 샘플은 제외.
```

`training_window`는 시각 feature를 마지막 query 기준으로 다시 정렬합니다. 동일 관측이 여러 query에서 유지되면 반복 토큰이 생깁니다. 원하는 경우 camera ID + capture timestamp로 중복 제거하세요. 이 프로그램은 Transformer 학습용 데이터를 생성하며 모델 학습 자체는 포함하지 않습니다. 학습/검증/시험 분리는 **참가자 또는 원본 시퀀스 단위로 먼저** 수행하고, 같은 원본의 다른 seed/중첩 window가 서로 다른 split에 들어가지 않도록 하세요.
