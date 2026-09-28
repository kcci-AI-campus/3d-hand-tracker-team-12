"""Rebuild the self-contained Colab (TPU) notebook from the current model sources.

Cells are templates with @NAME@ placeholders filled from the dataset under exports/;
embedded sources are written verbatim (outer whitespace stripped) so their hashes guard resumed runs.
The layout follows the HandLiteV3 TPU notebook (2026-09-28); ARCH picks HandLiteV3 or HandDirect."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [f'training/{name}.py' for name in ('__init__', 'train', 'evaluate', 'predict', 'export')]
FILES = SCRIPTS + [path.relative_to(ROOT).as_posix() for path in sorted((ROOT/'hand_tracking').glob('*.py'))]
# A new name: runs started from older notebooks have different source hashes and cannot resume.
DEFAULT_RUN_NAME = 'event_litev3_tpu_e85_v4'


def dataset_facts(dataset_name):
    """Placeholder values describing a finished export under exports/."""
    dataset = ROOT/'exports'/dataset_name
    if Path(dataset_name).name != dataset_name:
        raise ValueError('Use a dataset folder name under exports')
    state = json.loads((dataset/'progress.json').read_text(encoding='utf-8'))
    if state['status'] not in ('completed', 'size_limit'):
        raise ValueError('Dataset generation must finish before building the notebook')
    rows = [json.loads(line) for line in (dataset/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    data_bytes = sum(r['bytes'] for r in rows)
    archive = ROOT/'exports'/f'{dataset_name}.zip'
    zip_description = f'ZIP 약 {archive.stat().st_size/1e9:.2f}GB' if archive.is_file() else '데이터 ZIP'
    summary = (f'원본 클립 {len(rows):,}개, 학습 {len(train):,}개/검증 {len(val):,}개, '
               f'참가자 {len({r["participant"] for r in train})}명/{len({r["participant"] for r in val})}명. '
               f'{zip_description}, 압축 해제 후 약 {data_bytes/1e9:.2f}GB입니다.')
    return dict(DATASET=dataset_name, RUN_NAME=DEFAULT_RUN_NAME, DATA_SUMMARY=summary,
                EXTRACT_SPACE=f'{data_bytes} + 1_000_000_000', CACHE_SPACE=f'{2*data_bytes}', SOURCE_CELLS=str(len(FILES)),
                MANIFEST_SHA256=repr(hashlib.sha256((dataset/'manifest.jsonl').read_bytes()).hexdigest()))


def fill(text, values):
    """Replace @NAME@ placeholders; an unknown one is an error."""
    return re.sub(r'@([A-Z_0-9]+)@', lambda match: values[match.group(1)], text)


def build(dataset_name='gigahands_pi3_overlap'):
    values = dataset_facts(dataset_name)
    cells = []
    def add(kind, source):
        cell = dict(cell_type=kind, id=f'cell-{len(cells):02d}', metadata={}, source=source.strip()+'\n')
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    def md(text): add('markdown', fill(text, values))
    def code(text): add('code', fill(text, values))
    # Stripped like every cell (add), so the hash is of the file %%writefile recreates.
    sources = {name: (ROOT/name).read_text(encoding='utf-8').strip()+'\n' for name in FILES}
    values['SOURCE_HASHES'] = repr({name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()})
    md(r'''# GigaHands · 두 손 3D 포즈 학습 — HandLiteV3 (직선 맞춤 삼각측량 기준점 + 보정 신경망) · TPU v6e-1

카메라 3대의 비동기 이벤트(도착한 카메라 프레임의 2D 키포인트와 명목 ray)만 입력으로, 임의의 query 시각마다 두 손 관절 `[2,21,3]`과 관절별 예상 오차, 손별 '시야 안' 확률을 출력합니다. 설정 셀의 `ARCH`로 모델을 고릅니다: 기본 `'litev3'`(HandLiteV3) 또는 `'direct'`(HandDirect: 기하 계산 없이 신경망만, 비교용; 예상 오차·시야 판정·기준점 출력 없음).

**관절 상태 (배포·스트림):** 배포 런타임과 이벤트 단위 스트림은 관절마다 마지막 삼각측량(점과, 그때 쓴 최신 검출의 촬영 시각)을 기억합니다. 삼각측량이 성공할 때만 갱신하고 모델 예측은 넣지 않습니다. 상태는 과거 창 탐색(1.0초, 0.2초 간격)이 직전 위치를 **못 찾은 관절에만** 씁니다. 그래서 매 프레임 query해도 입력이 학습(`forward()`, 상태 없음)과 같고, 한 카메라에만 보이는 기간이 아무리 길어도 그 ray 위에서 마지막 삼각측량 깊이로 기준점을 잡습니다(종류 `ray`). 직전 위치 경과 시간 입력은 학습 범위 끝(1.2초)으로 잘라 넣습니다.

**시야 밖 손 (`PRESENCE`):** 손은 항상 있지만 세 카메라 시야를 모두 벗어날 수 있습니다.
- **라벨:** 클립마다 검출된 관절의 u·v와 명목 ray로 카메라별 픽셀 변환을 맞추고, **정답 3D 관절을 실제 카메라 자세로 투영**합니다. 관절의 절반 이상이 어느 한 카메라 화면 안(카메라 앞, [0,1] 범위)에 들어오는 손은 '시야 안', 아니면 '시야 밖'입니다(`data.hands_in_view`). 5번 셀이 맞춤 잔차와 시야 밖 비율을 출력합니다.
- **시야 밖 손:** 위치 손실이 없습니다. 대신 손마다 **'시야 안일 확률'**(`presence`, 토큰 gradient를 끊은 head)을 맞추도록 학습합니다(`PRESENCE_WEIGHT` 0.1).
- **시야 안인데 검출이 빠진 손:** 손이 분명히 거기 있으므로 위치도 계속 학습합니다.
- **지표:** MPJPE 등 위치 지표는 시야 안 손 기준입니다. 로그의 `in_view_acc`는 시야 판정 정확도, `out`은 시야 밖 손 비율입니다.
- **배포:** `runtime.in_view_probability`(손별)가 0.5보다 작으면 그 손은 세 카메라 모두 밖이니 좌표를 무시하면 됩니다.

## 모델 구조
1. **이벤트 도착 시 (프레임마다 한 번):** 관절별 u·v, ray 원점·방향, 촬영→도착 지연, 검출 여부 `[2,21,10]`를 손가락별 encoder에 넣어 손가락 토큰 10개(손 2 × 손가락 5) `[10,64]`를 만들어 저장합니다.
2. **query 시각 t마다: 기하 계산** (학습 파라미터 없음, float32, gradient 없음)
   - **ray:** 카메라 3대 × 관절 42개마다, 그 관절이 검출된 최근 0.2초 안의 프레임들의 ray 방향에 시간에 대한 직선을 최소제곱으로 맞추고 t에서 읽습니다. 검출이 하나면 그대로 씁니다.
   - **삼각측량:** 관절마다 카메라별 ray로 최소제곱 삼각측량합니다.
   - **직전 위치:** 0.2~1.0초 전(0.2초 간격, `prior_step_s`, `prior_span_s`; 각 0.2초 맞춤 구간이 겹치지 않음)에 같은 직선 맞춤 + 삼각측량을 해서, 그 관절이 성공한 가장 최근 위치를 찾습니다. 배포·스트림에서는 여기서 못 찾은 관절만 기억한 마지막 삼각측량을 씁니다.
   - **지금 실패한 관절의 기준점:**
     - 그 관절을 본 카메라가 있으면: 보이는 ray 중 직전 위치에 가장 가까이 지나는 ray 위에서, 직전 위치와 가장 가까운 점. 방향은 지금 관측, 깊이는 과거에서 옵니다.
     - 본 카메라가 없으면: 직전 위치
     - 직전 위치도 없으면(학습: 최근 1.2초 동안 한 번도 삼각측량 안 됨, 배포: 한 번도 삼각측량된 적 없음): 0 (보정 신경망이 위치 전체를 냄)
3. **관절 토큰 입력 56개**
   - 관절 11개: 기준점 3, 기준점 − 직전 위치 3, 기준점 종류(삼각측량 / ray + 직전 깊이 / 직전 위치) 3, 직전 위치가 있는지 1, 직전 위치의 경과 시간 1
   - 카메라별 15개 × 3: 검출 여부, 기울기 사용 여부, 검출 수, 최신·가장 오래된 검출 경과 시간, 최신 u·v 2, ray와 기준점의 어긋남 3, 맞춘 ray와 최신 ray의 차이 3(**맞춤이 과했는지**), 직선에서 벗어난 정도 RMS·최신 검출(**검출이 튀는지**)
4. **보정 신경망 (corrector)**
   - 관절 토큰마다 MLP(56→64→64) + 관절별 학습 embedding
   - 블록 × 2: 카메라별 최근 8개 이벤트(0.5초 안)의 손가락 토큰 240개를 참조(경과 시간 embedding, 빈 null 토큰) → 양손 42개 관절끼리 self-attention → feed-forward
   - 출력: 관절별 보정값 `[42,3]`(0 초기화: 학습 전에는 기준점 그대로)과 예상 오차 `[42]`(log(1+mm), 토큰 gradient를 끊은 별도 head)
5. **최종 좌표 = 기준점 + 보정값**
6. **손실:** 포즈(SmoothL1 β 6mm + 뼈 길이 0.1), 손목 기준 상대 좌표 1.0, 예상 오차 0.1

## 로그
- epoch 줄: MPJPE, PCK20, `rel`(손 모양), `error_miss`(예상 오차가 실제에서 평균 몇 mm 빗나가는지)
- `kinds` 줄: 기준점 종류(`none`, `triangulated`, `ray`, `prior`)별 비율 / MPJPE / (보정 전 기준점만의 오차). 괄호 안과의 차이가 보정 신경망의 효과입니다.

## TPU 학습
- 데이터 로딩이 CPU 병목이라 TPU 호스트 코어를 다 씁니다(worker = 코어 수 − 2, worker마다 스레드 1개, `PREFETCH`개씩 미리 준비). 윈도 이벤트 선택은 이진 탐색으로 필요한 구간만 봅니다.
- float32, 고정 배치 shape(이벤트 `MAX_EVENTS`개로 패딩, 학습 epoch의 마지막 불완전 배치는 버림). 첫 배치는 XLA 컴파일로 느립니다.
- 체크포인트는 CPU 텐서로 저장하므로 CPU 예측·내보내기에서 그대로 열립니다.

## 배포
ncnn 그래프 2개(`encoder` 이벤트 도착 시, `corrector` query 시)와 numpy 런타임(`hand_tracking/runtime.py`: 직선 맞춤, 삼각측량, 직전 위치 탐색, 기준점, 관절 입력, 슬롯 선택)만 씁니다. `runtime.error_mm`으로 관절별 예상 오차(mm), `runtime.anchor_kind`로 기준점 종류를 읽습니다.

## 먼저 할 일
1. Colab에서 이 `.ipynb`를 엽니다.
2. **런타임 → 런타임 유형 변경 → TPU (v6e-1)**를 선택합니다. PyTorch/XLA가 없으면 1번 셀이 설치합니다.
3. PC의 `exports/@DATASET@.zip`을 Google Drive의 `MyDrive/GigaHands/`에 올립니다. ZIP 안에는 `manifest.jsonl`, `train/`, `val/`가 있어야 합니다.
4. 설정 셀의 경로와 실행 이름을 확인한 뒤 위에서부터 실행합니다.

데이터: @DATA_SUMMARY@''')
    code(r'''# 1. 환경 확인 — TPU 런타임(v6e-1)의 PyTorch/XLA. 없으면 설치합니다.
# TPU는 한 프로세스만 쓸 수 있어 이 노트북 커널은 TPU를 잡지 않고, 학습·확인은 하위 프로세스에서 실행합니다.
import sys, os, json, hashlib, shutil, subprocess, importlib.util, importlib.metadata
from pathlib import Path, PurePosixPath

os.environ.setdefault('PJRT_DEVICE', 'TPU')    # 하위 프로세스(학습)에 물려줌
# 데이터 worker가 많으므로 각 프로세스의 numpy(BLAS) 스레드는 1개: 코어를 worker끼리 나눠 쓰고 서로 빼앗지 않게
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
LIBTPU_INDEXES = ['-f', 'https://storage.googleapis.com/libtpu-wheels/index.html',
                  '-f', 'https://storage.googleapis.com/libtpu-releases/index.html']
FALLBACK_VERSION = '2.8.0'   # 설치된 torch에 맞는 torch_xla가 없을 때 함께 설치할 짝 버전

def pip(*packages):
    return subprocess.run([sys.executable, '-m', 'pip', 'install', *packages, *LIBTPU_INDEXES]).returncode == 0

if importlib.util.find_spec('torch_xla') is None:
    # torch를 import하기 전에 설치해야 버전을 바꿔도 런타임을 다시 시작할 필요가 없습니다.
    try:
        torch_version = importlib.metadata.version('torch').split('+')[0]
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    if not (torch_version and pip(f'torch_xla[tpu]=={torch_version}')):
        assert pip(f'torch=={FALLBACK_VERSION}', f'torch_xla[tpu]=={FALLBACK_VERSION}'), 'PyTorch/XLA 설치 실패'
    importlib.invalidate_caches()

missing = [name for name in ('numpy', 'matplotlib', 'tqdm') if importlib.util.find_spec(name) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
import torch
import numpy as np
from tqdm.auto import tqdm
assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 5), 'PyTorch 2.5 이상이 필요합니다.'
probe = subprocess.run([sys.executable, '-c', (
    'import torch, torch_xla, torch_xla.core.xla_model as xm\n'
    'd = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()\n'
    'print("torch_xla:", torch_xla.__version__)\n'
    'print("device:", d, xm.xla_device_hw(d))\n'
    'print("check:", float((torch.ones(4, 4, device=d) @ torch.ones(4, 4, device=d)).sum()))')],
    capture_output=True, text=True)
print(probe.stdout.strip())
assert probe.returncode == 0 and 'TPU' in probe.stdout, (
    'TPU를 쓸 수 없습니다. 런타임 유형을 TPU(v6e-1)로 바꾸고 다시 실행하세요.\n' + probe.stderr[-2000:])
print('PyTorch:', torch.__version__, '| CPU cores:', os.cpu_count())
WORK_DIR = Path('/content/gigahands_training')
WORK_DIR.mkdir(exist_ok=True)
for package in ('hand_tracking', 'training'):
    (WORK_DIR / package).mkdir(exist_ok=True)
os.chdir(WORK_DIR)
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))''')
    code(r'''# 2. Drive 연결
from google.colab import drive
drive.mount('/content/drive')''')
    code(r'''# 3. 사용자 설정
DRIVE_ZIP = Path('/content/drive/MyDrive/GigaHands/@DATASET@.zip')
DRIVE_RUNS = Path('/content/drive/MyDrive/GigaHands/runs')
RUN_NAME = '@RUN_NAME@'   # 소스가 바뀐 노트북은 새 이름으로 학습 (5번 셀의 해시 검사)
ARCH = 'litev3'                   # 'litev3': HandLiteV3 (기본) / 'direct': HandDirect (신경망만, 비교용)
DEVICE = 'xla'                    # TPU (PyTorch/XLA). 예측·내보내기는 CPU
EPOCHS = 85                       # 배치 128: 배치 64·60 epoch보다 스텝이 적어 epoch로 보충
BATCH_SIZE = 128                  # × TRAIN_QUERIES = 스텝당 query 512개
LEARNING_RATE = 8.5e-4            # 배치의 제곱근에 비례: 64일 때 6e-4, 128일 때 8.5e-4
WINDOWS_PER_CLIP = 16             # epoch마다 클립당 무작위 구간; 0이면 전체
TRAIN_QUERIES = 4                 # 학습은 윈도의 마지막 query 4개만(평가 지표는 원래 마지막 query); 0이면 16개 전부
VAL_WINDOWS_PER_CLIP = 16         # 매 epoch 고정 검증 구간; 0이면 전체
# 데이터 로딩이 CPU 병목이라 TPU 호스트의 코어를 다 씁니다: 메인 프로세스와 TPU 전송 스레드 몫 2개를
# 빼고 모두 worker. worker마다 스레드 1개, 배치 PREFETCH개씩 미리 준비
WORKERS = max(2, (os.cpu_count() or 4)-2)
PREFETCH = 4
THREADS = 4                       # 학습 메인 프로세스의 PyTorch CPU 스레드
SEED = 42
MODEL_OPTIONS = {
    'litev3': {
        '--dim': 64, '--heads': 4,
        '--blocks': 2,                         # 보정 신경망 블록 (손가락 토큰 참조 → 관절끼리 → feed-forward)
        '--fit-span-s': 0.2,                   # 기준점: 카메라·관절마다 최근 0.2초의 검출에 직선을 맞춰 query 시각에서 읽고 삼각측량
        '--prior-span-s': 1.0, '--prior-step-s': 0.2,        # 실패한 관절: 0.2~1.0초 전(0.2초 간격)에 같은 방식으로 삼각측량한 가장 최근 성공 위치를 씀
                                                             # (배포·스트림은 이보다 오래된 마지막 삼각측량도 상태로 기억해 씀)
        '--slots-per-camera': 8, '--event-span-s': 0.5,          # 손가락 토큰: 카메라당 최근 이벤트 8개, 0.5초 안
    },
    'direct': {'--dim': 64, '--heads': 4, '--fusion-blocks': 2, '--blocks': 2, '--slots-per-camera': 8,
               '--event-span-s': 0.5},
}[ARCH]
# 한 구간이 담는 최대 이벤트 수. 과거 1.45초(직전 위치 탐색 1.0초 + 직선 맞춤 0.2초 + 도착 여유 0.25초)라
# 약 80~90개로 예상해 128. 고정 shape라 패딩까지 계산하므로 너무 크게 두지 않습니다. 99%가 넘치면 5번 셀이 경고합니다.
MAX_EVENTS = 128
DROPOUT = 0.1
# 손은 항상 있지만 세 카메라 시야를 모두 벗어날 수 있습니다. 정답 3D 관절을 실제 카메라로 투영해 절반 이상이 어느
# 카메라 화면 안에 들어오는 손만 '시야 안'으로 보고, 시야 밖 손은 위치 손실 없이 '시야 밖'이라는 것만 학습합니다.
PRESENCE = True                   # 손마다 '시야 안일 확률' 출력 (토큰의 gradient를 끊어 포즈 학습에는 영향 없음)
PRESENCE_WEIGHT = 0.1             # 시야 안/밖 손실 가중치
ERROR_ESTIMATE = True             # 관절마다 '내 예측이 몇 mm 틀렸을지' 출력. 토큰의 gradient를 끊어 포즈 학습에는 영향 없음
# 손실 가중치 (정답은 rig 좌표계뿐: 참 월드 좌표 손실 없음, 카메라 보정 신경망 없음)
RELATIVE_WEIGHT = 1.0             # 손목 기준 상대 좌표 손실 (손 모양; 손목 위치 오차와 분리)
ERROR_WEIGHT = 0.1                # 예상 오차 손실
MODEL_ARGS = (['--arch', ARCH, '--dropout', DROPOUT, '--max-events', MAX_EVENTS]
              + [value for item in MODEL_OPTIONS.items() for value in item]
              + ([] if ERROR_ESTIMATE or ARCH != 'litev3' else ['--no-error-estimate'])
              + ([] if PRESENCE or ARCH != 'litev3' else ['--no-presence']))   # HandDirect에는 두 출력이 없음
LOSS_ARGS = ['--relative-weight', RELATIVE_WEIGHT, '--error-weight', ERROR_WEIGHT, '--presence-weight', PRESENCE_WEIGHT]

assert RUN_NAME and Path(RUN_NAME).name == RUN_NAME and RUN_NAME not in ('.', '..')
assert ARCH in ('litev3', 'direct')
assert EPOCHS > 0 and BATCH_SIZE > 0 and WORKERS >= 0 and MAX_EVENTS > 0
assert LEARNING_RATE > 0 and WINDOWS_PER_CLIP >= 0 and VAL_WINDOWS_PER_CLIP >= 0 and TRAIN_QUERIES >= 0
assert RELATIVE_WEIGHT >= 0 and ERROR_WEIGHT >= 0 and PRESENCE_WEIGHT >= 0
assert DRIVE_ZIP.is_file(), f'Drive에 데이터 ZIP을 올리고 경로를 확인하세요: {DRIVE_ZIP}'
LOCAL_ZIP = Path('/content/@DATASET@.zip')
EXTRACT_DIR = Path('/content/gigahands_dataset')
CACHE_DIR = Path('/content/gigahands_cache')   # 첫 epoch에 전처리한 클립(압축 없음); 이후 epoch는 여기서 읽음
EXPERIMENT_DIR = DRIVE_RUNS / RUN_NAME
RUN_DIR = EXPERIMENT_DIR / 'checkpoints'
print('학습 결과:', RUN_DIR)''')
    md(r'''## 데이터 복사 및 검사
처음 실행할 때 데이터 ZIP을 복사·해제합니다. 연결이 끊겨 런타임 로컬 파일이 사라지면 이 셀을 다시 실행하세요. Drive의 원본 ZIP과 체크포인트는 보존됩니다. 진행 막대가 표시됩니다.''')
    code(r'''# 4. ZIP 로컬 복사 + 안전한 압축 해제 (기존 부분 해제는 다시 완료)
import zipfile, stat
EXPECTED_MANIFEST_SHA256 = @MANIFEST_SHA256@
zip_size = DRIVE_ZIP.stat().st_size
if not LOCAL_ZIP.is_file() or LOCAL_ZIP.stat().st_size != zip_size:
    assert shutil.disk_usage('/content').free > zip_size + @EXTRACT_SPACE@, '로컬 디스크 공간이 부족합니다.'
    partial = LOCAL_ZIP.with_suffix('.zip.part')
    with DRIVE_ZIP.open('rb') as src, partial.open('wb') as dst, tqdm(total=zip_size, unit='B', unit_scale=True, desc='Copy ZIP') as bar:
        while chunk := src.read(8 * 1024 * 1024):
            dst.write(chunk)
            bar.update(len(chunk))
    partial.replace(LOCAL_ZIP)

EXTRACT_DIR.mkdir(exist_ok=True)
ready = EXTRACT_DIR / '.ready.json'
reuse = ready.is_file() and json.loads(ready.read_text()).get('manifest_sha256') == EXPECTED_MANIFEST_SHA256
if not reuse:
    with zipfile.ZipFile(LOCAL_ZIP) as archive:
        infos = archive.infolist()
        assert shutil.disk_usage('/content').free > sum(i.file_size for i in infos) + 1_000_000_000, '압축 해제 공간이 부족합니다.'
        seen = set()
        for info in infos:
            relative = PurePosixPath(info.filename.replace('\\', '/'))
            target = (EXTRACT_DIR / str(relative)).resolve()
            if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(EXTRACT_DIR.resolve()):
                raise ValueError(f'잘못된 ZIP 경로: {info.filename}')
            if stat.S_ISLNK(info.external_attr >> 16) or str(relative) in seen:
                raise ValueError(f'링크 또는 중복 ZIP 경로: {info.filename}')
            seen.add(str(relative))
        for info in tqdm(infos, desc='Extract ZIP'):
            relative = PurePosixPath(info.filename.replace('\\', '/'))
            target = EXTRACT_DIR / str(relative)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)

manifests = list(EXTRACT_DIR.rglob('manifest.jsonl'))
assert len(manifests) == 1, 'manifest.jsonl이 하나인 데이터 ZIP이 필요합니다.'
DATA_DIR = manifests[0].parent
manifest_hash = hashlib.sha256(manifests[0].read_bytes()).hexdigest()
assert manifest_hash == EXPECTED_MANIFEST_SHA256, '이 노트북에 연결된 균등 데이터셋과 ZIP이 다릅니다.'
ready.write_text(json.dumps({'manifest_sha256': manifest_hash}))
print('DATA_DIR:', DATA_DIR)''')
    md(r'''## 모델 소스
아래 @SOURCE_CELLS@개 셀은 모델·로더·학습·평가·예측·ncnn 내보내기 코드를 로컬에 씁니다. 보통 수정할 필요가 없습니다. 5번 셀이 각 파일의 해시를 확인하므로, 코드를 바꾸는 실험은 노트북과 실행 이름을 새로 만드세요.
- `training/`: `train.py`, `evaluate.py`, `predict.py`, `export.py` (명령줄 진입점)
- `hand_tracking/model.py`: HandLiteV3 (직선 맞춤 ray, 삼각측량, 직전 위치로 실패 관절의 기준점, 관절 입력, 보정 신경망), 이벤트 단위 스트림
- `hand_tracking/geometry.py`: 최소제곱 삼각측량, ray와 점의 각도 잔차
- `hand_tracking/networks.py`, `layers.py`: encoder 입력, attention 블록, 손가락 토큰 encoder
- `hand_tracking/events.py`, `data.py`, `stream.py`: 이벤트 수락·슬롯 선택, NPZ 구간 로더, 스트리밍 버퍼
- `hand_tracking/engine.py`, `objectives.py`: epoch 실행, 손실·지표
- `hand_tracking/deploy.py`, `runtime.py`: ncnn 내보내기, PyTorch 없는 numpy 런타임
- `hand_tracking/direct.py`, `direct_export.py`, `direct_runtime.py`: HandDirect (비교용) 모델·내보내기·런타임
- `hand_tracking/accelerator.py`: PyTorch/XLA 장치·동기화·CPU 저장''')
    for name, source in sources.items():
        add('code', f'%%writefile /content/gigahands_training/{name}\n'+source)   # verbatim: hashed
    code(r'''# 5. 데이터 분리·파일 크기와 소스 확인
import importlib, uuid
importlib.invalidate_caches()
SOURCE_HASHES = @SOURCE_HASHES@
for name, expected in SOURCE_HASHES.items():
    actual = hashlib.sha256((WORK_DIR / name).read_text().encode()).hexdigest()
    assert actual == expected, f'소스 변경 감지: {name}. 변경 실험은 노트북과 실행 이름을 새로 만드세요.'
from hand_tracking.checkpoints import build_model
from hand_tracking.config import ARCHITECTURES
from hand_tracking.data import HandWindows, read_manifest
rows = read_manifest(DATA_DIR)  # 참가자/원본 train-val 중복이면 오류
for row in tqdm(rows, desc='Check dataset'):
    assert (DATA_DIR / row['file']).stat().st_size == row['bytes'], row['file']
for split in ('train', 'val'):
    selected = [r for r in rows if r['split'] == split]
    print(split, 'clips:', len(selected), 'participants:', len({r['participant'] for r in selected}))
heads = dict(error_estimate=ERROR_ESTIMATE, presence=PRESENCE) if ARCH == 'litev3' else {}
cfg = ARCHITECTURES[ARCH](dropout=DROPOUT, **heads, **{key[2:].replace('-', '_'): value for key, value in MODEL_OPTIONS.items()})
# 시야 판정 확인: 클립마다 검출된 관절의 u·v와 명목 ray로 카메라별 픽셀 변환(u = a·x + b)을 맞추고, 정답 관절을 실제
# 카메라로 투영해 시야 안/밖을 정합니다(data.hands_in_view). 맞춤 잔차가 크면 이 가정(핀홀, ray = 픽셀 @ R)이 틀린 것입니다.
from hand_tracking.data import pixel_model, read_clip
residuals, out_share = [], []
# 시야 밖 손은 전체 query의 약 1%이고 일부 클립에 몰려 있어 20개 표본으로는 0%가 나올 수 있으므로 100개를 봅니다.
for index in tqdm(np.random.default_rng(1).choice(len(rows), min(100, len(rows)), replace=False), desc='In-view check'):
    with np.load(DATA_DIR / rows[index]['file'], allow_pickle=False) as clip:
        _, rms = pixel_model(clip['features'], clip['input_mask'], clip['nominal_rotations'])
    residuals += [value for value in rms if np.isfinite(value)]
    in_view = read_clip(DATA_DIR / rows[index]['file'])['hand_in_view']
    out_share.append(1-in_view.mean())
print(f'픽셀 변환 맞춤 잔차(화면 폭 대비): 중앙 {np.median(residuals):.4f}, 최대 {np.max(residuals):.4f}; '
      f'시야 밖 손 비율(query 기준): 평균 {100*np.mean(out_share):.1f}%, 클립별 최대 {100*np.max(out_share):.1f}%')
if not residuals or np.max(residuals) > 0.01:
    print('경고: 픽셀 변환 맞춤 잔차가 화면의 1%를 넘습니다. 시야 판정이 부정확할 수 있으니 학습 전에 확인하세요.')
# 고정값 확인(값은 바꾸지 않음): 카메라 fps가 20 이하라 직선 맞춤 창에 카메라당 최대 fit_span_s*20 프레임,
# 구간당 이벤트는 MAX_EVENTS 안에 들어와야 합니다. 넘치면 가장 오래된 이벤트부터 버려집니다.
probe = HandWindows(DATA_DIR, 'val', windows_per_clip=4, max_clips=30, context_s=cfg.context_s, max_events=4096)
counts, intervals = [], []
for sample in probe:
    present = sample['event_present']
    counts.append(int(present.sum()))
    for camera in range(3):
        capture = np.sort(sample['event_capture'][present & (sample['event_camera'] == camera)].numpy())
        if len(capture) > 1:
            intervals.append(np.diff(capture))
counts = np.array(counts)
fps = 1/float(np.median(np.concatenate(intervals)))
fit = f'직선 맞춤 창 {cfg.fit_span_s:g}초에 카메라당 약 {fps*cfg.fit_span_s:.0f}프레임; ' if ARCH == 'litev3' else ''
print(f'데이터의 카메라 fps 약 {fps:.1f} ({fit}'
      f'손가락 토큰 창 {cfg.event_span_s:g}초에 약 {fps*cfg.event_span_s:.0f}프레임, slots_per_camera={cfg.slots_per_camera})')
print(f'구간당 이벤트 수: 중앙 {int(np.median(counts))}, 99% {int(np.percentile(counts, 99))}, 최대 {counts.max()} '
      f'(MAX_EVENTS={MAX_EVENTS}, context {cfg.context_s:.2f}s)')
if np.percentile(counts, 99) > MAX_EVENTS:
    print(f'경고: 구간 이벤트 수가 MAX_EVENTS를 넘습니다. 설정 셀에서 {int(np.percentile(counts, 99))} 이상으로 늘리세요.')
temporary_model = build_model(cfg)
print('parameters:', f'{sum(p.numel() for p in temporary_model.parameters()):,}')
del temporary_model


def _stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_python(module, *arguments):
    command = [sys.executable, '-u', '-m', module, *map(str, arguments)]
    print(' '.join(command), flush=True)
    process = subprocess.Popen(command, cwd=WORK_DIR, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        _stop(process)
        raise


def run_training(arguments, epochs, batch_size, done=0):
    """run_python('training.train') with progress bars read from its log: all epochs, and
    this epoch's train/val batches (moved every 50 batches, as train.py logs; the batch totals
    are upper bounds, so a bar fills at the phase's end). Other lines print as usual."""
    import math, re
    command = [sys.executable, '-u', '-m', 'training.train', *map(str, arguments)]
    print(' '.join(command), flush=True)
    overall = tqdm(total=epochs, initial=done, desc='전체', unit='epoch')
    phases = {name: tqdm(total=1, desc=f'{name} (epoch {done+1})', unit='batch') for name in ('train', 'val')}
    sizes = {}
    def finish(bar):
        if bar.n < bar.total:
            bar.update(bar.total-bar.n)
    process = subprocess.Popen(command, cwd=WORK_DIR, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            if sizes_line := re.search(r'train_windows/epoch<=([\d,]+) val_windows<=([\d,]+)', line):
                for name, windows in zip(phases, sizes_line.groups()):
                    sizes[name] = max(math.ceil(int(windows.replace(',', ''))/batch_size), 1)
                    phases[name].reset(total=sizes[name])
                print(line, end='', flush=True)
            elif progress := re.match(r'(train|val) batch=(\d+) samples=\d+ MPJPE=([\d.]+)mm(.*)', line):
                name, batches = progress[1], int(progress[2])
                if name == 'val':
                    finish(phases['train'])
                bar = phases[name]
                bar.update(min(batches, bar.total)-bar.n)
                bar.set_postfix_str(f'MPJPE {progress[3]}mm {progress[4].strip()}'.strip())
            elif ended := re.match(r'epoch=(\d+)/\d+ (.*)', line):
                epoch = int(ended[1])
                overall.update(epoch-overall.n)
                overall.set_postfix_str(ended[2].strip())
                print(line, end='', flush=True)
                for name, bar in phases.items():   # ready for the next epoch
                    if epoch < epochs:
                        bar.reset(total=sizes.get(name, 1))
                        bar.set_description(f'{name} (epoch {epoch+1})')
                    else:
                        finish(bar)
            else:
                print(line, end='', flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        _stop(process)
        raise
    finally:
        for bar in (overall, *phases.values()):
            bar.close()''')
    md(r'''## TPU 동작 확인
본학습 전에 실제 데이터의 작은 배치로 PyTorch/XLA·컴파일·역전파·체크포인트 저장(CPU로 옮겨 저장)을 확인합니다. 성능 평가가 아니며 본학습 가중치로 쓰지 않습니다. 첫 배치는 XLA 컴파일로 시간이 걸립니다.''')
    code(r'''# 6. TPU smoke test — 본학습 실행 폴더와 분리
smoke_dir = WORK_DIR / ('smoke_' + uuid.uuid4().hex[:8])
run_python('training.train',
           '--data', DATA_DIR, '--output', smoke_dir, '--epochs', 1,
           '--batch-size', min(BATCH_SIZE, 2), *MODEL_ARGS, *LOSS_ARGS,
           '--max-clips', 4, '--max-train-batches', 2, '--max-val-batches', 2, '--train-queries', TRAIN_QUERIES,
           '--windows-per-clip', 2, '--val-windows-per-clip', 2,
           '--workers', 0, '--device', DEVICE)
# TPU에서 저장한 체크포인트가 torch_xla 없이(CPU) 열리는지 확인 — 예측·내보내기 셀이 CPU에서 읽습니다.
from hand_tracking.checkpoints import load_checkpoint
smoke = load_checkpoint(smoke_dir / 'last.pt')
assert all(p.device.type == 'cpu' for p in smoke.model.parameters())
del smoke
print('TPU smoke test passed.')''')
    md(r'''## 본학습 / 재개
- 학습 클립마다 epoch당 최대 16개 무작위 윈도입니다. 윈도는 query 시각 16개와, 첫 query보다 1.45초(직전 위치 탐색 1.0초 + 직선 맞춤 0.2초 + 도착 여유 0.25초) 전부터 마지막 query까지 도착한 이벤트(최대 `MAX_EVENTS`개)입니다. 학습은 윈도의 마지막 query 4개만 씁니다.
- 검증 구간은 고정입니다. MPJPE는 마지막 query의 3D 관절 오차(mm)입니다. 정답 좌표계는 `rig`(카메라 배치를 실제로 놓인 대로 본 좌표계, 관측할 수 없는 rig 전체 어긋남 제거)이고, 참 월드 좌표 오차는 `mpjpe_world_mm`으로 함께 기록합니다.
- 로그: `rel=`은 손목 기준 상대 좌표 오차(손 모양), `error_miss=`는 예상 오차가 실제 오차에서 평균 몇 mm 빗나가는지입니다. 다음 줄 `kinds`는 기준점 종류별 비율 / MPJPE / (보정 전 기준점만의 오차)입니다.
- **매 epoch 완료 시** Drive에 `last.pt`, 개선 시 `best.pt`를 저장합니다. 같은 설정·같은 `RUN_NAME`으로 다시 실행하면 마지막 완료 epoch부터 재개합니다. 설정을 바꾸면 재개가 거부되니 새 `RUN_NAME`으로 시작하세요.
- 첫 epoch가 끝나기 전에 중단되어 `last.pt`가 없으면 새 `RUN_NAME`을 쓰세요. 기존 결과는 자동으로 지우지 않습니다.
- 학습 증강은 없습니다. 카메라 배치가 고정이라 그 배치의 월드 좌표를 그대로 배웁니다.''')
    code(r'''# 7. Train / Resume — TPU(XLA)에서 학습, Drive에 epoch마다 저장
# 첫 train/val 배치는 XLA 컴파일로 수십 초~수 분 걸립니다(로그의 'first XLA step'). 이후 같은 shape는 재사용합니다.
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
source_file = EXPERIMENT_DIR / 'source_hashes.json'
if source_file.is_file():
    assert json.loads(source_file.read_text()) == SOURCE_HASHES, '이 실행의 모델 소스가 달라졌습니다. 새 RUN_NAME을 사용하세요.'
else:
    source_file.write_text(json.dumps(SOURCE_HASHES, indent=2))
    snapshot = EXPERIMENT_DIR / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in SOURCE_HASHES:
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORK_DIR / name, snapshot / name)

if not CACHE_DIR.exists():
    assert shutil.disk_usage('/content').free > @CACHE_SPACE@, '전처리 캐시(데이터의 약 2배) 공간이 부족합니다.'
resume = (RUN_DIR / 'last.pt').is_file()
if RUN_DIR.exists() and not resume:
    raise RuntimeError('완료된 체크포인트가 없는 기존 실행 폴더입니다. RUN_NAME을 새 이름으로 바꾸고 설정 셀부터 다시 실행하세요.')
arguments = [
    '--data', DATA_DIR, '--output', RUN_DIR, '--epochs', EPOCHS, '--batch-size', BATCH_SIZE, '--lr', LEARNING_RATE,
    *MODEL_ARGS, *LOSS_ARGS,
    '--windows-per-clip', WINDOWS_PER_CLIP, '--val-windows-per-clip', VAL_WINDOWS_PER_CLIP, '--train-queries', TRAIN_QUERIES,
    '--workers', WORKERS, '--prefetch', PREFETCH, '--threads', THREADS, '--seed', SEED, '--device', DEVICE,
    '--cache-dir', CACHE_DIR,
]
done = 0
if resume:
    arguments.append('--resume')
    print('Resume:', RUN_DIR / 'last.pt')
    done = len({json.loads(line)['epoch'] for line in (RUN_DIR / 'metrics.jsonl').read_text().splitlines() if line.strip()})
run_training(arguments, EPOCHS, BATCH_SIZE, done)
print('체크포인트:', RUN_DIR / 'best.pt')''')
    code(r'''# 8. 학습 곡선
import matplotlib.pyplot as plt
records = [json.loads(line) for line in (RUN_DIR / 'metrics.jsonl').read_text().splitlines() if line.strip()]
# 중단 시 마지막 완료 epoch의 기록이 반복될 수 있어 epoch별 마지막 행 표시
records = list({r['epoch']: r for r in records}.values())
records.sort(key=lambda r: r['epoch'])
epochs = [r['epoch'] for r in records]
metric = lambda split, key, scale=1: [np.nan if r[split].get(key) is None else r[split][key] * scale for r in records]
kind_metric = lambda key, name: [np.nan if r['val'][key].get(name) is None else r['val'][key][name] for r in records]
fig, axes = plt.subplots(1, 4, figsize=(21, 4))
for split in ('train', 'val'):
    axes[0].plot(epochs, metric(split, 'mpjpe_mm'), label=split)
    axes[1].plot(epochs, metric(split, 'pck20', 100), label=split)
    axes[2].plot(epochs, metric(split, 'mpjpe_rel_mm'), label=split)
for name in ('triangulated', 'ray', 'prior'):             # val: per anchor kind, corrected vs the anchor alone
    line, = axes[3].plot(epochs, kind_metric('kind_mpjpe_mm', name), label=name)
    axes[3].plot(epochs, kind_metric('kind_anchor_mpjpe_mm', name), linestyle=':', color=line.get_color())
axes[0].set(title='MPJPE', xlabel='Epoch', ylabel='mm')
axes[1].set(title='PCK @ 20 mm', xlabel='Epoch', ylabel='%')
axes[2].set(title='Wrist-relative MPJPE (hand shape)', xlabel='Epoch', ylabel='mm')
axes[3].set(title='Val MPJPE by anchor kind (dotted: anchor alone)', xlabel='Epoch', ylabel='mm')
for ax in axes:
    ax.grid(alpha=.3)
    ax.legend()
fig.tight_layout()
fig.savefig(EXPERIMENT_DIR / 'learning_curves.png', dpi=150)
plt.show()''')
    md(r'''## 검증
고정 검증 윈도에서 `best.pt`를 평가합니다. 기본은 모든 검증 클립의 최대 4개 윈도입니다. `EVAL_WINDOWS_PER_CLIP=0`이면 전체 검증 윈도를 평가합니다. 검증은 모델 선택에 쓰였으므로 독립된 최종 test 성능은 아닙니다.''')
    code(r'''# 9. 검증 평가 (고정 검증 구간, TPU)
EVAL_WINDOWS_PER_CLIP = 4
assert (RUN_DIR / 'best.pt').is_file(), '먼저 학습 체크포인트를 생성하세요.'
eval_dir = EXPERIMENT_DIR / ('eval_' + uuid.uuid4().hex[:8])
eval_dir.mkdir()
run_python('training.evaluate', '--checkpoint', RUN_DIR / 'best.pt', '--output', eval_dir / 'model.json',
           '--data', DATA_DIR, '--windows-per-clip', EVAL_WINDOWS_PER_CLIP, '--batch-size', BATCH_SIZE,
           '--workers', WORKERS, '--device', DEVICE)
report = json.loads((eval_dir / 'model.json').read_text())
print(f"MPJPE: {report['mpjpe_mm']:.2f} mm (rig 좌표계), 참 월드 좌표: {report['mpjpe_world_mm']:.2f} mm")
print(f"PCK@20mm: {report['pck20']*100:.1f}%")
print(f"손목 기준 MPJPE(손 모양): {report['mpjpe_rel_mm']:.2f} mm")
if report.get('presence_accuracy') is not None:
    print(f"시야 안/밖 판정 정확도: {report['presence_accuracy']*100:.1f}% (시야 밖 손 비율 {report['out_of_view_rate']*100:.1f}%, 마지막 query)")
if report.get('error_miss_mm') is not None:
    print(f"예상 오차의 평균 빗나감: {report['error_miss_mm']:.2f} mm (작을수록 신뢰도가 정확)")
names = {'triangulated': '삼각측량 기준점', 'ray': '한 ray + 직전 삼각측량 깊이', 'prior': '직전 삼각측량 위치(ray 없음)',
         'none': '기준점 없음(직전 삼각측량도 없음)'}
print('기준점 종류별 (마지막 query, 정답 있는 관절):')
for kind, name in names.items():
    rate, mm = report['kind_rate'].get(kind), report['kind_mpjpe_mm'].get(kind)
    alone = report['kind_anchor_mpjpe_mm'].get(kind)
    if rate is not None:
        print(f"  {name}: {rate*100:.1f}%" + ('' if mm is None else f", MPJPE {mm:.2f} mm")
              + ('' if alone is None else f" (보정 전 기준점만 {alone:.2f} mm)"))
print('결과 저장:', eval_dir)''')
    code(r'''# 10. 검증 클립 1개 예측 및 3D 비교
import matplotlib.pyplot as plt
from hand_tracking.objectives import BONES
chosen = next(row for row in rows if row['split'] == 'val' and row['windows'] > 0)
prediction_file = EXPERIMENT_DIR / ('prediction_' + uuid.uuid4().hex[:8] + '.npz')
run_python('training.predict', '--checkpoint', RUN_DIR / 'best.pt',
           '--input', DATA_DIR / chosen['file'], '--output', prediction_file)
with np.load(prediction_file, allow_pickle=False) as data:
    predicted = data['predicted_xyz_cm'].copy()
    target = data['target_xyz'] * float(data['world_unit_cm'])
    valid = data['target_mask'].copy()
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
for hand, color in enumerate(('tab:blue', 'tab:orange')):
    for a, b in BONES:
        if valid[hand, a] and valid[hand, b]:
            ax.plot(*target[hand, [a,b]].T, color=color, linestyle='--', alpha=.6)
        ax.plot(*predicted[hand, [a,b]].T, color=color)
ax.set(xlabel='X (cm)', ylabel='Y (cm)', zlabel='Z (cm)', title='Prediction: solid / Ground truth: dashed')
combined = np.concatenate((predicted.reshape(-1,3), target[valid]), axis=0)
center = (combined.min(0)+combined.max(0))/2
radius = max(float(np.ptp(combined,axis=0).max())/2, 1.)
ax.set_xlim(center[0]-radius,center[0]+radius)
ax.set_ylim(center[1]-radius,center[1]+radius)
ax.set_zlim(center[2]-radius,center[2]+radius)
ax.set_box_aspect((1,1,1))
plt.show()
print('원본:', chosen['source'])
print('저장:', prediction_file)''')
    md(r'''## 배포용 내보내기 (선택)
`training/export.py`가 신경망을 **ncnn**으로 내보내고(pnnx 변환, 그래프 `encoder`·`corrector`), 라즈베리 파이용 numpy 런타임(`hand_tracking/runtime.py`)의 결과를 검증 클립에서 PyTorch 스트림과 비교합니다. 결과 폴더 전체를 장치로 복사하면 PyTorch 없이 `numpy`와 `ncnn`만으로 추론합니다. 10번 셀(예측)의 검증 클립을 쓰므로 그 셀을 먼저 실행하세요.''')
    code(r'''# 11. 배포용 내보내기 (ncnn) + 런타임 확인
missing = [name for name in ('ncnn', 'pnnx') if importlib.util.find_spec(name) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
export_target = EXPERIMENT_DIR / ('deploy_' + uuid.uuid4().hex[:8])
run_python('training.export', '--checkpoint', RUN_DIR / 'best.pt', '--output', export_target,
           '--check-input', DATA_DIR / chosen['file'])
print('내보내기:', export_target)''')
    md(r'''## 저장 위치 및 다시 시작하기
- `MyDrive/GigaHands/runs/<RUN_NAME>/checkpoints/best.pt`: 검증 오차가 가장 낮은 가중치 (CPU 텐서로 저장되어 GPU·CPU에서도 그대로 열림)
- `last.pt`: 마지막 완료 epoch의 모델·optimizer·scheduler·난수 상태(TPU 난수 포함)
- `run.json`, `metrics.jsonl`: 설정과 학습 기록
- 상위 실행 폴더: 소스 사본, 학습 곡선, 검증 결과(`eval_*/`), 예측 결과, 배포용 ncnn 모델(`deploy_*/`)

Colab 재연결 시 위에서부터 실행하고 같은 `RUN_NAME`과 설정을 쓰세요. TPU smoke test는 생략할 수 있습니다. 7번 셀이 `last.pt`를 찾아 자동 재개합니다(끝나지 않은 epoch는 처음부터 다시 학습). 재개할 때도 첫 배치는 다시 컴파일합니다.

**실행 검증 범위:** 내장 코드는 로컬 CPU 테스트로 검증했습니다. 이 노트북을 만들 때 실제 TPU에서 실행한 것은 아니니 6번 TPU smoke test로 먼저 확인하세요. 매 배치가 첫 배치만큼 느리면 배치마다 shape가 바뀌어 다시 컴파일한다는 뜻입니다.''')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                  language_info=dict(name='python'), accelerator='TPU', gpuType='V6E1',
                                  colab=dict(name='gigahands_colab.ipynb', provenance=[], gpuType='V6E1')))
    output = ROOT/'notebooks/gigahands_colab.ipynb'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Wrote {output} ({len(cells)} cells, {output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='gigahands_pi3_overlap')
    build(parser.parse_args().dataset)
