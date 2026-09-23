"""Rebuild the self-contained Colab notebook from the current model sources.

Cells are templates with @NAME@ placeholders filled from the dataset under exports/;
embedded sources are written byte for byte so their hashes guard resumed runs."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ['train_hand_transformer.py', 'evaluate_hand_transformer.py', 'predict_hand_transformer.py', 'export_lite.py',
           'export_onnx.py']
FILES = SCRIPTS + [path.relative_to(ROOT).as_posix() for path in sorted((ROOT/'hand_tracking').glob('*.py'))]
# A new name: runs started from older notebooks have different source hashes and cannot resume.
DEFAULT_RUN_NAME = 'event_lite_no_roll_v1'


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
                EXTRACT_SPACE=f'{data_bytes} + 1_000_000_000', SOURCE_CELLS=str(len(FILES)),
                MANIFEST_SHA256=repr(hashlib.sha256((dataset/'manifest.jsonl').read_bytes()).hexdigest()))


def fill(text, values):
    """Replace @NAME@ placeholders; an unknown one is an error."""
    text = re.sub(r'@([A-Z_0-9]+)@', lambda match: values[match.group(1)], text)
    return text


def build(dataset_name='gigahands_balanced_no_roll_full'):
    values = dataset_facts(dataset_name)
    cells = []
    def add(kind, source):
        cell = dict(cell_type=kind, id=f'cell-{len(cells):02d}', metadata={}, source=source.strip()+'\n')
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    def md(text): add('markdown', fill(text, values))
    def code(text): add('code', fill(text, values))
    sources = {name: (ROOT/name).read_text(encoding='utf-8') for name in FILES}
    values['SOURCE_HASHES'] = repr({name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()})
    md('''# GigaHands · 두 손 3D Transformer 학습

현재 프로젝트의 **시간(초) 기준 이벤트 모델**을 포함한 독립 실행 노트북입니다. 입력은 실제 도착한 카메라 프레임(이벤트) 목록이고, 임의의 query 시각마다 `[2,21,3]`을 출력합니다. 설정 셀의 `ARCH`로 모델을 고릅니다: 기본 `'lite'`(HandLite: 카메라당 최근 이벤트 8개만 쓰는 고정 크기 모델, ncnn/ONNX로 라즈베리 파이 배포) 또는 `'transformer'`(HandTransformer: 비교 기준). 카메라 보정은 장기 기억 없이 최근 약 0.5초의 이벤트에서만 추정합니다. 별도 저장소 복제가 필요하지 않습니다. 모든 범위가 초 단위라 출력 fps에 묶이지 않으며 실제 파라미터 수는 아래 확인 셀에서 표시합니다. 카메라 보정용 보조 정답 `[3,6]`은 NPZ의 실제·명목 카메라 값에서 계산해 손실에만 사용합니다.

## 먼저 할 일
1. Colab에서 이 `.ipynb`를 엽니다.
2. **런타임 → 런타임 유형 변경 → GPU**를 선택합니다.
3. PC의 `exports/@DATASET@.zip`을 Google Drive의 `MyDrive/GigaHands/`에 올립니다. ZIP 안에는 `manifest.jsonl`, `train/`, `val/`가 있어야 합니다. 원본 GigaHands tar.gz가 아닙니다.
4. 아래 설정 셀의 경로와 실행 이름을 확인한 뒤 위에서부터 실행합니다.

데이터: @DATA_SUMMARY@ Colab GPU 종류·사용 시간은 가용성에 따라 달라집니다. Drive에 많은 작은 파일을 직접 반복 읽는 대신 ZIP을 런타임 로컬로 복사합니다. [Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html)

기존 이전 아키텍처의 smoke 가중치는 사용하지 않습니다. 새 모델을 처음부터 학습하며, 중단 후에는 이 노트북이 저장한 같은 실행의 가중치만 재개합니다.''')
    code('''# 1. 환경 확인 — Colab에 설치된 CUDA용 PyTorch를 그대로 사용합니다.
import sys, os, json, hashlib, shutil, subprocess, importlib.util
from pathlib import Path, PurePosixPath
import torch

assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 5), 'PyTorch 2.5 이상이 필요합니다.'
assert torch.cuda.is_available(), 'Colab 런타임 유형을 GPU로 바꾸고 다시 실행하세요.'
missing = [name for name in ('numpy', 'matplotlib', 'tqdm') if importlib.util.find_spec(name) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
import numpy as np
from tqdm.auto import tqdm
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('GPU memory: %.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 1e9))
WORK_DIR = Path('/content/gigahands_training')
WORK_DIR.mkdir(exist_ok=True)
(WORK_DIR / 'hand_tracking').mkdir(exist_ok=True)
os.chdir(WORK_DIR)
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))''')
    code('''# 2. Drive 연결
from google.colab import drive
drive.mount('/content/drive')''')
    code('''# 3. 사용자 설정
DRIVE_ZIP = Path('/content/drive/MyDrive/GigaHands/@DATASET@.zip')
DRIVE_RUNS = Path('/content/drive/MyDrive/GigaHands/runs')
RUN_NAME = '@RUN_NAME@'  # 이전 노트북의 실행과 소스가 달라 새 이름으로 학습
EPOCHS = 30
BATCH_SIZE = 16                   # GPU 메모리 부족이면 새 실행 이름으로 8 또는 4
LEARNING_RATE = 3e-4
WINDOWS_PER_CLIP = 16             # epoch마다 클립당 무작위 구간; 0이면 전체
VAL_WINDOWS_PER_CLIP = 16         # 매 epoch 고정 검증 구간; 0이면 전체
WORKERS = 2
SEED = 42
RIG_ROTATE_DEG = 45.0             # 현재 수정 모델의 학습 증강 기본값
RIG_SHIFT_WORLD = 0.25            # 1월드=30cm → 축별 최대 7.5cm 이동
# 'lite': HandLite — 고정 크기, ncnn/ONNX 배포용(라즈베리 파이 실시간 목표)
# 'transformer': HandTransformer — 이벤트 전체 attention(비교 기준)
ARCH = 'lite'
MODEL_OPTIONS = {
    'lite': {'--dim': 64, '--heads': 4, '--blocks': 2, '--slots-per-camera': 8,   # 카메라당 최근 이벤트 8개
             '--event-span-s': 0.5, '--anchor-lookback-s': 0.2},
    'transformer': {'--dim': 96, '--heads': 4, '--blocks': 2, '--encoder-layers': 1,
                    '--event-span-s': 0.4, '--calibration-span-s': 0.5, '--anchor-lookback-s': 0.2},
}[ARCH]
DROPOUT = 0.1
CALIBRATION_WEIGHT = 0.01         # 카메라 보정 보조 정답의 손실 가중치
MODEL_ARGS = ['--arch', ARCH, '--dropout', DROPOUT] + [value for item in MODEL_OPTIONS.items() for value in item]

assert RUN_NAME and Path(RUN_NAME).name == RUN_NAME and RUN_NAME not in ('.', '..')
assert EPOCHS > 0 and BATCH_SIZE > 0 and WORKERS >= 0
assert LEARNING_RATE > 0 and WINDOWS_PER_CLIP >= 0 and VAL_WINDOWS_PER_CLIP >= 0
assert DRIVE_ZIP.is_file(), f'Drive에 데이터 ZIP을 올리고 경로를 확인하세요: {DRIVE_ZIP}'
LOCAL_ZIP = Path('/content/@DATASET@.zip')
EXTRACT_DIR = Path('/content/gigahands_dataset')
EXPERIMENT_DIR = DRIVE_RUNS / RUN_NAME
RUN_DIR = EXPERIMENT_DIR / 'checkpoints'
print('학습 결과:', RUN_DIR)''')
    md('''## 데이터 복사 및 검사
처음 실행할 때 데이터 ZIP을 복사·해제합니다. 연결이 끊겨 런타임 로컬 파일이 사라지면 이 셀을 다시 실행하세요. Drive의 원본 ZIP과 체크포인트는 보존됩니다. 진행 막대가 표시됩니다.''')
    code('''# 4. ZIP 로컬 복사 + 안전한 압축 해제 (기존 부분 해제는 다시 완료)
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
            relative = PurePosixPath(info.filename.replace('\\\\', '/'))
            target = (EXTRACT_DIR / str(relative)).resolve()
            if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(EXTRACT_DIR.resolve()):
                raise ValueError(f'잘못된 ZIP 경로: {info.filename}')
            if stat.S_ISLNK(info.external_attr >> 16) or str(relative) in seen:
                raise ValueError(f'링크 또는 중복 ZIP 경로: {info.filename}')
            seen.add(str(relative))
        for info in tqdm(infos, desc='Extract ZIP'):
            relative = PurePosixPath(info.filename.replace('\\\\', '/'))
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
    md('''## 모델 소스
아래 @SOURCE_CELLS@개 셀은 노트북 생성 시점의 모델·로더·학습·평가·추론·ONNX 내보내기 코드를 로컬에 씁니다. 보통 수정할 필요가 없습니다. 보정 보조 정답은 회전 벡터 3개(라디안)와 위치 차이 3개(월드 단위)이며, 실제 카메라 roll=0°라도 월드 축으로 표현한 회전 보정 벡터의 세 번째 성분이 0일 필요는 없습니다. 정답은 모델 forward 입력이 아닌 보조 loss에만 전달합니다.''')
    for name, source in sources.items():
        add('code', f'%%writefile /content/gigahands_training/{name}\n'+source)   # verbatim: hashed
    code('''# 5. 데이터 분리·파일 크기와 소스 확인
import importlib
importlib.invalidate_caches()
from hand_tracking.checkpoints import build_model
from hand_tracking.config import config_class
from hand_tracking.data import read_manifest
SOURCE_HASHES = @SOURCE_HASHES@
for name, expected in SOURCE_HASHES.items():
    actual = hashlib.sha256((WORK_DIR / name).read_text().encode()).hexdigest()
    assert actual == expected, f'소스 변경 감지: {name}. 변경 실험은 노트북과 실행 이름을 새로 만드세요.'
rows = read_manifest(DATA_DIR)  # 참가자/원본 train-val 중복이면 오류
for row in tqdm(rows, desc='Check dataset'):
    assert (DATA_DIR / row['file']).stat().st_size == row['bytes'], row['file']
for split in ('train', 'val'):
    selected = [r for r in rows if r['split'] == split]
    print(split, 'clips:', len(selected), 'participants:', len({r['participant'] for r in selected}))
cfg = config_class(ARCH)(dropout=DROPOUT, **{key[2:].replace('-', '_'): value for key, value in MODEL_OPTIONS.items()})
temporary_model = build_model(ARCH, cfg)
print('parameters:', f'{sum(p.numel() for p in temporary_model.parameters()):,}')
del temporary_model

def run_python(script, *arguments):
    command = [sys.executable, '-u', str(WORK_DIR / script), *map(str, arguments)]
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
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise''')
    md('''## GPU 동작 확인
본학습 전에 실제 데이터의 작은 배치로 CUDA·mixed precision·역전파·체크포인트 저장을 확인합니다. 이 결과는 성능 평가가 아니며 본학습 가중치로 사용하지 않습니다. GPU 메모리가 부족하면 배치 크기를 줄이고 다시 실행하세요.''')
    code('''# 6. GPU smoke test — 본학습 실행 폴더와 분리
import uuid
smoke_dir = WORK_DIR / ('smoke_' + uuid.uuid4().hex[:8])
run_python('train_hand_transformer.py',
           '--data', DATA_DIR, '--output', smoke_dir, '--epochs', 1,
           '--batch-size', min(BATCH_SIZE, 2), *MODEL_ARGS, '--calibration-weight', CALIBRATION_WEIGHT,
           '--max-clips', 4, '--max-train-batches', 2, '--max-val-batches', 2,
           '--windows-per-clip', 2, '--val-windows-per-clip', 2,
           '--workers', 0, '--device', 'cuda')
print('GPU smoke test passed.')''')
    md('''## 본학습 / 재개
- 기본값은 학습 클립마다 epoch당 최대 16개 무작위 윈도우입니다. 윈도우는 query 시각 16개와, 첫 query보다 약 1.35초 전부터 마지막 query까지 도착한 이벤트(최대 128개)입니다.
- 검증 구간은 고정됩니다. MPJPE는 마지막 query의 절대 3D 관절 오차(mm)이며 작을수록 좋습니다.
- 카메라 보정은 최근 약 0.5초의 이벤트에서만 추정합니다(장기 기억 없음; lite는 카메라당 최근 이벤트 8개).
- 카메라/손 전체에 같은 강체 변환을 적용하는 학습 증강이 활성화됩니다. 이것은 데이터 생성기의 개별 카메라 설치 오차와 별개입니다.
- 보조 정답 `calibration_target[3,6]`은 카메라마다 회전 보정 벡터(rad) 3개와 위치 보정(world) 3개입니다. 모델 입력에는 넣지 않고 `CALIBRATION_WEIGHT`로 보조 손실을 조절합니다. 회전 벡터의 세 번째 성분은 카메라의 roll 각도와 다르므로 roll 0°에서도 0이 아닐 수 있습니다.
- **매 epoch 완료 시** Drive에 `last.pt`, 개선 시 `best.pt`를 저장합니다. 런타임이 끊기면 완료되지 않은 epoch는 다시 학습합니다.
- anchor는 이벤트마다 카메라별 최신 ray를 삼각측량한 표본에 직선을 맞춰 query 시각으로 외삽합니다. ray 궤적 직접 풀이와 잔차 게이트는 시뮬레이션 데이터에서 정확도를 낮춰 기본으로 끕니다. 이전 구조의 가중치는 쓸 수 없으니 새 실행으로 학습하세요.
- 같은 설정·같은 `RUN_NAME`으로 다시 실행하면 마지막 완료 epoch부터 재개합니다. 재개 중 배치 크기·epoch 수·모델 설정을 바꾸면 거부됩니다. 설정 변경은 새 `RUN_NAME`으로 시작하세요.
- 첫 epoch가 끝나기 전에 중단되어 `last.pt`가 없으면 새 `RUN_NAME`을 사용하세요. 기존 결과를 자동 삭제하지 않습니다.''')
    code('''# 7. Train / Resume — Drive에 epoch마다 저장
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

resume = (RUN_DIR / 'last.pt').is_file()
if RUN_DIR.exists() and not resume:
    raise RuntimeError('완료된 체크포인트가 없는 기존 실행 폴더입니다. RUN_NAME을 새 이름으로 바꾸고 설정 셀부터 다시 실행하세요.')
arguments = [
    '--data', DATA_DIR, '--output', RUN_DIR, '--epochs', EPOCHS,
    '--batch-size', BATCH_SIZE, *MODEL_ARGS, '--lr', LEARNING_RATE, '--calibration-weight', CALIBRATION_WEIGHT,
    '--windows-per-clip', WINDOWS_PER_CLIP, '--val-windows-per-clip', VAL_WINDOWS_PER_CLIP,
    '--rig-rotate-deg', RIG_ROTATE_DEG, '--rig-shift', RIG_SHIFT_WORLD,
    '--workers', WORKERS, '--threads', 2, '--seed', SEED, '--device', 'cuda',
]
if resume:
    arguments.append('--resume')
    print('Resume:', RUN_DIR / 'last.pt')
run_python('train_hand_transformer.py', *arguments)
print('체크포인트:', RUN_DIR / 'best.pt')''')
    code('''# 8. 학습 곡선
import matplotlib.pyplot as plt
records = [json.loads(line) for line in (RUN_DIR / 'metrics.jsonl').read_text().splitlines() if line.strip()]
# 중단 시 마지막 완료 epoch의 기록이 반복될 수 있어 epoch별 마지막 행 표시
records = list({r['epoch']: r for r in records}.values())
records.sort(key=lambda r: r['epoch'])
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for split in ('train', 'val'):
    axes[0].plot([r['epoch'] for r in records], [r[split]['mpjpe_mm'] for r in records], label=split)
    axes[1].plot([r['epoch'] for r in records], [r[split]['pck20'] * 100 for r in records], label=split)
    axes[2].plot([r['epoch'] for r in records], [r[split]['calibration_mse'] for r in records], label=split)
axes[0].set(title='MPJPE', xlabel='Epoch', ylabel='mm')
axes[1].set(title='PCK @ 20 mm', xlabel='Epoch', ylabel='%')
axes[2].set(title='Calibration auxiliary loss', xlabel='Epoch', ylabel='Normalized MSE')
for ax in axes:
    ax.grid(alpha=.3)
    ax.legend()
fig.tight_layout()
fig.savefig(EXPERIMENT_DIR / 'learning_curves.png', dpi=150)
plt.show()''')
    md('''## 검증: 학습 모델 vs 삼각측량 기준선
동일한 검증 윈도우에서 비교합니다. 기본은 모든 검증 클립의 최대 4개 윈도우입니다. `EVAL_WINDOWS_PER_CLIP=0`으로 바꾸면 전체 검증 윈도우를 평가하므로 시간이 더 걸립니다. 기준선은 학습 없이 nominal 캘리브레이션으로 만든 삼각측량 anchor입니다. 검증은 모델 선택에 사용되었으므로 독립된 최종 test 성능은 아닙니다.''')
    code('''# 9. 같은 조건의 모델/기준선 평가
EVAL_WINDOWS_PER_CLIP = 4
assert (RUN_DIR / 'best.pt').is_file(), '먼저 학습 체크포인트를 생성하세요.'
eval_dir = EXPERIMENT_DIR / ('eval_' + uuid.uuid4().hex[:8])
eval_dir.mkdir()
common = ['--data', DATA_DIR, '--windows-per-clip', EVAL_WINDOWS_PER_CLIP,
          '--batch-size', BATCH_SIZE, '--workers', WORKERS, '--device', 'cuda']
run_python('evaluate_hand_transformer.py', '--triangulation-only', '--output', eval_dir / 'anchor.json',
           '--anchor-lookback-s', MODEL_OPTIONS['--anchor-lookback-s'], *common)
run_python('evaluate_hand_transformer.py', '--checkpoint', RUN_DIR / 'best.pt', '--output', eval_dir / 'model.json', *common)
anchor_report = json.loads((eval_dir / 'anchor.json').read_text())
model_report = json.loads((eval_dir / 'model.json').read_text())
assert anchor_report['samples'] == model_report['samples']
print(f"Anchor MPJPE: {anchor_report['mpjpe_mm']:.2f} mm")
print(f"Model  MPJPE: {model_report['mpjpe_mm']:.2f} mm")
print(f"Reduction: {anchor_report['mpjpe_mm'] - model_report['mpjpe_mm']:.2f} mm (양수이면 개선)")
print('결과 저장:', eval_dir)''')
    code('''# 10. 검증 클립 1개 예측 및 3D 비교
chosen = next(row for row in rows if row['split'] == 'val' and row['windows'] > 0)
prediction_file = EXPERIMENT_DIR / ('prediction_' + uuid.uuid4().hex[:8] + '.npz')
run_python('predict_hand_transformer.py', '--checkpoint', RUN_DIR / 'best.pt',
           '--input', DATA_DIR / chosen['file'], '--output', prediction_file)
with np.load(prediction_file, allow_pickle=False) as data:
    predicted = data['predicted_xyz_cm'].copy()
    target = data['target_xyz'] * float(data['world_unit_cm'])
    valid = data['target_mask'].copy()
from hand_tracking.objectives import BONES
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
    md('''## 배포용 내보내기 (선택)
`ARCH='lite'`이면 `export_lite.py`가 신경망 3개(이벤트 encoder·보정기·decoder)를 **ncnn과 ONNX**로 내보내고, 라즈베리 파이용 런타임(`hand_tracking/lite_runtime.py`: numpy 기하 + ncnn)과 PyTorch 스트림의 차이를 검증 클립에서 확인합니다. 결과 폴더 전체를 장치로 복사하면 PyTorch 없이 `numpy`와 `ncnn`만으로 추론합니다. `ARCH='transformer'`이면 `forward()` 전체를 ONNX 한 개로 내보냅니다(ncnn 미지원).''')
    code('''# 11. 배포용 내보내기 + 런타임 확인
needed = ('onnx', 'onnxscript', 'onnxruntime') + (('ncnn', 'pnnx') if ARCH == 'lite' else ())
missing = [name for name in needed if importlib.util.find_spec(name) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
export_target = EXPERIMENT_DIR / (('deploy_' if ARCH == 'lite' else 'model_') + uuid.uuid4().hex[:8] + ('' if ARCH == 'lite' else '.onnx'))
script = 'export_lite.py' if ARCH == 'lite' else 'export_onnx.py'
run_python(script, '--checkpoint', RUN_DIR / 'best.pt', '--output', export_target,
           '--check-input', DATA_DIR / chosen['file'])
print('내보내기:', export_target)''')
    md('''## 저장 위치 및 다시 시작하기
- `MyDrive/GigaHands/runs/<RUN_NAME>/checkpoints/best.pt`: 검증 오차가 가장 낮은 가중치
- `last.pt`: 마지막 완료 epoch의 모델·optimizer·scheduler·난수 상태
- `run.json`, `metrics.jsonl`: 설정과 학습 기록
- 상위 실행 폴더: 소스 사본, 학습 곡선, 기준선 비교, 예측 결과, 배포용 모델(`deploy_*/` 또는 `model_*.onnx`)

Colab 재연결 시 위에서부터 실행하고 같은 `RUN_NAME`과 설정을 사용하세요. GPU smoke test는 생략할 수 있습니다. 7번 셀이 `last.pt`를 찾아 자동 재개합니다.

**실행 검증 범위:** 노트북 구조·내장 코드·데이터 검사·학습/재개/평가/추론 파이프라인은 로컬 CPU로 검증했습니다. 이 파일을 만들 때 실제 Colab GPU 세션에서 전체 학습을 수행한 것은 아닙니다. Colab에서 6번 GPU smoke test로 CUDA 경로를 먼저 확인하세요.''')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                  language_info=dict(name='python'), accelerator='GPU',
                                  colab=dict(name='gigahands_transformer_colab.ipynb', provenance=[])))
    output=ROOT/'notebooks/gigahands_transformer_colab.ipynb'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'Wrote {output} ({len(cells)} cells, {output.stat().st_size:,} bytes)')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='gigahands_balanced_no_roll_full')
    build(parser.parse_args().dataset)
