"""Write models/rig.json: the nominal camera rig a dataset was generated with (and so the one the
model learned). Run from the repository root on the development PC:

    python hand_tracker_master/tools/export_rig.py --data exports/gigahands_pi3_overlap

Every clip of a dataset shares the nominal rig; this checks a sample of them."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='exports/gigahands_pi3_overlap')
    p.add_argument('--output', default=str(Path(__file__).resolve().parents[1]/'models'/'rig.json'))
    args = p.parse_args()
    root = Path(args.data)
    rows = [json.loads(line) for line in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    reference = None
    for row in rows[::max(1, len(rows)//50)]:
        with np.load(root/row['file'], allow_pickle=False) as data:
            rig = [np.asarray(data[key], np.float64) for key in ('nominal_origins', 'nominal_rotations', 'nominal_intrinsics')]
            meta = json.loads(str(data['metadata']))
        if reference is None:
            reference, config, units = rig, meta['config'], meta['units']
        elif any(np.abs(a-b).max() > 1e-9 for a, b in zip(rig, reference)):
            raise SystemExit(f'{row["file"]} has another nominal rig; the dataset mixes rigs')
    origins, rotations, intrinsics = reference
    rig = dict(
        source=root.name,
        note=('Nominal rig of the training data (gigahands_sim). World units: 1 = world_unit_cm; Y up. '
              'Ray of pixel (px, py) = normalize([(px - cx)/fx, (py - cy)/fy, 1] @ R), from the camera origin. '
              'Cameras 0 and 1 are the slaves, 2 the master Pi. The real rig must be installed like this.'),
        width=int(config['width']), height=int(config['height']), world_unit_cm=float(units['world_unit_cm']),
        origins=origins.tolist(), rotations=rotations.tolist(), intrinsics=intrinsics.tolist())
    Path(args.output).write_text(json.dumps(rig, indent=2), encoding='utf-8')
    print('Wrote', args.output)
    for c in range(3):
        print(f'camera {c}: origin {np.round(origins[c]*rig["world_unit_cm"], 1).tolist()} cm, '
              f'looking along {np.round(rotations[c][2], 3).tolist()}')


if __name__ == '__main__':
    main()
