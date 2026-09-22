"""Download Google's version 1 float16 Hand Landmarker task bundle."""
from pathlib import Path
from urllib.request import urlopen

URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'


def main():
    target = Path(__file__).resolve().parent / 'models' / 'hand_landmarker.task'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        print(f'Already exists: {target}')
        return
    partial = target.with_suffix('.part')
    try:
        with urlopen(URL, timeout=60) as response, partial.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if partial.stat().st_size < 1_000_000:
            raise RuntimeError('Downloaded model is unexpectedly small')
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    print(target)


if __name__ == '__main__':
    main()
