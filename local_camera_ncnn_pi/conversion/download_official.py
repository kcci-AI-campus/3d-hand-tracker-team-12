"""Download the official latest bundle, then enforce the reviewed artifact hash."""
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import zipfile
from convert_models import BUNDLE_SHA256, SOURCE_HASHES

URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
ROOT = Path(__file__).resolve().parent

def main():
    with urlopen(URL, timeout=120) as response:
        data = response.read()
        metadata = {'url':URL, 'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                    'last_modified':response.headers.get('Last-Modified'), 'etag':response.headers.get('ETag')}
    digest = hashlib.sha256(data).hexdigest()
    if digest != BUNDLE_SHA256:
        raise RuntimeError(f'Official latest has changed ({digest}). Review structure and parity before updating the pinned hashes.')
    target = ROOT / 'source'
    target.mkdir(exist_ok=True)
    archive = zipfile.ZipFile(io.BytesIO(data))
    extracted = {}
    for name, expected in SOURCE_HASHES.items():
        content = archive.read(name+'.tflite')
        if hashlib.sha256(content).hexdigest() != expected: raise RuntimeError(f'Model mismatch: {name}')
        extracted[name+'.tflite'] = content
    (target / 'hand_landmarker.task').write_bytes(data)
    for name, content in extracted.items(): (target / name).write_bytes(content)
    metadata.update({'sha256':digest, 'size_bytes':len(data), 'model_sha256':SOURCE_HASHES})
    (ROOT / 'official_source.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Official bundle verified:',digest)

if __name__ == '__main__': main()
