"""Inspect a relabeled hand dataset in a local browser (Python standard library only)."""
import argparse
from collections import Counter
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
import zipfile


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
SPLITS = {'train', 'val', 'test'}
UI = Path(__file__).resolve().parent / 'viewer' / 'index.html'


def parse_labels(text):
    """Keep invalid labels visible as errors rather than silently dropping them."""
    hands, errors = [], []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
            if len(values) != 68 or not all(math.isfinite(v) for v in values):
                raise ValueError('유한한 숫자 68개가 필요합니다')
            if values[0] not in (0, 1):
                raise ValueError('클래스는 0(left_hand) 또는 1(right_hand)이어야 합니다')
            cx, cy, width, height = values[1:5]
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError('bbox 범위가 잘못되었습니다')
            points = [values[i:i + 3] for i in range(5, 68, 3)]
            if any(not (0 <= x <= 1 and 0 <= y <= 1 and v in (0, 1, 2)) for x, y, v in points):
                raise ValueError('관절 좌표 또는 표기 마스크 범위가 잘못되었습니다')
            hands.append({'class_id': int(values[0]), 'bbox': values[1:5], 'points': points,
                          'line': line_number})
        except ValueError as exc:
            errors.append(f'{line_number}행: {exc}')
    return hands, errors


class Dataset:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.zip = None
        if self.path.is_file() and zipfile.is_zipfile(self.path):
            self.zip = zipfile.ZipFile(self.path)
            info = [i for i in self.zip.infolist() if not i.is_dir()]
            self.files = {i.filename for i in info}
            if len(self.files) != len(info):
                self.zip.close()
                raise ValueError('중복 파일명이 있는 ZIP은 지원하지 않습니다.')
            signature = [(i.filename, i.file_size, i.CRC) for i in info]
        elif self.path.is_dir():
            paths = [p for p in self.path.rglob('*') if p.is_file() and not p.is_symlink()
                     and p.resolve().is_relative_to(self.path)]
            self.files = {p.relative_to(self.path).as_posix() for p in paths}
            signature = [(p.relative_to(self.path).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                         for p in paths]
        else:
            raise ValueError('재라벨링 ZIP 또는 압축을 푼 데이터 폴더를 지정하세요.')
        candidates = []
        for name in sorted(self.files):
            p = PurePosixPath(name)
            if p.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            for i in range(len(p.parts) - 2):
                if p.parts[i] == 'images' and p.parts[i + 1] in SPLITS:
                    candidates.append((PurePosixPath(*p.parts[:i]), name, i))
        roots = {root for root, _, _ in candidates}
        if len(roots) != 1:
            self.close()
            raise ValueError('images/train 또는 images/val을 가진 데이터 루트가 하나여야 합니다.')
        self.root = roots.pop()
        self.entries = []
        by_image = {}
        for _, name, i in candidates:
            p = PurePosixPath(name)
            rel = p.relative_to(self.root)
            label = self.root / 'labels' / PurePosixPath(*p.parts[i + 1:]).with_suffix('.txt')
            entry = {'id': len(self.entries), 'image': rel.as_posix(), 'split': p.parts[i + 1],
                     'source': '', 'label_exists': label.as_posix() in self.files,
                     '_image': name, '_label': label.as_posix(), 'scores': []}
            self.entries.append(entry)
            by_image[rel.as_posix()] = entry
        self.audit = Counter()
        manifest = (self.root / 'manifest.jsonl').as_posix()
        self.manifest_errors = 0
        if manifest in self.files:
            with self.open_file(manifest) as source:
                for line in source:
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError('Invalid manifest row')
                        status = row.get('status', 'unknown')
                        self.audit[str(status)] += 1
                        entry = by_image.get(row.get('image'))
                        if entry and status == 'accepted':
                            entry['source'] = str(row.get('source', ''))
                            entry['scores'] = [
                                {'handedness': hand.get('handedness'),
                                 'score': hand.get('handedness_score')}
                                for hand in row.get('hands', []) if isinstance(hand, dict)]
                    except (ValueError, TypeError):
                        self.manifest_errors += 1
        self.fingerprint = hashlib.sha256(json.dumps(sorted(signature)).encode()).hexdigest()

    def open_file(self, name):
        if name not in self.files:
            raise FileNotFoundError(name)
        if self.zip:
            return self.zip.open(name)
        target = (self.path / name).resolve()
        if not target.is_relative_to(self.path):
            raise ValueError('데이터 폴더 외부 경로')
        return target.open('rb')

    def read(self, name):
        with self.open_file(name) as source:
            return source.read()

    def index(self):
        return {'name': self.path.name, 'fingerprint': self.fingerprint,
                'items': [{k: v for k, v in item.items() if not k.startswith('_') and k != 'scores'}
                          for item in self.entries],
                'audit': dict(self.audit), 'manifest_errors': self.manifest_errors}

    def item(self, index):
        entry = self.entries[index]
        if not entry['label_exists']:
            hands, errors, raw = [], ['라벨 파일이 없습니다.'], ''
        else:
            raw = self.read(entry['_label']).decode('utf-8-sig')
            hands, errors = parse_labels(raw)
        for i, hand in enumerate(hands):
            # Manifest scores are aligned to original rows, even if one row failed validation.
            row = hand['line'] - 1
            hand['handedness_score'] = (entry['scores'][row].get('score')
                                       if row < len(entry['scores']) else None)
        return {'hands': hands, 'errors': errors, 'raw': raw}

    def close(self):
        if self.zip:
            self.zip.close()


def make_handler(dataset):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, content, mime):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            route = urlparse(self.path).path
            try:
                if route == '/':
                    return self.send(200, UI.read_bytes(), 'text/html; charset=utf-8')
                if route == '/api/index':
                    result = dataset.index()
                elif route.startswith('/api/item/') or route.startswith('/image/'):
                    index = int(route.rsplit('/', 1)[1])
                    if not 0 <= index < len(dataset.entries):
                        raise IndexError(index)
                    if route.startswith('/image/'):
                        name = dataset.entries[index]['_image']
                        return self.send(200, dataset.read(name), mimetypes.guess_type(name)[0] or 'application/octet-stream')
                    result = dataset.item(index)
                else:
                    return self.send(404, b'Not found', 'text/plain')
                self.send(200, json.dumps(result, ensure_ascii=False, allow_nan=False).encode(),
                          'application/json; charset=utf-8')
            except (ValueError, IndexError, OSError, zipfile.BadZipFile) as exc:
                self.send(400, json.dumps({'error': str(exc)}, ensure_ascii=False).encode(),
                          'application/json; charset=utf-8')
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', help='재라벨링 ZIP 또는 데이터 폴더')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('port: 0..65535')
    try:
        dataset = Dataset(args.dataset)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    try:
        with ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(dataset)) as server:
            print(f'Viewer: http://127.0.0.1:{server.server_port}', flush=True)
            print(f'{len(dataset.entries)} images. Ctrl+C to stop. Dataset files are read-only.', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
