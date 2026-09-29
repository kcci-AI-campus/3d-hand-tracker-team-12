"""Read motion members directly from TAR archives without extracting paths."""
import io
from pathlib import PurePosixPath
import tarfile


MAX_MEMBER_BYTES = 512 * 1024 * 1024


def is_motion_archive(path):
    return str(path).lower().endswith(('.tar.gz', '.tgz', '.tar'))


def list_motion_members(path):
    candidates, preferred, seen = [], [], set()
    with tarfile.open(path, 'r|*') as archive:
        for info in archive:
            p = PurePosixPath(info.name)
            if not info.isfile() or p.suffix.lower() not in ('.json', '.npy', '.npz'):
                continue
            if info.name in seen:
                raise ValueError(f'중복된 압축 내부 파일명: {info.name}')
            seen.add(info.name)
            if any(part in ('params', 'keypoints_2d', 'bboxes') for part in p.parts):
                continue
            if not 0 < info.size <= MAX_MEMBER_BYTES:
                continue
            candidates.append(info.name)
            if any(part.startswith('keypoints_3d') for part in p.parts):
                preferred.append(info.name)
    result = preferred or candidates
    if not result:
        raise ValueError('읽을 수 있는 3D 시퀀스 JSON/NPY/NPZ가 없습니다. RGB/params 전용 압축인지 확인하세요. 파일당 한도는 512MiB입니다.')
    return sorted(result, key=lambda name: ('keypoints_3d_mano' not in name, name))


def read_motion_member(path, name):
    # No extract()/extractall(): member names never become local file paths.
    with tarfile.open(path, 'r|*') as archive:
        for info in archive:
            if info.name != name:
                continue
            if not info.isfile():
                raise ValueError('일반 파일만 읽을 수 있습니다. 링크는 지원하지 않습니다.')
            if not 0 < info.size <= MAX_MEMBER_BYTES:
                raise ValueError('선택한 시퀀스 크기는 0 초과 512MiB 이하여야 합니다.')
            with archive.extractfile(info) as stream:
                content = stream.read(MAX_MEMBER_BYTES + 1)
            if len(content) != info.size:
                raise ValueError('압축 시퀀스가 잘렸거나 크기가 일치하지 않습니다.')
            return io.BytesIO(content)
    raise ValueError(f'압축 내부 시퀀스를 찾을 수 없습니다: {name}')
