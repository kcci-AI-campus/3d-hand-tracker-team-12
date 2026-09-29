import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import numpy as np
from gigahands_sim import demo_motion, load_motion
from motion_archive import list_motion_members, read_motion_member


class ArchiveTests(unittest.TestCase):
    def test_list_load_and_no_extract(self):
        p,t = demo_motion(5)
        payload = json.dumps(p.reshape(5,42,3).tolist()).encode()
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/'motions.tar.gz'
            with tarfile.open(archive,'w:gz') as out:
                for name in ['p001/keypoints_3d_mano_align/000.json', 'p001/keypoints_3d_mano_align/001.json', 'p001/params/000.json']:
                    info = tarfile.TarInfo(name); info.size = len(payload)
                    out.addfile(info, io.BytesIO(payload))
                link = tarfile.TarInfo('p001/keypoints_3d_mano_align/link.json')
                link.type = tarfile.SYMTYPE; link.linkname = '000.json'; out.addfile(link)
            members = list_motion_members(archive)
            self.assertEqual(len(members),2)
            loaded,times = load_motion(archive,30,members[0])
            np.testing.assert_allclose(loaded,p); np.testing.assert_allclose(times,t)
            self.assertEqual(list(Path(folder).iterdir()),[archive])
            with self.assertRaises(ValueError): load_motion(archive)
            with self.assertRaises(ValueError): read_motion_member(archive,'missing.json')
            with self.assertRaises(ValueError): read_motion_member(archive,link.name)

    def test_npy_archive(self):
        p,t = demo_motion(4); buffer = io.BytesIO(); np.save(buffer,p)
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/'motion.tgz'
            with tarfile.open(archive,'w:gz') as out:
                info = tarfile.TarInfo('motion.npy'); info.size = len(buffer.getvalue())
                out.addfile(info,io.BytesIO(buffer.getvalue()))
            loaded,_ = load_motion(archive)
            np.testing.assert_allclose(loaded,p)


if __name__ == '__main__': unittest.main()
