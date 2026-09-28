"""The C++ runtime's golden fixture (cpp/hand_lite/tests/data) still describes the Python
LiteRuntime: replay the recorded network outputs through it and compare every input,
outcome and pose. When this fails after a deliberate Python change, regenerate the fixture
with cpp/hand_lite/tools/make_golden.py (and rebuild/run the C++ golden_test)."""
from collections import deque
from pathlib import Path
import unittest
import numpy as np
from hand_tracking.lite_runtime import LiteRuntime

DATA = Path(__file__).resolve().parents[1]/'cpp'/'hand_lite'/'tests'/'data'


def read_golden(path):
    words = iter(path.read_text(encoding='utf-8').split())
    take = lambda n, kind=float: np.array([kind(next(words)) for _ in range(n)])
    def tensor():
        shape = tuple(int(next(words)) for _ in range(int(next(words))))
        return take(int(np.prod(shape))).astype(np.float32).reshape(shape)
    assert [next(words), next(words)] == ['hand_lite_golden', '1']
    operations = []
    for word in words:
        if word == 'end':
            return operations
        assert word == 'op'
        op = dict(kind=next(words))
        if op['kind'] == 'push':
            op['camera'], op['capture'], op['arrival'] = int(next(words)), float(next(words)), float(next(words))
            op['features'] = take(42*14).astype(np.float32).reshape(2, 21, 14)
            op['valid'] = take(42, int).astype(bool).reshape(2, 21)
        else:
            op['time'] = float(next(words))
        assert next(words) == 'calls'
        op['calls'] = []
        for _ in range(int(next(words))):
            assert next(words) == 'call'
            name, inputs = next(words), int(next(words))
            op['calls'].append((name, [tensor() for _ in range(inputs)], tensor()))
        assert next(words) == 'result'
        op['result'] = next(words)
        if op['result'] == 'ok':
            op['expected'] = take(42*3+18)
        operations.append(op)
    raise AssertionError('golden.txt has no end')


class Replay:
    def __init__(self):
        self.pending = deque()

    def run(self, name, *arrays):
        recorded, inputs, output = self.pending.popleft()
        assert recorded == name, (recorded, name)
        for actual, expected in zip(arrays, inputs, strict=True):
            np.testing.assert_allclose(np.asarray(actual, np.float32), expected, atol=1e-5, err_msg=name)
        return output


@unittest.skipUnless((DATA/'golden.txt').is_file(), 'C++ golden fixture not generated')
class CppGoldenFixtureTests(unittest.TestCase):
    def test_fixture_matches_python_runtime(self):
        runtime = LiteRuntime(DATA, graphs=Replay())            # recorded outputs: no ncnn needed
        operations = read_golden(DATA/'golden.txt')
        for number, op in enumerate(operations):
            runtime.graphs.pending = deque(op['calls'])
            try:
                if op['kind'] == 'push':
                    outcome = str(int(runtime.push(op['camera'], op['features'], op['valid'], op['capture'], op['arrival'])))
                else:
                    pose, calibration = runtime.query_details(op['time'])
                    outcome = 'ok'
                    np.testing.assert_allclose(np.concatenate((pose.ravel(), calibration.ravel())), op['expected'],
                                               atol=1e-9, err_msg=f'operation {number}')
            except ValueError:
                outcome = op['result'] if op['result'].startswith('error_') else 'error'
            self.assertEqual(outcome, op['result'], f'operation {number}')
            self.assertFalse(runtime.graphs.pending, f'operation {number}: unused recorded calls')
        self.assertGreater(sum(op['kind'] == 'query' for op in operations), 40)


if __name__ == '__main__':
    unittest.main()
