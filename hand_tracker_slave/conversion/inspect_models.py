"""Inspect the unmodified official TFLite FlatBuffers; no guessed tensor names."""
import collections
import hashlib
import json
from pathlib import Path
import tflite

ROOT = Path(__file__).resolve().parent

def inspect(path):
    data = path.read_bytes()
    model = tflite.Model.GetRootAsModel(data, 0)
    graph = model.Subgraphs(0)
    names = {v: k for k, v in vars(tflite.BuiltinOperator).items() if isinstance(v, int)}
    types = {v: k for k, v in vars(tflite.TensorType).items() if isinstance(v, int)}
    def tensor(i):
        t = graph.Tensors(int(i))
        return {'index': int(i), 'name': t.Name().decode(), 'shape': t.ShapeAsNumpy().tolist(), 'type': types[t.Type()]}
    operations = []
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        operations.append({'op': names[code], 'inputs': op.InputsAsNumpy().tolist(), 'outputs': op.OutputsAsNumpy().tolist()})
    return {'file': path.name, 'sha256': hashlib.sha256(data).hexdigest(),
            'inputs': [tensor(i) for i in graph.InputsAsNumpy()],
            'outputs': [tensor(i) for i in graph.OutputsAsNumpy()],
            'operator_counts': dict(collections.Counter(op['op'] for op in operations)),
            'tensors': [tensor(i) for i in range(graph.TensorsLength())], 'operations': operations}

if __name__ == '__main__':
    result = [inspect(p) for p in sorted((ROOT / 'source').glob('*.tflite'))]
    (ROOT / 'model_structure.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for info in result:
        print(json.dumps({k:v for k,v in info.items() if k not in ('tensors', 'operations')}, indent=2))
