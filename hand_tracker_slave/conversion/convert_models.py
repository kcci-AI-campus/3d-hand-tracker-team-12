"""Official TFLite -> weight-preserving PyTorch graph -> pnnx -> ncnn.

This deliberately supports only the operators present in the pinned bundle.
Unsupported operators/options fail instead of silently approximating inference.
Conversion tools run on a development PC; none are runtime dependencies on Pi.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tflite
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent
BUNDLE_SHA256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
SOURCE_HASHES = {
    'hand_detector': '945f713bc23570bd4ed60f848c401dc8eaf95713183d43ba14cf12e467d27a7d',
    'hand_landmarks_detector': '6acda74af3fbf40e68265c20c7394b2bad81a16a481dcd79ad7a081887c3d6b9',
}
OUTPUTS = {
    'hand_detector': ['regressors', 'scores'],
    'hand_landmarks_detector': ['landmarks', 'presence', 'handedness', 'world_landmarks'],
}

def activation(code):
    if code == 0: return nn.Identity()
    if code == 1: return nn.ReLU()
    if code == 2: return nn.Hardtanh(-1, 1)
    if code == 3: return nn.ReLU6()
    raise ValueError(f'Unsupported fused activation {code}')

class Resize(nn.Module):
    def __init__(self, size, align):
        super().__init__()
        self.size, self.align = size, align
    def forward(self, x):
        return F.interpolate(x, size=self.size, mode='bilinear', align_corners=self.align)

class Pad(nn.Module):
    def __init__(self, padding, value=0.):
        super().__init__()
        self.padding, self.value = padding, value
    def forward(self, x):
        return F.pad(x, self.padding, value=self.value)

class TFLiteGraph(nn.Module):
    def __init__(self, path):
        super().__init__()
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != SOURCE_HASHES[path.stem]:
            raise ValueError('Unreviewed model: inspect its structure before updating the pinned conversion')
        m = tflite.Model.GetRootAsModel(data, 0)
        g = m.Subgraphs(0)
        self.input_id = int(g.Inputs(0))
        self.output_ids = [int(i) for i in g.OutputsAsNumpy()]
        shape = lambda i: tuple(int(n) for n in g.Tensors(int(i)).ShapeAsNumpy())
        self.input_shape = (1, 3, shape(self.input_id)[1], shape(self.input_id)[2])
        codes = {v:k for k,v in vars(tflite.BuiltinOperator).items() if isinstance(v,int)}
        constants = {}
        dtypes = {tflite.TensorType.FLOAT32: np.float32, tflite.TensorType.FLOAT16: np.float16,
                  tflite.TensorType.INT32: np.int32}
        for i in range(g.TensorsLength()):
            t = g.Tensors(i)
            buf = m.Buffers(t.Buffer())
            if buf.DataLength():
                if t.Type() not in dtypes: raise ValueError(f'Unsupported constant dtype {t.Type()}')
                constants[i] = np.frombuffer(buf.DataAsNumpy().tobytes(), dtypes[t.Type()]).reshape(shape(i)).copy()
        self.layers = nn.ModuleList()
        self.steps = []
        self.options_log = []
        for i in range(g.OperatorsLength()):
            op = g.Operators(i)
            kind = codes[m.OperatorCodes(op.OpcodeIndex()).BuiltinCode()]
            ins, outs = [int(n) for n in op.InputsAsNumpy()], [int(n) for n in op.OutputsAsNumpy()]
            if len(outs) != 1: raise ValueError('Expected one output per TFLite operator')
            output = outs[0]
            def options(name):
                obj = getattr(tflite, name)()
                obj.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
                return obj
            module, extra = nn.Identity(), None
            if kind == 'DEQUANTIZE':
                if ins[0] not in constants or constants[ins[0]].dtype != np.float16:
                    raise ValueError('Only constant FP16 -> FP32 dequantization is supported')
                constants[output] = constants[ins[0]].astype(np.float32)
                continue
            if kind in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
                opt = options('Conv2DOptions' if kind == 'CONV_2D' else 'DepthwiseConv2DOptions')
                weight = constants[ins[1]]
                stride = (opt.StrideH(), opt.StrideW())
                dilation = (opt.DilationHFactor(), opt.DilationWFactor())
                groups = shape(ins[0])[3] if kind == 'DEPTHWISE_CONV_2D' else 1
                weight = weight.transpose(3, 0, 1, 2) if groups != 1 or kind == 'DEPTHWISE_CONV_2D' else weight.transpose(0, 3, 1, 2)
                conv = nn.Conv2d(shape(ins[0])[3], weight.shape[0], weight.shape[2:], stride=stride,
                                 dilation=dilation, groups=groups, bias=len(ins) > 2 and ins[2] >= 0)
                conv.weight.data.copy_(torch.from_numpy(weight.copy()))
                if conv.bias is not None: conv.bias.data.copy_(torch.from_numpy(constants[ins[2]].copy()))
                padding = [0, 0, 0, 0]
                if opt.Padding() == tflite.Padding.SAME:
                    for axis in range(2):
                        total = max(0, (shape(output)[axis+1]-1)*stride[axis] +
                                    (weight.shape[axis+2]-1)*dilation[axis]+1-shape(ins[0])[axis+1])
                        offset = 2 if axis == 0 else 0
                        padding[offset:offset+2] = [total//2, total-total//2]
                elif opt.Padding() != tflite.Padding.VALID: raise ValueError('Unknown padding')
                module = nn.Sequential(Pad(tuple(padding)), conv, activation(opt.FusedActivationFunction()))
            elif kind == 'PRELU':
                alpha = constants[ins[1]].flatten()
                module = nn.PReLU(len(alpha))
                module.weight.data.copy_(torch.from_numpy(alpha.copy()))
            elif kind == 'ADD':
                module = activation(options('AddOptions').FusedActivationFunction())
            elif kind == 'MAX_POOL_2D':
                opt = options('Pool2DOptions')
                kernel, stride = (opt.FilterHeight(), opt.FilterWidth()), (opt.StrideH(), opt.StrideW())
                pads = [0,0,0,0]
                if opt.Padding() == tflite.Padding.SAME:
                    for axis in range(2):
                        total = max(0,(shape(output)[axis+1]-1)*stride[axis]+kernel[axis]-shape(ins[0])[axis+1])
                        offset = 2 if axis == 0 else 0
                        pads[offset:offset+2] = [total//2,total-total//2]
                module = nn.Sequential(Pad(tuple(pads), float('-inf')), nn.MaxPool2d(kernel, stride), activation(opt.FusedActivationFunction()))
            elif kind == 'PAD':
                pads = constants[ins[1]]
                if np.any(pads[0]): raise ValueError('Batch padding is unsupported')
                module = Pad(tuple(int(v) for v in np.concatenate([pads[2],pads[1],pads[3]])))
            elif kind == 'RESIZE_BILINEAR':
                opt = options('ResizeBilinearOptions')
                if not opt.AlignCorners() and not opt.HalfPixelCenters(): raise ValueError('Asymmetric resize requires separate implementation')
                module = Resize(tuple(int(v) for v in constants[ins[1]]), bool(opt.AlignCorners()))
                self.options_log.append({'operator': kind, 'align_corners': bool(opt.AlignCorners()), 'half_pixel_centers': bool(opt.HalfPixelCenters())})
            elif kind == 'RESHAPE':
                extra = (shape(output), len(shape(ins[0])) == 4)
            elif kind == 'CONCATENATION':
                opt = options('ConcatenationOptions')
                if opt.FusedActivationFunction() != 0: raise ValueError('Fused concat unsupported')
                axis = opt.Axis()
                extra = {0:0,1:2,2:3,3:1,-1:1}[axis] if len(shape(output)) == 4 else axis
            elif kind == 'MEAN':
                opt = options('ReducerOptions')
                axes = constants[ins[1]].flatten().tolist()
                if axes != [1,2]: raise ValueError(f'Unexpected mean axes {axes}')
                extra = bool(opt.KeepDims())
            elif kind == 'FULLY_CONNECTED':
                opt = options('FullyConnectedOptions')
                if opt.WeightsFormat() != 0 or opt.KeepNumDims(): raise ValueError('Unsupported FC options')
                weight = constants[ins[1]]
                fc = nn.Linear(weight.shape[1], weight.shape[0])
                fc.weight.data.copy_(torch.from_numpy(weight.copy()))
                fc.bias.data.copy_(torch.from_numpy(constants[ins[2]].copy()))
                module = nn.Sequential(fc, activation(opt.FusedActivationFunction()))
            elif kind == 'LOGISTIC': module = nn.Sigmoid()
            else: raise ValueError(f'Unsupported TFLite operator {kind}')
            self.layers.append(module)
            self.steps.append((kind, ins, output, extra))
        self.eval()

    def forward(self, x):
        values = {self.input_id: x}
        for layer, (kind, ins, output, extra) in zip(self.layers, self.steps):
            a = values[ins[0]]
            if kind == 'ADD': result = layer(a + values[ins[1]])
            elif kind == 'RESHAPE':
                target, from_nhwc = extra
                result = (a.permute(0,2,3,1) if from_nhwc else a).reshape(target)
            elif kind == 'CONCATENATION': result = torch.cat([values[i] for i in ins], dim=extra)
            elif kind == 'MEAN': result = a.mean(dim=(2,3), keepdim=extra)
            elif kind == 'FULLY_CONNECTED': result = layer(a.flatten(1))
            else: result = layer(a)
            values[output] = result
        return tuple(values[i] for i in self.output_ids)

def convert(name, work, destination, pnnx_path):
    path = ROOT / 'source' / (name + '.tflite')
    graph = TFLiteGraph(path)
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    sample = rng.random(graph.input_shape, dtype=np.float32)
    interpreter = Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], sample.transpose(0,2,3,1).copy())
    interpreter.invoke()
    reference = [interpreter.get_tensor(d['index']) for d in interpreter.get_output_details()]
    with torch.no_grad(): outputs = graph(torch.from_numpy(sample))
    errors = {}
    for label, expected, actual in zip(OUTPUTS[name], reference, outputs):
        actual = actual.numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
        errors[label] = {'max_abs':float(np.max(np.abs(actual-expected))), 'shape':list(actual.shape)}
    print(name, 'TFLite/PyTorch', errors, flush=True)
    traced = torch.jit.trace(graph, torch.from_numpy(sample), check_trace=True)
    pt = work / (name + '.pt')
    traced.save(str(pt))
    command = [str(pnnx_path), str(pt), f'inputshape=[{",".join(str(n) for n in graph.input_shape)}]', 'fp16=0',
               f'ncnnparam={destination / (name + ".param")}', f'ncnnbin={destination / (name + ".bin")}',
               f'pnnxparam={work / (name + ".pnnx.param")}', f'pnnxbin={work / (name + ".pnnx.bin")}',
               f'pnnxpy={work / (name + "_pnnx.py")}', f'ncnnpy={work / (name + "_ncnn.py")}']
    with (work / (name + '_pnnx.log')).open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=work)
    param = destination / (name + '.param')
    names = {'in0':'input', **{f'out{i}':label for i,label in enumerate(OUTPUTS[name])}}
    lines = [' '.join(names.get(token,token) for token in line.split()) for line in param.read_text().splitlines()]
    fixes = []
    if name == 'hand_detector':
        # pnnx 20260704 retains the singleton batch as c=1 in these four
        # [1, anchors, values] heads. ncnn 20260526 has no batch axis: heads
        # must be 2D [anchors, values] for Concat(axis=0) to join anchor rows.
        expected = {(1, 864), (18, 864), (1, 1152), (18, 1152)}
        for i, line in enumerate(lines):
            tokens = line.split()
            if not tokens or tokens[0] != 'Reshape': continue
            params = dict(token.split('=', 1) for token in tokens[6:])
            shape = (int(params['0']), int(params['1']))
            if shape not in expected or params != {'0':str(shape[0]), '1':str(shape[1]), '12':'0', '13':'0', '2':'1'}:
                raise RuntimeError(f'Unreviewed palm reshape from pnnx: {line}')
            expected.remove(shape)
            lines[i] = ' '.join(tokens[:6] + [f'0={shape[0]}', f'1={shape[1]}'])
            fixes.append({'layer':tokens[1], 'width':shape[0], 'height':shape[1], 'removed_singleton_batch':True})
        if expected: raise RuntimeError(f'Missing palm reshape heads: {expected}')
    param.write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    return {'tflite_to_torch':errors, 'options':graph.options_log, 'ncnn_reshape_fixes':fixes}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(OUTPUTS), action='append')
    parser.add_argument('--pnnx', type=Path, required=True, help='Official pnnx 20260704 release executable')
    args = parser.parse_args()
    args.pnnx = args.pnnx.resolve(strict=True)
    work = ROOT / 'work'
    destination = ROOT / 'converted'
    work.mkdir(exist_ok=True)
    destination.mkdir(exist_ok=True)
    results = {name:convert(name,work,destination,args.pnnx) for name in (args.model or list(OUTPUTS))}
    results['_toolchain'] = {'pnnx_release':'20260704', 'pnnx_sha256':hashlib.sha256(args.pnnx.read_bytes()).hexdigest()}
    (ROOT / 'conversion_report.json').write_text(json.dumps(results,indent=2))
