"""Synthetic inputs shared by the hand_tracking tests (and .tools review scripts)."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('Install requirements-transformer.txt for model tests')
import torch
from hand_tracking.config import ModelConfig
from hand_tracking.contracts import MODEL_INPUT_KEYS
from hand_tracking.data import read_clip

# Camera centres of the synthetic three-camera rig (world units).
ORIGINS = torch.tensor([[-1.,-1.,.5], [1.,-1.,.5], [0.,1.,1.]])


def small(**overrides):
    """A fast model: width 32, 4 heads, 1 block, 1 encoder layer, no dropout."""
    return ModelConfig(**{**dict(dim=32, heads=4, blocks=1, encoder_layers=1, dropout=0.), **overrides})


def randomize_heads(model, std=.1):
    """Give the zero-initialised residual heads weights, so outputs depend on the inputs."""
    heads = [model.output] + ([model.calibration_output] if model.config.calibration_head else [])
    for head in heads:
        for parameter in head[-1].parameters():
            torch.nn.init.normal_(parameter, std=std)
    return model


def synthetic_events(position, captures, cameras, delay=.06):
    """Exact rays from each event's camera to position(capture) [2,21,3]; batch of one,
    sorted by arrival. Returns the model inputs without the query times."""
    captures = torch.as_tensor(captures, dtype=torch.float32)
    cameras = torch.as_tensor(cameras)
    arrivals = captures+delay
    order = torch.argsort(arrivals)
    captures, cameras, arrivals = captures[order], cameras[order], arrivals[order]
    events = len(captures)
    features = torch.zeros(1, events, 2, 21, 14)
    for i, (time, camera) in enumerate(zip(captures, cameras)):
        direction = position(time)-ORIGINS[camera]
        features[0,i,...,2:5] = ORIGINS[camera]
        features[0,i,...,5:8] = direction/direction.norm(dim=-1, keepdim=True)
        features[0,i,...,10] = delay
    return (features, torch.ones(1, events, 2, 21, dtype=torch.bool), cameras[None], captures[None], arrivals[None],
            torch.ones(1, events, dtype=torch.bool))


def asynchronous(duration=.6, rate=17.1, phases=(0.,.02,.04)):
    """Capture times and cameras of three free-running cameras with phase offsets."""
    captures, cameras = [], []
    for camera, phase in enumerate(phases):
        for time in np.arange(-duration+phase, 0, 1/rate):
            captures.append(time)
            cameras.append(camera)
    return captures, cameras


def linear_events(ages=(.02,.10,.18), frames=16, rate=30):
    """One event per camera and query frame, captured `ages` before it; the point moves
    linearly. Returns (inputs without query times, query times [Q], true positions [Q,3])."""
    base = torch.tensor([.1,-.2,.05])
    velocity = torch.tensor([1.,.3,0.])
    query = torch.arange(frames)/rate-(frames-1)/rate
    capture = (query[:,None]-torch.tensor(ages)).reshape(-1)
    camera = torch.arange(3).repeat(frames)
    arrival = query.repeat_interleave(3)-.005
    order = torch.argsort(arrival, stable=True)
    capture, camera, arrival = capture[order], camera[order], arrival[order]
    rays = torch.nn.functional.normalize(base+capture[:,None]*velocity-ORIGINS[camera], dim=-1)
    features = torch.zeros(1, len(camera), 2, 21, 14)
    features[0,...,2:5] = ORIGINS[camera][:,None,None]
    features[0,...,5:8] = rays[:,None,None]
    valid = torch.ones(features.shape[:-1], dtype=torch.bool)
    inputs = (features, valid, camera[None], capture[None], arrival[None], torch.ones(1, len(camera), dtype=torch.bool))
    return inputs, query, base+query[:,None]*velocity


def sim_clip(folder, **overrides):
    """Simulate the 30-frame demo motion into folder/a.npz and read it back as a clip."""
    from gigahands_sim import Config, demo_motion, simulate, save_dataset
    positions, times = demo_motion(30)
    config = Config()
    for key, value in overrides.items():
        setattr(config, key, value)
    save_dataset(Path(folder)/'a.npz', simulate(positions, times, config))
    return read_clip(Path(folder)/'a.npz')


def batch_of(sample):
    """A WindowSample as a batch of one, in model input order."""
    return [sample[key][None] for key in MODEL_INPUT_KEYS]


def push_all(stream, inputs, indices=None):
    """Push events of a batch-of-one input tuple into an EventStream; returns push results."""
    features, valid, camera, capture, arrival, _ = inputs
    indices = range(features.shape[1]) if indices is None else indices
    return [stream.push(int(camera[0,i]), features[0,i], valid[0,i], float(capture[0,i]), float(arrival[0,i]))
            for i in indices]
