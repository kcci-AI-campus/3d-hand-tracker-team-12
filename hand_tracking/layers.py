"""Network building blocks shared by the models: attention with an additive key bias,
pre-norm blocks and the per-event finger-token encoder. Linear/LayerNorm/GELU/MatMul/
Softmax only and tensors of at most 4 dimensions, so the graphs export to ONNX and ncnn."""
import torch
from torch import nn
import torch.nn.functional as F
from .constants import NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, FINGERS, FINGER_JOINTS, MASK_OFF


def key_bias(valid):
    """Boolean key validity -> additive attention bias (0 or MASK_OFF)."""
    return (valid.float()-1.)*-MASK_OFF


class Attention(nn.Module):
    """Multi-head attention with an additive bias; 4-D tensors at most (ncnn limit)."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.query, self.key, self.value, self.out = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, x, keys, bias=None):
        """x [N,L,d], keys [N,S,d], bias broadcastable to [N,heads,L,S] -> [N,L,d]."""
        n, l, d = x.shape
        s, h = keys.shape[1], self.heads
        q = self.query(x).reshape(n, l, h, d//h).permute(0, 2, 1, 3)
        k = self.key(keys).reshape(n, s, h, d//h).permute(0, 2, 3, 1)
        v = self.value(keys).reshape(n, s, h, d//h).permute(0, 2, 1, 3)
        scores = (q @ k)*(d//h)**-.5
        if bias is not None:
            scores = scores+bias
        weights = F.dropout(scores.softmax(-1), self.dropout, self.training)
        return self.out((weights @ v).permute(0, 2, 1, 3).reshape(n, l, d))


class Block(nn.Module):
    """Pre-norm: optional cross-attention to keys, self-attention, feed-forward. dropout on
    the residual branches, attention_dropout on the attention weights."""

    def __init__(self, dim, heads, dropout, cross=True, attention_dropout=0.):
        super().__init__()
        self.cross = Attention(dim, heads, attention_dropout) if cross else None
        if cross:
            self.norm_query, self.norm_keys = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.norm_self, self.norm_ff = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, attention_dropout)
        self.feedforward = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, keys=None, bias=None, self_bias=None):
        """bias masks the cross-attention keys, self_bias the self-attention ones."""
        if self.cross is not None:
            x = x+self.drop(self.cross(self.norm_query(x), self.norm_keys(keys), bias))
        y = self.norm_self(x)
        x = x+self.drop(self.attention(y, y, self_bias))
        return x+self.drop(self.feedforward(self.norm_ff(x)))


class EventEncoder(nn.Module):
    """features [N,2,21*F], camera one-hot [N,3] -> tokens [N,2*5,dim] (hand-major, one per
    finger) with finger_tokens, else [N,2,dim] (one per hand).

    Finger tokens share one MLP over the 5 joints (wrist and finger) of a group; a learned
    finger embedding inside the MLP lets each finger be read differently. Groups are taken
    with a constant 0/1 selection matrix (MatMul), which ncnn runs without Gather."""

    def __init__(self, dim, finger_tokens=True, joint_features=14):
        super().__init__()
        self.fingers, self.joint_features = finger_tokens, joint_features
        if finger_tokens:
            group = len(FINGER_JOINTS[0])
            select = torch.zeros(FINGERS*group, NUM_JOINTS)
            select[torch.arange(FINGERS*group), torch.tensor(FINGER_JOINTS).flatten()] = 1.
            self.register_buffer('select', select, persistent=False)
            self.joints = nn.Sequential(nn.Linear(group*joint_features, dim), nn.GELU(), nn.Linear(dim, dim))
            self.finger = nn.Parameter(torch.randn(FINGERS, dim)*.02)
        else:
            self.joints = nn.Sequential(nn.Linear(NUM_JOINTS*joint_features, dim), nn.GELU(), nn.Linear(dim, dim))
        self.hand = nn.Parameter(torch.randn(NUM_HANDS, dim)*.02)
        self.camera = nn.Linear(NUM_CAMERAS, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features, camera):
        if not self.fingers:
            return self.norm(self.joints(features)+self.hand+self.camera(camera)[:, None])
        n = features.shape[0]
        groups = self.select@features.reshape(n, NUM_HANDS, NUM_JOINTS, self.joint_features)   # [N,2,5*5,F]
        groups = groups.reshape(n, NUM_HANDS, FINGERS, -1)                                     # [N,2,5,5*F]
        x = self.joints[2](self.joints[1](self.joints[0](groups)+self.finger))
        x = x+self.hand[:, None]+self.camera(camera)[:, None, None]
        return self.norm(x).reshape(n, NUM_HANDS*FINGERS, -1)
