"""Differentiable ray geometry: least-squares triangulation and ray-to-point residuals. No learned parameters."""
import torch
from .constants import MIN_RAY_SIN2


def cross3(a, b):
    """Cross product of [...,3] vectors (torch.linalg.cross has no ONNX operator)."""
    a0, a1, a2 = a.unbind(-1)
    b0, b1, b2 = b.unbind(-1)
    return torch.stack((a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0), -1)


def solve3(a, b):
    """a [...,3,3] invertible, b [...,3] -> a^-1 b by the adjugate (Cramer's rule): exact for 3x3
    and exportable; callers substitute the identity for singular systems."""
    r0, r1, r2 = a.unbind(-2)
    c0, c1, c2 = cross3(r1, r2), cross3(r2, r0), cross3(r0, r1)
    det = (r0*c0).sum(-1, keepdim=True)
    return (b[..., 0:1]*c0+b[..., 1:2]*c1+b[..., 2:3]*c2)/det


def triangulate(origins, directions, mask):
    """Least-squares point closest to the masked rays: origins/directions [...,C,3], mask [...,C]
    -> point [...,3], valid [...]. Needs two or more rays at least MIN_RAY_SIN2 from parallel and a
    point in front of every used camera; otherwise the point is 0 and valid is False."""
    mask = (mask.bool() & torch.isfinite(origins).all(-1) & torch.isfinite(directions).all(-1)
            & (directions.norm(dim=-1) > 1e-8))
    d = torch.where(mask[..., None], directions, 0.)
    o = torch.where(mask[..., None], origins, 0.)
    d = d/d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    eye = torch.eye(3, dtype=d.dtype, device=d.device)
    projection = (eye-d[..., :, None]*d[..., None, :])*mask[..., None, None]                 # I - d d^T per ray
    a = projection.sum(-3)
    rhs = (projection@o[..., None]).squeeze(-1).sum(-2)
    # Widest pairwise ray angle instead of eigvalsh (batched eigh can fail on the empty or
    # single-ray matrices present here).
    pair = cross3(d[..., :, None, :], d[..., None, :, :]).square().sum(-1)
    sin2 = (pair*(mask[..., :, None] & mask[..., None, :])).amax((-2, -1))
    valid = (mask.sum(-1) >= 2) & (sin2 > MIN_RAY_SIN2)
    point = solve3(torch.where(valid[..., None, None], a, eye), rhs)
    # A point behind any contributing camera means inconsistent rays, not a hand.
    depth = ((point[..., None, :]-o)*d).sum(-1)
    valid = valid & ((depth > 0) | ~mask).all(-1)
    return torch.where(valid[..., None], point, 0.), valid


def ray_residuals(point, origins, directions):
    """Angular residual (miss distance / depth) of each ray [...,C] to point [...,3]; huge for a
    point behind the camera."""
    d = directions/directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    relative = point[..., None, :]-origins
    depth = (relative*d).sum(-1)
    return (relative-depth[..., None]*d).norm(dim=-1)/depth.clamp_min(1e-6)
