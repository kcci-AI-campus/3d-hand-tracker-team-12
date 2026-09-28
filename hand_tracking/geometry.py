"""Differentiable multi-view ray geometry: triangulation (optionally dropping one outlier
ray) and camera corrections. No learned parameters."""
import torch
from .constants import MIN_RAY_SIN2, RAY_RESIDUAL_FLOOR, OUTLIER_MARGIN


def cross3(a, b):
    """Cross product of [...,3] vectors (torch.linalg.cross has no ONNX operator)."""
    a0, a1, a2 = a.unbind(-1); b0, b1, b2 = b.unbind(-1)
    return torch.stack((a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0), -1)


def solve3(a, b):
    """a [...,3,3] invertible, b [...,3] -> a^-1 b by the adjugate (Cramer's rule).

    Exact algebra for 3x3 and exportable (torch.linalg.solve has no ONNX operator);
    callers substitute the identity for singular systems, as for linalg.solve.
    """
    r0, r1, r2 = a.unbind(-2)
    c0, c1, c2 = cross3(r1, r2), cross3(r2, r0), cross3(r0, r1)
    det = (r0*c0).sum(-1, keepdim=True)
    return (b[...,0:1]*c0+b[...,1:2]*c1+b[...,2:3]*c2)/det


def _rays(origins, directions, mask):
    """Usable rays: mask, unit directions and origins (zero where unusable), projections
    I - d d^T [...,C,3,3] (zero where unusable) and pairwise sin^2 of ray angles [...,C,C]."""
    mask = (mask.bool() & torch.isfinite(origins).all(-1) & torch.isfinite(directions).all(-1)
            & (directions.norm(dim=-1) > 1e-8))
    d = torch.where(mask[...,None], directions, 0.); o = torch.where(mask[...,None], origins, 0.)
    d = d/d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    eye = torch.eye(3, dtype=d.dtype, device=d.device)
    projection = (eye - d[...,:,None]*d[...,None,:])*mask[...,None,None]
    # Widest pairwise ray angle instead of eigvalsh: batched CUDA eigh can fail to
    # converge on the empty/single-ray matrices (repeated eigenvalues) present here.
    pair = cross3(d[...,:,None,:], d[...,None,:,:]).square().sum(-1)
    return mask, d, o, projection, pair


def _point(mask, d, o, projection, pair):
    """Least-squares point of the rays in mask (a subset of _rays' rays) and its validity."""
    used = projection*mask[...,None,None]
    return _solve(used.sum(-3), (used @ o[...,None]).sum(-3).squeeze(-1), mask, d, o, pair)


def _solve(a, rhs, mask, d, o, pair):
    """Point from the normal equations a x = rhs of the rays in mask; d, o, pair may
    broadcast over leading axes of mask (leave-one-out problems share them)."""
    eye = torch.eye(3, dtype=d.dtype, device=d.device)
    sin2 = (pair*(mask[...,:,None] & mask[...,None,:])).amax((-2,-1))
    valid = (mask.sum(-1) >= 2) & (sin2 > MIN_RAY_SIN2)
    point = solve3(torch.where(valid[...,None,None], a, eye), rhs)
    # A point behind any contributing camera means inconsistent rays (outliers), not a hand.
    depth = ((point[...,None,:]-o)*d).sum(-1)
    valid = valid & ((depth > 0) | ~mask).all(-1)
    return torch.where(valid[...,None], point, 0.), valid


def triangulate(origins, directions, mask):
    """Least-squares point closest to the masked rays.

    origins/directions [...,C,3], mask [...,C] -> point [...,3], valid [...].
    Needs two or more non-parallel rays with positive depths; otherwise point is 0 and
    valid is False.
    """
    return _point(*_rays(origins, directions, mask))


def ray_residuals(point, origins, directions):
    """Angular residual (miss distance / depth) of each ray [...,C] to point [...,3]."""
    d = directions/directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    relative = point[...,None,:]-origins
    depth = (relative*d).sum(-1)
    return (relative-depth[...,None]*d).norm(dim=-1)/depth.clamp_min(1e-6)


def robust_triangulate(origins, directions, mask, outlier_ratio):
    """triangulate, rejecting at most one outlier ray among three or more.

    For each ray k: e = its residual to the point of the other rays, s = their largest
    residual to that point. The ray with the largest e/max(s, RAY_RESIDUAL_FLOOR) is
    dropped when that ratio exceeds outlier_ratio and OUTLIER_MARGIN times the next ray's:
    the other rays agree and it does not. When the next ray's ratio is within that margin
    the sample is ambiguous and invalid: a ray displaced within the epipolar plane of
    another camera agrees with that camera, so pairs (0,1) and (0,2) can both agree while
    (1,2) do not, and geometry cannot tell whether 1 or 2 is wrong. A camera whose
    calibration is off looks like an outlier, so apply this to corrected rays.
    The leave-one-out points share the rays' preparation and are solved in one batch.
    Returns point, valid and the rays used [...,C].
    """
    mask, d, o, projection, pair = _rays(origins, directions, mask)
    # Normal equations of all rays, and of all but ray k by removing that ray's term
    # (projection is zero for unusable rays, so their "removal" changes nothing).
    term = (projection @ o[...,None]).squeeze(-1)                                        # [...,C,3]
    a, rhs = projection.sum(-3), term.sum(-2)
    point, valid = _solve(a, rhs, mask, d, o, pair)
    count = mask.shape[-1]
    others = ~torch.eye(count, dtype=torch.bool, device=mask.device)                    # [K,C]: all but k
    keep = mask[...,None,:] & others                                                    # [...,K,C]
    loo_point, loo_valid = _solve(a[...,None,:,:]-projection, rhs[...,None,:]-term, keep, d[...,None,:,:],
                                  o[...,None,:,:], pair[...,None,:,:])
    relative = loo_point[...,None,:]-o[...,None,:,:]                                    # [...,K,C,3]
    along = (relative*d[...,None,:,:]).sum(-1)
    residual = (relative-along[...,None]*d[...,None,:,:]).norm(dim=-1)/along.clamp_min(1e-6)   # [...,K,C]
    spread = torch.where(keep, residual, 0.).amax(-1)
    own = residual.diagonal(dim1=-2, dim2=-1)                                           # [...,K]
    ratio = torch.where(loo_valid & mask & (mask.sum(-1, keepdim=True) >= 3), own/spread.clamp_min(RAY_RESIDUAL_FLOOR), 0.)
    top = ratio.topk(min(2, count), dim=-1)
    best, dropped = top.values[...,0], top.indices[...,0]
    second = top.values[...,1] if count > 1 else torch.zeros_like(best)
    pick = lambda value: value.gather(-1, dropped[...,None]).squeeze(-1)
    best_point = loo_point.gather(-2, dropped[...,None,None].expand(*dropped.shape, 1, 3)).squeeze(-2)
    suspect = best > outlier_ratio
    ambiguous = suspect & (best < OUTLIER_MARGIN*second)
    reject = suspect & ~ambiguous
    valid = torch.where(reject, pick(loo_valid), valid) & ~ambiguous
    point = torch.where(valid[...,None], torch.where(reject[...,None], best_point, point), 0.)
    return point, valid, mask & ~(reject[...,None] & (torch.arange(count, device=mask.device) == dropped[...,None]))


def _sinc(theta, theta2, small):
    """sin(theta)/theta, Taylor below the `small` threshold (finite value and gradient at 0)."""
    safe = torch.where(small, torch.ones_like(theta), theta)
    return torch.where(small, 1-theta2/6, torch.sin(safe)/safe)


def rodrigues(w):
    """Rotation vectors [...,3] -> matrices [...,3,3]; smooth and differentiable at zero.

    R = I + sinc(t) [w]x + (1-cos t)/t^2 [w]x^2 with (1-cos t)/t^2 = sinc(t/2)^2/2, which
    has no float32 cancellation. No epsilon inside a sqrt: ONNX export folded it away.
    """
    theta2 = w.square().sum(-1)[...,None,None]
    small = theta2 < 1e-8
    theta = torch.where(small, torch.ones_like(theta2), theta2).sqrt()
    a = _sinc(theta, theta2, small)
    b = .5*_sinc(theta/2, theta2/4, small).square()
    zero = torch.zeros_like(w[...,0])
    skew = torch.stack((zero,-w[...,2],w[...,1], w[...,2],zero,-w[...,0], -w[...,1],w[...,0],zero), -1)
    skew = skew.reshape(*w.shape[:-1],3,3)
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    return eye+a*skew+b*(skew@skew)


def apply_calibration(params, origins, directions):
    """params [B,3,6] (rotation vector, translation per camera); rays [B,T,3,2,21,3].

    Rotates each camera's rays about its centre and shifts the centre, both in world axes.
    """
    rotation = rodrigues(params[...,:3])
    origins = origins+params[:,None,:,None,None,3:]
    directions = torch.einsum('bcxy,btchjy->btchjx', rotation, directions)
    return origins, directions


