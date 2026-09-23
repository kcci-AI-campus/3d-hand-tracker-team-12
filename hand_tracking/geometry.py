"""Differentiable multi-view ray geometry: triangulation, camera corrections and the
optional constant-velocity fit to individual rays. No learned parameters."""
import torch
import torch.nn.functional as F
from .constants import REPEAT_TOLERANCE_S, MIN_TIME_STD_S, MAX_EXTRAPOLATION_S, MIN_RAY_SIN2, TIME_UNIT_S
# Robust motion fit (anchor_motion_fit) residual limits, dimensionless distance/depth.
HUBER_RAY_RESIDUAL = .02
MAX_MOTION_RESIDUAL = .08
MAX_MOTION_RMS = .005


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


def triangulate(origins, directions, mask, max_residual=None):
    """Least-squares point closest to the masked rays.

    origins/directions [...,C,3], mask [...,C] -> point [...,3], valid [...].
    Needs two or more non-parallel rays with positive depths, and with max_residual
    every ray's angular residual within it; otherwise point is 0 and valid is False.
    """
    mask = (mask.bool() & torch.isfinite(origins).all(-1) & torch.isfinite(directions).all(-1)
            & (directions.norm(dim=-1) > 1e-8))
    d = torch.where(mask[...,None], directions, 0.); o = torch.where(mask[...,None], origins, 0.)
    d = d/d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    eye = torch.eye(3, dtype=d.dtype, device=d.device)
    projection = (eye - d[...,:,None]*d[...,None,:])*mask[...,None,None]
    a = projection.sum(-3)
    rhs = (projection @ o[...,None]).sum(-3)
    # Widest pairwise ray angle instead of eigvalsh: batched CUDA eigh can fail to
    # converge on the empty/single-ray matrices (repeated eigenvalues) present here.
    sin2 = cross3(d[...,:,None,:], d[...,None,:,:]).square().sum(-1).amax((-2,-1))
    valid = (mask.sum(-1) >= 2) & (sin2 > MIN_RAY_SIN2)
    point = solve3(torch.where(valid[...,None,None], a, eye), rhs.squeeze(-1))
    # A point behind any contributing camera means inconsistent rays (outliers), not a hand.
    depth = ((point[...,None,:]-o)*d).sum(-1)
    valid = valid & ((depth > 0) | ~mask).all(-1)
    if max_residual is not None:
        miss = point[...,None,:]-o-depth[...,None]*d
        residual = miss.norm(dim=-1)/depth.clamp_min(1e-6)
        valid = valid & ((residual <= max_residual) | ~mask).all(-1)
    return torch.where(valid[...,None], point, 0.), valid


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


def _calibration_residuals(params, origins, directions, mask, noise, rotation_std, position_std):
    """Whitened ray-to-point angular residuals plus prior residuals for one window."""
    o, d = apply_calibration(params[None], origins[None], directions[None])
    o, d = o[0].permute(0,2,3,1,4), d[0].permute(0,2,3,1,4)          # [T,2,21,C,3]
    m = mask.permute(0,2,3,1)
    point, ok = triangulate(o, d, m)
    relative = point[...,None,:]-o
    depth = (relative*d).sum(-1, keepdim=True)
    residual = (relative-d*depth)/depth.clamp_min(1e-3)
    residual = torch.where((m & ok[...,None])[...,None], residual, 0.)/noise
    prior = torch.cat(((params[:,:3]/rotation_std).flatten(), (params[:,3:]/position_std).flatten()))
    return torch.cat((residual.flatten(), prior))


def calibrate_cameras(origins, directions, mask, steps, rotation_std, position_std, noise, initial=None):
    """MAP per-camera extrinsic correction from multi-view ray consistency, per window.

    origins/directions [B,T,3,2,21,3], mask [B,T,3,2,21]. Only relative camera errors are
    observable; the Gaussian prior keeps the common (unobservable) rig motion at the
    nominal calibration. Returns params [B,3,6]. Model-free baseline.
    """
    b = origins.shape[0]
    params = origins.new_zeros(b,3,6) if initial is None else initial.to(origins.dtype)
    if steps <= 0: return params
    def residuals(p, o, d, m):
        return _calibration_residuals(p, o, d, m, noise, rotation_std, position_std)
    evaluate = torch.func.vmap(residuals)
    jacobian = torch.func.vmap(torch.func.jacfwd(residuals))
    for _ in range(steps):
        r = evaluate(params, origins, directions, mask)
        j = jacobian(params, origins, directions, mask).reshape(b, r.shape[1], 18)
        # The prior rows make J^T J positive definite, so plain Gauss-Newton is stable.
        # Data rows (1/noise^2) outweigh the prior by ~1e6: accumulate and solve in float64.
        j, r = j.double(), r.double(); jt = j.transpose(1,2)
        step = torch.linalg.solve(jt@j, jt@r[...,None]).squeeze(-1)
        params = params-step.reshape(b,3,6).to(params.dtype)
    return params


def motion_triangulate(origins, directions, valid, capture, query, span, lookback_s):
    """Fit p + v * capture_time directly to individual rays, per query (anchor_motion_fit).

    origins/directions [B,S,C,H,J,3], valid/capture [B,S,C,H,J] (capture on the query's
    time origin), query [B,Q], span [B,Q,S] rays each query may use. Repeated captures
    share one unit of weight; Huber IRLS then rejection/refit handles outliers.
    Underconstrained or inconsistent fits return valid=False. Returns position [B,Q,H,J,3],
    validity and the latest accepted ray's age [B,Q,H,J].
    """
    order = (0,1,3,4,2,5)
    o, d = origins.permute(order), directions.permute(order)  # B,S,H,J,C,3
    m = valid.permute(0,1,3,4,2)
    stamp = capture.permute(0,1,3,4,2)
    finite = torch.isfinite(o).all(-1) & torch.isfinite(d).all(-1) & torch.isfinite(stamp)
    m = m & finite & (d.norm(dim=-1) > 1e-8)
    o, d = torch.where(m[...,None], o, 0.), torch.where(m[...,None], d, 0.)
    d = F.normalize(d, dim=-1)
    stamp = torch.where(m, stamp, 0.)
    age = query[:,:,None,None,None,None]-stamp[:,None]
    use = m[:,None] & span[:,:,:,None,None,None] & (age >= -REPEAT_TOLERANCE_S) & (age <= lookback_s)
    same = ((stamp[:,:,None]-stamp[:,None]).abs() <= REPEAT_TOLERANCE_S) & m[:,:,None] & m[:,None]
    count = torch.einsum('bsuhjc,btuhjc->btshjc', same.float(), use.float())
    base_weight = use.float()/count.clamp_min(1)
    eye3 = torch.eye(3, device=o.device, dtype=o.dtype)
    projection = eye3-d[..., :,None]*d[...,None,:]
    rhs = (projection @ o[...,None]).squeeze(-1)
    # Centre and scale time for the 6x6 system; v below is displacement per TIME_UNIT_S.
    total = base_weight.sum((2,5))
    mean_age = (base_weight*age).sum((2,5))/total.clamp_min(1)
    tau = (mean_age[:,:,None,:,:,None]-age)/TIME_UNIT_S
    eye6 = torch.eye(6, device=o.device, dtype=o.dtype)

    def fit(weight):
        a00 = torch.einsum('btshjc,bshjcxy->bthjxy', weight, projection)
        a01 = torch.einsum('btshjc,bshjcxy->bthjxy', weight*tau, projection)
        a11 = torch.einsum('btshjc,bshjcxy->bthjxy', weight*tau.square(), projection)
        matrix = torch.cat((torch.cat((a00,a01), -1), torch.cat((a01,a11), -1)), -2)
        b0 = torch.einsum('btshjc,bshjcx->bthjx', weight, rhs)
        b1 = torch.einsum('btshjc,bshjcx->bthjx', weight*tau, rhs)
        # Cholesky checks rank without differentiating eigenvectors at repeated
        # eigenvalues. Reject poor pivots before the differentiable solve.
        with torch.no_grad():
            factor, info = torch.linalg.cholesky_ex(matrix.detach())
            pivots = factor.diagonal(dim1=-2, dim2=-1).square()
            cameras = (weight.sum(2) > 0).sum(-1)
            evidence = torch.where(weight > 0, base_weight, 0.).sum((2,5))
            temporal_weight = weight.sum((2,5)).clamp_min(1e-8)
            centre = (weight*tau).sum((2,5))/temporal_weight
            variance = (weight*(tau-centre[:,:,None,:,:,None]).square()).sum((2,5))/temporal_weight
            good = ((info == 0) & (evidence >= 4-1e-5) & (cameras >= 2)
                    & (variance*TIME_UNIT_S**2 > MIN_TIME_STD_S**2)
                    & (pivots.amin(-1) > 1e-4*pivots.amax(-1).clamp_min(1e-8)))
        solution = torch.linalg.solve(torch.where(good[...,None,None], matrix, eye6),
                                      torch.cat((b0,b1), -1)[...,None]).squeeze(-1)
        return solution, good

    def residual(solution):
        position = solution[:,:,None,:,:,None,:3] + tau[...,None]*solution[:,:,None,:,:,None,3:]
        relative = position-o[:,None]
        depth = (relative*d[:,None]).sum(-1)
        angular = (relative-depth[...,None]*d[:,None]).norm(dim=-1)/depth.clamp_min(1e-6)
        return angular, depth

    weight = base_weight
    for _ in range(3):
        solution, good = fit(weight)
        with torch.no_grad():
            angular, depth = residual(solution)
            robust = (HUBER_RAY_RESIDUAL/angular.clamp_min(1e-6)).clamp(max=1)
            weight = base_weight*robust*(depth > 0)
    with torch.no_grad():
        accepted = (angular <= MAX_MOTION_RESIDUAL) & (depth > 0)
        weight = base_weight*accepted
    solution, good = fit(weight)
    with torch.no_grad():
        angular, depth = residual(solution)
        consistent = ((angular <= MAX_MOTION_RESIDUAL) & (depth > 0)) | (weight == 0)
        rms = ((weight*angular.square()).sum((2,5))/weight.sum((2,5)).clamp_min(1)).sqrt()
        # Installation errors or non-linear motion must not be explained as a large
        # spurious velocity. Use the motion model only when its inliers agree well.
        good = good & consistent.all(2).all(-1) & (rms <= MAX_MOTION_RMS) & torch.isfinite(solution).all(-1)
    horizon = mean_age.clamp(min=0, max=MAX_EXTRAPOLATION_S)/TIME_UNIT_S
    point = solution[...,:3]+solution[...,3:]*horizon[...,None]
    latest_age = torch.where(weight > 0, age, float('inf')).amin((2,5)).clamp_min(0)
    return torch.where(good[...,None], point, 0.), good, torch.where(good, latest_age, 0.)
