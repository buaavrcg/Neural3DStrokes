import torch
import torch.nn as nn
import collections
import numpy as np
from source import configs
from source.utils import stepfun
import random

# _cost_matric_cache = {}
# def _cost_matrix(batch_size, h, w, device : torch.device):
#     key = (batch_size, h, w, device)
#     if key in _cost_matric_cache:
#         c = _cost_matric_cache[key]
#     else:
#         a = torch.linspace(0.0, h - 1.0, h, device=device)
#         b = torch.linspace(0.0, w - 1.0, w, device=device)
#         y_grid = a.view(-1, 1).repeat(batch_size, 1, w) / h
#         x_grid = b.view(1, -1).repeat(batch_size, h, 1) / w
#         grids = torch.cat([y_grid.view(batch_size, -1, 1), x_grid.view(batch_size, -1, 1)], dim=-1)

#         x_col = grids.unsqueeze(2)
#         y_lin = grids.unsqueeze(1)
#         p = 2
#         # Returns the matrix of $|x_i-y_j|^p$.
#         c = torch.sum((torch.abs(x_col - y_lin))**p, -1)
#         _cost_matric_cache[key] = c
#     return c


def point_point_distance(points1, points2):
    """
    Calculate the minimal distance between points1 and points2.
    Args:
        points1: (..., 3), the points1
        points2: (..., 3), the points2
    Returns:
        dist: (...), the minimal distance between points1 and points2
    """
    dist = torch.sqrt(torch.sum(torch.square(points1 - points2), dim=-1))
    return dist


def point_segment_distance(points, segment_A, segment_B):
    """
    Calculate the minimal distance between points and line segments.
    Args:
        points: (..., 3), the points
        segment_A: (..., 3), the start point of the line segment
        segment_B: (..., 3), the end point of the line segment
    Returns:
        dist: (...), the minimal distance between points and line segments
    """
    AP = points - segment_A
    BP = points - segment_B
    AB = segment_B - segment_A
    AP_dot_AB = torch.sum(AP * AB, dim=-1)
    AB_square = torch.sum(torch.square(AB), dim=-1)
    AP_square = torch.sum(torch.square(AP), dim=-1)
    BP_square = torch.sum(torch.square(BP), dim=-1)
    t = AP_dot_AB / AB_square
    dist = torch.sqrt(torch.clamp(AP_square - t * AP_dot_AB, min=1e-8))
    distA = torch.sqrt(AP_square)
    distB = torch.sqrt(BP_square)
    dist = torch.where(torch.logical_and(0 < t, t < 1), dist, torch.minimum(distA, distB))
    return dist


def segment_segment_distance(segment1_A, segment1_B, segment2_A, segment2_B):
    """
    Calculate the minimal distance between two line segments.
    Args:
        segment1_A: (..., 3), the start point of the first line segment
        segment1_B: (..., 3), the end point of the first line segment
        segment2_A: (..., 3), the start point of the second line segment
        segment2_B: (..., 3), the end point of the second line segment
    Returns:
        dist: (...), the minimal distance between two line segments
    """
    r = segment2_A - segment1_A
    u = segment1_B - segment1_A
    v = segment2_B - segment2_A

    ru = torch.sum(r * u, dim=-1)
    rv = torch.sum(r * v, dim=-1)
    uu = torch.sum(u * u, dim=-1)
    uv = torch.sum(u * v, dim=-1)
    vv = torch.sum(v * v, dim=-1)

    det = uu * vv - uv * uv
    cond = det < 1e-6 * uu * vv
    s = torch.clamp(torch.where(cond, ru / uu, (ru * vv - rv * uv) / det), 0, 1)
    t = torch.where(cond, torch.zeros_like(det), torch.clamp((ru * uv - rv * uu) / det, 0, 1))

    S = torch.clamp((t * uv + ru) / uu, 0, 1)
    T = torch.clamp((s * uv - rv) / vv, 0, 1)

    A = segment1_A + u * S.unsqueeze(-1)
    B = segment2_A + v * T.unsqueeze(-1)
    dist = torch.sqrt(torch.sum(torch.square(A - B), dim=-1))
    return dist


def segment_segment_integrated_distance(segment1_A, segment1_B, segment2_A, segment2_B):
    """
    Calculate the integrated mean distance between two line segments.
    Args:
        segment1_A: (..., 3), the start point of the first line segment
        segment1_B: (..., 3), the end point of the first line segment
        segment2_A: (..., 3), the start point of the second line segment
        segment2_B: (..., 3), the end point of the second line segment
    Returns:
        dist: (...), the integrated mean distance between two line segments
    """
    A1A2 = segment1_A - segment2_A
    B1B2 = segment1_B - segment2_B
    dist = (torch.sum(A1A2 * A1A2, dim=-1) + torch.sum(B1B2 * B1B2, dim=-1) +
            torch.sum(A1A2 * B1B2, dim=-1)) / 3
    return dist


@torch.no_grad()
def _cost_matrix(points, weights, segment_A, segment_B):
    # weighted points to line segments distance matrix
    # dist = point_segment_distance(
    #     points[None, :, :, :],
    #     segment_A[:, None, None, :],
    #     segment_B[:, None, None, :],
    # )
    # weights_norm = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-7)
    # cost_points = torch.sum(dist * weights_norm[None, :, :], dim=-1)
    # line segments to line segments distance matrix
    # cost_segments = segment_segment_distance(
    #     segment_A[:, None, :],
    #     segment_B[:, None, :],
    #     segment_A[None, :, :],
    #     segment_B[None, :, :],
    # )
    # cost = cost_points + cost_segments * (1 - torch.sum(weights, dim=-1))
    # integrated line segments to line segments distance matrix
    cost_segments = segment_segment_integrated_distance(
        segment_A[:, None, :],
        segment_B[:, None, :],
        segment_A[None, :, :],
        segment_B[None, :, :],
    )
    cost = cost_segments
    return cost


def _compute_sinkhorn_loss(C, epsilon, niter, mass_x, mass_y):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """
    # normalize mass
    mass_x = torch.clamp(mass_x, min=0, max=1e9)
    mass_x = mass_x + 1e-9
    mu = (mass_x / mass_x.sum(dim=-1, keepdim=True)).to(C.device)

    mass_y = torch.clamp(mass_y, min=0, max=1e9)
    mass_y = mass_y + 1e-9
    nu = (mass_y / mass_y.sum(dim=-1, keepdim=True)).to(C.device)

    def M(u, v):
        """Modified cost for logarithmic updates
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"""
        return (-C + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.logsumexp(A, dim=2, keepdim=True)

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.

    for i in range(niter):
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).transpose(1, 2)).squeeze()) + v

    pi = torch.exp(M(u, v))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C, dim=[1, 2])  # Sinkhorn cost

    return cost


def _get_sinkhorn_loss(rendering, ray_history, batch, config):
    # output = rendering['rgb'].permute(0, 3, 1, 2)
    # target = batch['rgb'].permute(0, 3, 1, 2)
    # batch_size, _, H, W = output.shape

    # # randomly select a color channel, to speedup and consume memory
    # i = random.randint(0, 2)
    # output = output[:, [i], :, :]
    # target = target[:, [i], :, :]

    # if max(H, W) > config.sinkhorn_patch_size:
    #     if H > W:
    #         W = int(config.sinkhorn_patch_size * W / H)
    #         H = config.sinkhorn_patch_size
    #     else:
    #         H = int(config.sinkhorn_patch_size * H / W)
    #         W = config.sinkhorn_patch_size
    #     output = nn.functional.interpolate(output, [H, W], mode='area')
    #     target = nn.functional.interpolate(target, [H, W], mode='area')

    # cost_matrix = _cost_matrix(batch_size, H, W, output.device)
    # sinkhorn_loss = _compute_sinkhorn_loss(cost_matrix,
    #                                        epsilon=0.1,
    #                                        niter=5,
    #                                        mass_x=output.reshape(batch_size, -1),
    #                                        mass_y=target.reshape(batch_size, -1))

    # randomly select a color channel, to speedup and consume memory
    num_channels = batch['rgb'].shape[-1]
    i = random.randint(0, num_channels - 1)
    output = rendering['rgb'][..., i].reshape(-1)
    target = batch['rgb'][..., i].reshape(-1)
    segment_A = (batch['origins'] + batch['directions'] * batch['near']).flatten(0, 2)
    segment_B = (batch['origins'] + batch['directions'] * batch['far']).flatten(0, 2)
    points = ray_history['coord'].flatten(0, 2)
    weights = ray_history['weights'].flatten(0, 2)

    sample_num = 1024
    # weights_sum = weights.sum(dim=-1)
    # sample_index = torch.topk(weights_sum, sample_num)[1]
    output = output[:sample_num]  # [sample_index]
    target = target[:sample_num]  # [sample_index]
    segment_A = segment_A[:sample_num]  # [sample_index]
    segment_B = segment_B[:sample_num]  # [sample_index]
    points = points[:sample_num]  # [sample_index]
    weights = weights[:sample_num]  # [sample_index]

    cost_matrix = _cost_matrix(points, weights, segment_A, segment_B)
    sinkhorn_loss = _compute_sinkhorn_loss(cost_matrix,
                                           epsilon=0.1,
                                           niter=5,
                                           mass_x=output.unsqueeze(0),
                                           mass_y=target.unsqueeze(0))

    return sinkhorn_loss


def _get_data_loss(residual_sq, residual_abs, config):
    if config.data_loss_type == 'mse':
        # Mean-squared error (L2) loss.
        data_loss = residual_sq
    elif config.data_loss_type == 'l1':
        # Mean-absolute error (L1) loss.
        data_loss = residual_abs
    elif config.data_loss_type == 'charb':
        # Charbonnier loss.
        data_loss = torch.sqrt(residual_sq + config.charb_padding**2)
    elif config.data_loss_type == 'huber':
        data_loss = torch.where(residual_abs < config.huber_delta, 0.5 * residual_sq,
                                config.huber_delta * (residual_abs - 0.5 * config.huber_delta))
    else:
        assert False, f'Unknown data loss type {config.data_loss_type}'
    return data_loss


def compute_data_loss(batch, renderings, ray_history, config: configs.Config):
    """Computes data loss terms for RGB, normal, and depth outputs."""
    data_losses = []
    mask_losses = []
    sinkhorn_losses = []
    stats = collections.defaultdict(lambda: [])
    use_mask_loss = config.mask_loss_mult > 0 and 'alphas' in batch
    use_sinkhorn_loss = config.sinkhorn_loss_mult > 0

    for level, rendering in enumerate(renderings):
        is_final_level = level == len(renderings) - 1

        if config.data_coarse_loss_mult > 0 or is_final_level:
            residual = rendering['rgb'] - batch['rgb'][..., :3]
            residual_sq = torch.square(residual)
            residual_abs = torch.abs(residual)
            stats['mses'].append(residual_sq.mean().item())
            stats['maes'].append(residual_abs.mean().item())

            data_loss = _get_data_loss(residual_sq, residual_abs, config)
            data_losses.append(data_loss.mean())

        if use_mask_loss and is_final_level:
            mask_residual = rendering['acc'] - batch['alphas']
            mask_residual_sq = torch.square(mask_residual)
            mask_residual_abs = torch.abs(mask_residual)
            stats['mask_mses'].append(mask_residual_sq.mean().item())
            stats['mask_maes'].append(mask_residual_abs.mean().item())

            mask_loss = _get_data_loss(mask_residual_sq, mask_residual_abs, config)
            mask_losses.append(mask_loss.mean())

        if use_sinkhorn_loss and is_final_level:
            sinkhorn_loss = _get_sinkhorn_loss(rendering, ray_history[level], batch, config)
            stats['sinkhorn_loss'].append(sinkhorn_loss.mean().item())
            sinkhorn_losses.append(sinkhorn_loss.mean())

    loss = (config.data_coarse_loss_mult * sum(data_losses[:-1]) +
            config.data_loss_mult * data_losses[-1])

    if use_mask_loss:
        loss += config.mask_loss_mult * mask_losses[-1]

    if use_sinkhorn_loss:
        loss += config.sinkhorn_loss_mult * sinkhorn_losses[-1]

    stats = {k: np.array(stats[k]) for k in stats}
    return loss, stats


def interlevel_loss(ray_history, config: configs.Config):
    """Computes the interlevel loss defined in mip-NeRF 360."""
    # Stop the gradient from the interlevel loss onto the NeRF MLP.
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist'].detach()
    w = last_ray_results['weights'].detach()
    loss_interlevel = torch.tensor(0., device=c.device)
    for ray_results in ray_history[:-1]:
        cp = ray_results['sdist']
        wp = ray_results['weights']
        loss_interlevel += stepfun.lossfun_outer(c, w, cp, wp).mean()
    return config.interlevel_loss_mult * loss_interlevel


def anti_interlevel_loss(ray_history, config: configs.Config):
    """Computes the interlevel loss defined in mip-NeRF 360."""
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist'].detach()
    w = last_ray_results['weights'].detach()
    w_normalize = w / (c[..., 1:] - c[..., :-1])
    loss_anti_interlevel = torch.tensor(0., device=c.device)
    for i, ray_results in enumerate(ray_history[:-1]):
        cp = ray_results['sdist']
        wp = ray_results['weights']
        c_, w_ = stepfun.blur_stepfun(c, w_normalize, config.pulse_width[i])

        # piecewise linear pdf to piecewise quadratic cdf
        area = 0.5 * (w_[..., 1:] + w_[..., :-1]) * (c_[..., 1:] - c_[..., :-1])

        cdf = torch.cat([torch.zeros_like(area[..., :1]), torch.cumsum(area, dim=-1)], dim=-1)

        # query piecewise quadratic interpolation
        cdf_interp = stepfun.sorted_interp_quad(cp, c_, w_, cdf)
        # difference between adjacent interpolated values
        w_s = torch.diff(cdf_interp, dim=-1)

        loss_anti_interlevel += ((w_s - wp).clamp_min(0)**2 / (wp + 1e-5)).mean()
    return config.anti_interlevel_loss_mult * loss_anti_interlevel


def distortion_loss(ray_history, config: configs.Config):
    """Computes the distortion loss regularizer defined in mip-NeRF 360."""
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist']
    w = last_ray_results['weights']
    loss = stepfun.lossfun_distortion(c, w).mean()
    return config.distortion_loss_mult * loss


def opacity_reg_loss(renderings, config: configs.Config):
    total_loss = 0.
    for rendering in renderings:
        o = rendering['acc']
        total_loss += config.opacity_loss_mult * (-o * torch.log(o + 1e-5)).mean()
    return total_loss


def hash_decay_loss(ray_history, config: configs.Config):
    last_ray_results = ray_history[-1]
    total_loss = torch.tensor(0., device=last_ray_results['sdist'].device)
    for ray_results in ray_history:
        if 'hash_levelwise_mean' not in ray_results:
            continue
        hash_levelwise_mean = ray_results['hash_levelwise_mean'].mean()
        total_loss += config.hash_decay_mult * hash_levelwise_mean
    return total_loss


def error_loss(batch, renderings, ray_history, config: configs.Config):
    rendering = renderings[-1]
    ray_history = ray_history[-1]
    residual = rendering['rgb'].detach() - batch['rgb'][..., :3]
    residual_sq = torch.square(residual)
    residual_target = residual_sq.sum(-1, keepdim=True)

    error_residual = rendering['error'] - torch.clamp(residual_target, 0.0, 1.0)
    error_residual = torch.where(error_residual > 0.0, error_residual,
                                 -config.error_loss_lower_lambda * error_residual)
    
    density = ray_history['error_density']
    rgb = ray_history['error_rgb']
    error_reg = 0.01 * density + 0.1 * rgb.mean(-1)

    loss = config.error_loss_mult * (error_residual.mean() + error_reg.mean())
    return loss


def density_reg_loss(model, config: configs.Config):
    total_loss = 0.
    density_alpha = model.nerf.density_params[:model.nerf.step]
    # Encourage the density alpha to be close to 0.
    loss = config.density_reg_loss_mult * density_alpha.mean()
    return loss
