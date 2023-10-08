import torch
import torch.nn as nn
import collections
import numpy as np
from source import configs
from source.utils import stepfun
import random


_cost_matric_cache = {}
def _cost_matrix(batch_size, h, w, device : torch.device):
    key = (batch_size, h, w, device)
    if key in _cost_matric_cache:
        c = _cost_matric_cache[key]
    else:
        a = torch.linspace(0.0, h - 1.0, h, device=device)
        b = torch.linspace(0.0, w - 1.0, w, device=device)
        y_grid = a.view(-1, 1).repeat(batch_size, 1, w) / h
        x_grid = b.view(1, -1).repeat(batch_size, h, 1) / w
        grids = torch.cat([y_grid.view(batch_size, -1, 1), x_grid.view(batch_size, -1, 1)], dim=-1)
        
        x_col = grids.unsqueeze(2)
        y_lin = grids.unsqueeze(1)
        p = 2  
        # Returns the matrix of $|x_i-y_j|^p$.
        c = torch.sum((torch.abs(x_col - y_lin))**p, -1)
        _cost_matric_cache[key] = c
    return c


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
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(2, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.

    for i in range(niter):
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).transpose(dim0=1, dim1=2)).squeeze()) + v

    pi = torch.exp(M(u, v))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C, dim=[1, 2])  # Sinkhorn cost

    return cost


def _get_sinkhorn_loss(rendering, batch, config):
    output = rendering['rgb'].permute(0, 3, 1, 2)
    target = batch['rgb'].permute(0, 3, 1, 2)
    batch_size, _, H, W = output.shape
    
    # randomly select a color channel, to speedup and consume memory
    i = random.randint(0, 2)
    output = output[:, [i], :, :]
    target = target[:, [i], :, :]

    if max(H, W) > config.sinkhorn_patch_size:
        if H > W:
            W = int(config.sinkhorn_patch_size * W / H)
            H = config.sinkhorn_patch_size
        else:
            H = int(config.sinkhorn_patch_size * H / W)
            W = config.sinkhorn_patch_size
        output = nn.functional.interpolate(output, [H, W], mode='area')
        target = nn.functional.interpolate(target, [H, W], mode='area')

    cost_matrix = _cost_matrix(batch_size, H, W, output.device)
    sinkhorn_loss = _compute_sinkhorn_loss(cost_matrix,
                                           epsilon=0.1,
                                           niter=5,
                                           mass_x=output.reshape(batch_size, -1),
                                           mass_y=target.reshape(batch_size, -1))

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


def compute_data_loss(batch, renderings, config: configs.Config):
    """Computes data loss terms for RGB, normal, and depth outputs."""
    data_losses = []
    mask_losses = []
    sinkhorn_losses = []
    stats = collections.defaultdict(lambda: [])
    use_mask_loss = config.mask_loss_mult > 0 and 'alphas' in batch
    use_sinkhorn_loss = config.sinkhorn_loss_mult > 0

    for rendering in renderings:
        residual = rendering['rgb'] - batch['rgb'][..., :3]
        residual_sq = torch.square(residual)
        residual_abs = torch.abs(residual)
        stats['mses'].append(residual_sq.mean().item())
        stats['maes'].append(residual_abs.mean().item())

        data_loss = _get_data_loss(residual_sq, residual_abs, config)
        data_losses.append(data_loss.mean())

        if use_mask_loss:
            mask_residual = rendering['acc'] - batch['alphas']
            mask_residual_sq = torch.square(mask_residual)
            mask_residual_abs = torch.abs(mask_residual)
            stats['mask_mses'].append(mask_residual_sq.mean().item())
            stats['mask_maes'].append(mask_residual_abs.mean().item())

            mask_loss = _get_data_loss(mask_residual_sq, mask_residual_abs, config)
            mask_losses.append(mask_loss.mean())

        if use_sinkhorn_loss:
            sinkhorn_loss = _get_sinkhorn_loss(rendering, batch, config)
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


def distortion_loss(ray_history, config):
    """Computes the distortion loss regularizer defined in mip-NeRF 360."""
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist']
    w = last_ray_results['weights']
    loss = stepfun.lossfun_distortion(c, w).mean()
    return config.distortion_loss_mult * loss


def opacity_reg_loss(renderings, config):
    total_loss = 0.
    for rendering in renderings:
        o = rendering['acc']
        total_loss += config.opacity_loss_mult * (-o * torch.log(o + 1e-5)).mean()
    return total_loss


def hash_decay_loss(ray_history, config):
    last_ray_results = ray_history[-1]
    total_loss = torch.tensor(0., device=last_ray_results['sdist'].device)
    for ray_results in ray_history:
        if 'hash_levelwise_mean' not in ray_results:
            continue
        hash_levelwise_mean = ray_results['hash_levelwise_mean'].mean()
        total_loss += config.hash_decay_mult * hash_levelwise_mean
    return total_loss