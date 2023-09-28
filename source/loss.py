import torch
import collections
import numpy as np
from source import configs
from source.utils import stepfun


def compute_data_loss(batch, renderings, config : configs.Config):
    """Computes data loss terms for RGB, normal, and depth outputs."""
    data_losses = []
    stats = collections.defaultdict(lambda: [])

    for rendering in renderings:
        residual = rendering['rgb'] - batch['rgb'][..., :3]
        residual_sq = torch.square(residual)
        residual_abs = torch.abs(residual)
        stats['mses'].append(residual_sq.mean().item())
        stats['maes'].append(residual_abs.mean().item())

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
        data_losses.append(data_loss.mean())

    loss = (config.data_coarse_loss_mult * sum(data_losses[:-1]) +
            config.data_loss_mult * data_losses[-1])

    stats = {k: np.array(stats[k]) for k in stats}
    return loss, stats


def interlevel_loss(ray_history, config : configs.Config):
    """Computes the interlevel loss defined in mip-NeRF 360."""
    # Stop the gradient from the interlevel loss onto the NeRF MLP.
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist'].detach()
    w = last_ray_results['weights'].detach()
    loss_interlevel = 0.
    for ray_results in ray_history[:-1]:
        cp = ray_results['sdist']
        wp = ray_results['weights']
        loss_interlevel += stepfun.lossfun_outer(c, w, cp, wp).mean()
    return config.interlevel_loss_mult * loss_interlevel


def anti_interlevel_loss(ray_history, config : configs.Config):
    """Computes the interlevel loss defined in mip-NeRF 360."""
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist'].detach()
    w = last_ray_results['weights'].detach()
    w_normalize = w / (c[..., 1:] - c[..., :-1])
    loss_anti_interlevel = 0.
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

        loss_anti_interlevel += ((w_s - wp).clamp_min(0) ** 2 / (wp + 1e-5)).mean()
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
    total_loss = 0.
    for ray_results in ray_history:
        total_loss += config.hash_decay_mult * ray_results['hash_levelwise_mean'].mean()
    return total_loss