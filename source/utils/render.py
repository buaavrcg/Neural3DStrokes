import torch
from . import stepfun


def cast_rays(tdist, origins, directions, radii):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

    Args:
        tdist: float array, the "fencepost" distances along the ray.
        origins: float array, the ray origin coordinates.
        directions: float array, the ray direction vectors.
        radii: float array, the radii (base radii for cones) of the rays.

    Returns:
        a tuple of arrays of means and covariances:
            coords: [batch_size, H, W, n_samples, 3], the sample coordinates.
            radius: [batch_size, H, W, n_samples], the sample radii.
            t: [batch_size, H, W, n_samples], the ray distances.
    """
    t0 = tdist[..., :-1]
    t1 = tdist[..., 1:]
    t = (t0 + t1) / 2
    coords = directions[..., None, :] * t[..., None]
    coords = coords + origins[..., None, :]
    radius = radii * t
    return coords, radius, t


def compute_alpha_weights(density, tdist, dirs, opaque_background=False):
    """Helper function for computing alpha compositing weights."""
    t_delta = tdist[..., 1:] - tdist[..., :-1]
    delta = t_delta * torch.norm(dirs[..., None, :], dim=-1)
    density_delta = density * delta

    if opaque_background:
        # Equivalent to making the final t-interval infinitely wide.
        density_delta = torch.cat([
            density_delta[..., :-1],
            torch.full_like(density_delta[..., -1:], torch.inf)
        ], dim=-1)

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans
    return weights, alpha, trans


def volumetric_rendering(rgbs,
                         weights,
                         tdist,
                         bg_rgbs,
                         t_far,
                         compute_extras,
                         extras=None):
    """Volumetric Rendering Function.

    Args:
        rgbs: color, [batch_size, num_samples, 3]
        weights: weights, [batch_size, num_samples].
        tdist: [batch_size, num_samples].
        bg_rgbs: the color(s) to use for the background.
        t_far: [batch_size, 1], the distance of the far plane.
        compute_extras: bool, if True, compute extra quantities besides color.
        extras: dict, a set of values along rays to render by alpha compositing.

    Returns:
        rendering: a dict containing an rgb image of size [batch_size, 3], and other
        visualizations if compute_extras=True.
    """
    eps = torch.finfo(rgbs.dtype).eps
    # eps = 1e-3
    rendering = {}

    acc = weights.sum(dim=-1)
    bg_w = (1 - acc[..., None]).clamp_min(0.)  # The weight of the background.
    rgb = (weights[..., None] * rgbs).sum(dim=-2) + bg_w * bg_rgbs
    t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
    depth = (
        torch.clip(
            torch.nan_to_num((weights * t_mids).sum(dim=-1) / acc.clamp_min(eps), torch.inf),
            tdist[..., 0], tdist[..., -1]))

    rendering['rgb'] = rgb
    rendering['depth'] = depth
    rendering['acc'] = acc

    if compute_extras:
        if extras is not None:
            for k, v in extras.items():
                if v is not None:
                    rendering[k] = (weights[..., None] * v).sum(dim=-2)

        expectation = lambda x: (weights * x).sum(dim=-1) / acc.clamp_min(eps)
        # For numerical stability this expectation is computing using log-distance.
        rendering['distance_mean'] = (
            torch.clip(
                torch.nan_to_num(torch.exp(expectation(torch.log(t_mids))), torch.inf),
                tdist[..., 0], tdist[..., -1]))

        # Add an extra fencepost with the far distance at the end of each ray, with
        # whatever weight is needed to make the new weight vector sum to exactly 1
        # (`weights` is only guaranteed to sum to <= 1, not == 1).
        t_aug = torch.cat([tdist, t_far], dim=-1)
        weights_aug = torch.cat([weights, bg_w], dim=-1)

        ps = [5, 50, 95]
        distance_percentiles = stepfun.weighted_percentile(t_aug, weights_aug, ps)

        for i, p in enumerate(ps):
            s = 'median' if p == 50 else 'percentile_' + str(p)
            rendering['distance_' + s] = distance_percentiles[..., i]

    return rendering
