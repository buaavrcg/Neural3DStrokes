import numpy as np
import torch


def contract(x):
    """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
    eps = torch.finfo(x.dtype).eps
    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
    z = torch.where(x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
    return z


def inv_contract(z):
    """The inverse of contract()."""
    eps = torch.finfo(z.dtype).eps
    # Clamping to eps prevents non-finite gradients when z == 0.
    z_mag_sq = torch.sum(z ** 2, dim=-1, keepdim=True).clamp_min(eps)
    x = torch.where(z_mag_sq <= 1, z, z / (2 * torch.sqrt(z_mag_sq) - z_mag_sq).clamp_min(eps))
    return x


def contract_mean_jacobi(x):
    eps = torch.finfo(x.dtype).eps

    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
    x_mag_sqrt = torch.sqrt(x_mag_sq)
    x_xT = torch.matmul(x[..., None], x[..., None, :])
    mask = x_mag_sq <= 1
    z = torch.where(mask, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)

    eye = torch.broadcast_to(torch.eye(3, device=x.device), z.shape[:-1] + z.shape[-1:] * 2)
    jacobi = (2 * x_xT * (1 - x_mag_sqrt[..., None]) + (2 * x_mag_sqrt[..., None] ** 3 - x_mag_sqrt[..., None] ** 2) * eye) / x_mag_sqrt[..., None] ** 4
    jacobi = torch.where(mask[..., None], eye, jacobi)
    return z, jacobi


def power_transformation(x, lam):
    """
    power transformation for Eq(4) in zip-nerf
    """
    lam_1 = np.abs(lam - 1)
    return lam_1 / lam * ((x / lam_1 + 1) ** lam - 1)


def inv_power_transformation(x, lam):
    """
    inverse power transformation
    """
    lam_1 = np.abs(lam - 1)
    eps = torch.finfo(x.dtype).eps  # may cause inf
    return ((x * lam / lam_1 + 1 + eps) ** (1 / lam) - 1) * lam_1


def construct_ray_warps(fn, t_near, t_far, lam=-1.5):
    """Construct a bijection between metric distances and normalized distances.

    See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
    detailed explanation.

    Args:
        fn: the function to ray distances.
        t_near: a tensor of near-plane distances.
        t_far: a tensor of far-plane distances.
        lam: for lam in Eq(4) in zip-nerf

    Returns:
        t_to_s: a function that maps distances to normalized distances in [0, 1].
        s_to_t: the inverse of t_to_s.
    """
    if fn is None:
        fn_fwd = lambda x: x
        fn_inv = lambda x: x
    else:
        fwd_mapping = {
            'power_transformation': lambda x: power_transformation(x * 2, lam=lam),
            'piecewise': lambda x: torch.where(x < 1, .5 * x, 1 - .5 / x),
            'reciprocal': torch.reciprocal,
            'log': torch.log,
            'exp': torch.exp,
            'sqrt': torch.sqrt,
            'square': torch.square,
        }
        inv_mapping = {
            'power_transformation': lambda x: inv_power_transformation(x, lam=lam) / 2,
            'piecewise': lambda x: torch.where(x < .5, 2 * x, .5 / (1 - x)),
            'reciprocal': torch.reciprocal,
            'log': torch.exp,
            'exp': torch.log,
            'sqrt': torch.square,
            'square': torch.sqrt,
        }
        fn_fwd = fwd_mapping[fn]
        fn_inv = inv_mapping[fn]

    s_near, s_far = [fn_fwd(x) for x in (t_near, t_far)]
    t_to_s = lambda t: (fn_fwd(t) - s_near) / (s_far - s_near)
    s_to_t = lambda s: fn_inv(s * s_far + (1 - s) * s_near)
    return t_to_s, s_to_t


def pos_enc(x, min_deg, max_deg, with_identity=True):
    """The positional encoding used by the original NeRF paper.
    
    Args:
        x: tensor, the coordinates to be encoded
        min_deg: int, the min degree of the encoding.
        max_deg: int, the max degree of the encoding.
        with_identity: bool, whether to include the identity encoding.

    Returns:
        encoded: tensor, encoded variables.
    """
    scales = 2 ** torch.arange(min_deg, max_deg, device=x.device)
    shape = x.shape[:-1] + (-1,)
    scaled_x = (x[..., None, :] * scales[:, None]).reshape(*shape)
    four_feat = torch.sin(torch.cat([scaled_x, scaled_x + 0.5 * torch.pi], dim=-1))
    if with_identity:
        return torch.cat([x, four_feat], dim=-1)
    else:
        return four_feat
