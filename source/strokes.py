import torch


def _broadcast_params(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast params to match the shape of x."""
    expand_shape = x.shape[:-1]
    p = p.view([1] * len(expand_shape) + [-1]).broadcast_to(expand_shape + (-1,))
    assert x.shape[:-1] == p.shape[:-1]
    return p


def unit_sphere(x: torch.Tensor, p: torch.Tensor):
    """SDF for a unit sphere."""
    x = torch.square(x)
    d = torch.sqrt(x[..., 0] + x[..., 1] + x[..., 2] + 1e-8)
    return d - 1.


def unit_cube(x: torch.Tensor, p: torch.Tensor):
    """SDF for a unit cube."""
    return torch.max(torch.abs(x), dim=-1).values - 1.


def rotate(x: torch.Tensor, r: torch.Tensor):
    """Rotate a point."""
    u = r[..., 0]
    v = r[..., 1]
    w = r[..., 2]
    su, cu = torch.sin(u), torch.cos(u)
    sv, cv = torch.sin(v), torch.cos(v)
    sw, cw = torch.sin(w), torch.cos(w)

    rot = torch.stack([
        torch.stack([cv * cw, su * sv * cw - cu * sw, su * sw + cu * sv * cw], dim=-1),
        torch.stack([cv * sw, cu * cw + su * sv * sw, cu * sv * sw - su * cw], dim=-1),
        torch.stack([-sv, su * cv, cu * cv], dim=-1),
    ], dim=-2)

    return torch.einsum('...ij,...j->...i', rot.T, x)


def stroke_sdf(shape_type: str):
    """Get the SDF, shape params dim, and param sampler for a shape type."""
    sdfs = {
        'sphere': (unit_sphere, [], None, True, False, True, False),
        'ellipsoid': (unit_sphere, [], None, True, True, False, True),
        'cube': (unit_cube, [], None, True, True, True, False),
        'obb': (unit_cube, [], None, True, True, False, True),
    }
    base_sdf, param_ranges, sampler, enable_translation, enable_rotation, \
        enable_singlescale, enable_multiscale = sdfs[shape_type]

    if enable_singlescale:
        param_ranges += [(1e-2, 0.5)]
    elif enable_multiscale:
        param_ranges += [(1e-2, 0.5)] * 3
    if enable_rotation:
        param_ranges += [(-torch.pi, torch.pi)] * 3
    if enable_translation:
        param_ranges += [(None, None)] * 3
    dim_shape = len(param_ranges)

    def composite_sdf(x: torch.Tensor, p: torch.Tensor):
        p = _broadcast_params(p, x)
        if enable_translation:
            t = p[..., -3:]
            x = x - t
            p = p[..., :-3]
        if enable_rotation:
            x = rotate(x, p[..., -3:])
            p = p[..., :-3]
        if enable_singlescale:
            s = p[..., -1:]
            x = x / s
            p = p[..., :-1]
        elif enable_multiscale:
            s = p[..., -3:]
            x = x / s
            p = p[..., :-3]
        return base_sdf(x, p), x

    def composite_sampler(train_frac):
        params = [sampler()] if sampler is not None else []
        scale_min = 0.05 + 0.08 * (1 - train_frac)
        scale_max = 0.05 + 0.15 * (1 - train_frac)
        if enable_singlescale:
            params.append(torch.rand(1) * (scale_max - scale_min) + scale_min)
        elif enable_multiscale:
            params.append(torch.rand(3) * (scale_max - scale_min) + scale_min)
        if enable_rotation:
            params.append((torch.rand(3) * 2 - 1) * torch.pi)
        if enable_translation:
            params.append((torch.rand(3) * 2 - 1) * 0.4)
        return torch.cat(params, dim=-1)

    return composite_sdf, dim_shape, param_ranges, composite_sampler


def gradient_color(x: torch.Tensor, p: torch.Tensor):
    x0 = p[..., 0:3]
    x1 = p[..., 3:6]
    c0 = p[..., 6:9]
    c1 = p[..., 9:12]
    v1 = x1 - x0
    d = torch.einsum('...i,...i -> ...', x - x0, v1)
    d1 = torch.einsum('...i,...i -> ...', v1, v1)
    t_raw = d / d1  # gradient range in [0, 1]
    t = torch.sigmoid(t_raw * 5 - 2.5)  # use sigmoid for soft clamping
    c = c0 * (1 - t[..., None]) + c1 * t[..., None]
    return c


def stroke_color(color_type: str):
    """Get the color function, color params dim, and param sampler for a color type."""
    colors = {
        'constant': (lambda x, c: c, [(0, 1)] * 3, lambda: torch.rand(3), False, False),
        'gradient': (gradient_color, 
                     [(None, None)] * 6 + [(0, 1)] * 6, 
                     lambda: torch.cat([torch.rand(6) * 2 - 1, torch.rand(6)], dim=-1), 
                     False, False),
    }
    color_fn, param_ranges, sampler, use_sdf, use_shape_params = colors[color_type]
    dim_color = len(param_ranges)

    def composite_color(x: torch.Tensor, p: torch.Tensor, 
                        sdf: torch.Tensor, shape_params: torch.Tensor):
        p = _broadcast_params(p, x)
        kwargs = {}
        if use_sdf:
            kwargs['sdf'] = sdf
        if use_shape_params:
            kwargs['shape_params'] = shape_params
        return color_fn(x, p, **kwargs)

    return composite_color, dim_color, param_ranges, sampler
