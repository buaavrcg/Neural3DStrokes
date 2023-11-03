import math
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _strokelib as _backend
except ImportError:
    from .backend import _backend

_base_sdf_id = {
    'unit_sphere': 0,
    'unit_cube': 1,
    'unit_round_cube': 2,
    'unit_capped_torus': 3,
    'unit_capsule': 4,
    'unit_line': 5,
    'unit_triprism': 6,
    'unit_octahedron': 7,
    'quadratic_bezier': 8,
    'cubic_bezier': 9,
    'catmull_rom': 10,
}

def _make_spline_init_fn(num_control_points, num_radius):
    def _init_fn(args):
        (scale_min, scale_max, sample_coord) = args
        scales = torch.rand(1) * (scale_max - scale_min) + scale_min
        control_points = []
        for i in range(num_control_points):
            P = sample_coord + (torch.rand(3) - 0.5) * scales
            control_points.append(P)
        radius = (0.2 + 0.2 * torch.rand(num_radius)) * scales
        return torch.cat(control_points + [radius])
    
    return _init_fn

_sdf_dict = {
    'sphere': ('unit_sphere', [], None, True, False, True, False),
    'ellipsoid': ('unit_sphere', [], None, True, True, False, True),
    'aacube': ('unit_cube', [], None, True, False, True, False),
    'cube': ('unit_cube', [], None, True, True, True, False),
    'aabb': ('unit_cube', [], None, True, False, False, True),
    'obb': ('unit_cube', [], None, True, True, False, True),
    'roundcube': ('unit_round_cube', [(0, 1)], lambda _: torch.rand(1), True, True, True, False),
    'cappedtorus':
    ('unit_capped_torus', [(0, 2 * torch.acos(torch.tensor(0.0))),
                           (0, None)], lambda _: torch.rand(2), True, True, True, False),
    'capsule':
    ('unit_capsule', [(0.5, None)], lambda _: torch.rand(1) + 0.5, True, True, True, False),
    'scapsule':
    ('unit_capsule', [(0.5, None)], lambda _: torch.rand(1) + 0.5, True, True, False, True),
    'line': ('unit_line', [
        (0.5, None), (0, 1)
    ], lambda _: torch.cat([torch.rand(1) * 2.0 + 0.5, torch.rand(1)]), True, True, True, False),
    'triprism': ('unit_triprism', [(0, None)], lambda _: torch.rand(1), True, True, True, False),
    'octahedron': ('unit_octahedron', [], None, True, True, True, False),
    'quadratic_bezier':
    ('quadratic_bezier', [(-1, 1), (-1, 1), (-1, 1), 
                          (-1, 1), (-1, 1), (-1, 1), 
                          (-1, 1), (-1, 1), (-1, 1), 
                          (0.001, 0.2), (0.001, 0.2)],
     _make_spline_init_fn(3, 2), False, False, False, False),
    'cubic_bezier':
    ('cubic_bezier', [(-1, 1), (-1, 1), (-1, 1), 
                      (-1, 1), (-1, 1), (-1, 1), 
                      (-1, 1), (-1, 1), (-1, 1), 
                      (-1, 1), (-1, 1), (-1, 1),
                      (0.001, 0.2), (0.001, 0.2)],
     _make_spline_init_fn(4, 2), False, False, False, False),
    'catmull_rom':
    ('catmull_rom', [(-1, 1), (-1, 1), (-1, 1), 
                      (-1, 1), (-1, 1), (-1, 1), 
                      (-1, 1), (-1, 1), (-1, 1), 
                      (-1, 1), (-1, 1), (-1, 1),
                      (0.001, 0.2), (0.001, 0.2)],
     _make_spline_init_fn(4, 2), False, False, False, False),
}

_color_dict = {
    'constant_rgb': (0, [(0, 1)] * 3, lambda: torch.rand(3)),
    'gradient_rgb': (1, [(-1, 1)] * 6 + [(0, 1)] * 6,
                     lambda: torch.cat([torch.rand(6) - 0.5, torch.rand(6)], dim=-1)),
    'noise_brush_rgb': (2, [(0, 1)] * 3, lambda: torch.rand(3)),
    'constant_sh2': (3, [(None, None)] * 12, lambda: torch.randn(12)),
    'constant_sh3': (4, [(None, None)] * 27, lambda: torch.randn(27)),
}

_color_dim = [3, 3, 3, 3, 3]


def _make_sdf_id(base_sdf_name: str, enable_translation: bool, enable_rotation: bool,
                 enable_singlescale: bool, enable_multiscale: bool) -> int:
    sdf_id = _base_sdf_id[base_sdf_name] << 4
    if enable_translation:
        sdf_id = sdf_id | (1 << 0)
    if enable_rotation:
        sdf_id = sdf_id | (1 << 1)
    if enable_singlescale:
        sdf_id = sdf_id | (1 << 2)
    if enable_multiscale:
        sdf_id = sdf_id | (1 << 3)
    return sdf_id


class _stroke_fn(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,
                x: torch.Tensor,
                viewdir: torch.Tensor,
                shape_params: torch.Tensor,
                color_params: torch.Tensor,
                sdf_id: int,
                color_id: int,
                sdf_delta: float,
                use_laplace_transform: bool = False,
                no_sdf: bool = True):
        """Compute the SDF value and the base coordinates of a batch of strokes.

        Args:
            ctx: Function context.
            x (torch.Tensor): Sample coordinates of shape [..., num_samples, 3].
            viewdir (torch.Tensor): View direction of shape [..., 3].
            shape_params (torch.Tensor): Shape parameters of shape [num_strokes, num_params].
            color_params (torch.Tensor): Color parameters of shape [num_strokes, num_params].
            sdf_id (int): Composite id of the signed distance function to use.
            color_id (int): Id of the color function to use.
            sdf_delta (float): Delta value for the clamping signed distance function.
            use_laplace_transform (bool): Use sigmoid clamping or linear clamping?
            no_sdf (bool): Return None for raw sdf values?
            
        Returns:
            alpha (torch.Tensor): Alpha values in range [0,1] of shape [..., num_strokes].
            sdf (torch.Tensor): Signed distance function values of shape [..., num_strokes].
        """
        assert x.shape[-1] == 3, 'x must have shape [..., num_samples, 3]'
        assert viewdir.shape[:-1] == x.shape[:-2] and viewdir.shape[-1] == 3, 'viewdir must have shape [..., 3]'
        assert shape_params.ndim == 2, 'params must have shape [num_strokes, num_shape_params]'
        assert color_params.ndim == 2, 'color_params must have shape [num_strokes, num_color_params]'
        assert shape_params.shape[0] == color_params.shape[0], 'num_strokes must be the same'
        pre_shape = x.shape[:-1]
        x = x.contiguous().reshape(-1, 3).float()
        viewdir = viewdir.contiguous().reshape(-1, 3).float()
        shape_params = shape_params.contiguous().float()
        color_params = color_params.contiguous().float()
        num_strokes = shape_params.shape[0]

        alpha_shape = (x.shape[0], num_strokes)
        color_shape = (x.shape[0], num_strokes, _color_dim[color_id])
        sdf_shape = (x.shape[0], num_strokes)
        alpha_output = torch.empty(alpha_shape, dtype=x.dtype, device=x.device)
        color_output = torch.empty(color_shape, dtype=x.dtype, device=x.device)
        sdf_output = torch.empty(0 if no_sdf else sdf_shape, dtype=x.dtype, device=x.device)
        _backend.stroke_forward(alpha_output, color_output, sdf_output, x, viewdir, shape_params,
                                color_params, sdf_id, color_id, sdf_delta, use_laplace_transform)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            ctx.save_for_backward(x, viewdir, alpha_output, shape_params, color_params)
            ctx.sdf_id = sdf_id
            ctx.color_id = color_id
            ctx.sdf_delta = sdf_delta
            ctx.use_laplace_transform = use_laplace_transform
            ctx.pre_shape = pre_shape

        alpha_output = alpha_output.reshape(*pre_shape, num_strokes)
        color_output = color_output.reshape(*pre_shape, num_strokes, -1)
        if no_sdf:
            return alpha_output, color_output
        else:
            sdf_output = sdf_output.reshape(*pre_shape, num_strokes)
            return alpha_output, color_output, sdf_output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx,
                 grad_alpha: torch.Tensor,
                 grad_color: torch.Tensor,
                 grad_sdf: torch.Tensor = None):
        x, viewdir, alpha_output, shape_params, color_params = ctx.saved_tensors
        num_strokes = shape_params.shape[0]
        sdf_id = ctx.sdf_id
        color_id = ctx.color_id
        sdf_delta = ctx.sdf_delta
        use_laplace_transform = ctx.use_laplace_transform
        pre_shape = ctx.pre_shape

        grad_alpha = grad_alpha.contiguous().float().reshape(-1, num_strokes)
        grad_color = grad_color.contiguous().float().reshape(-1, num_strokes, _color_dim[color_id])
        if grad_sdf is not None:
            grad_sdf = grad_sdf.contiguous().float().reshape(-1, num_strokes)
        else:
            grad_sdf = torch.zeros(0, dtype=x.dtype, device=x.device)

        grad_shape_params = torch.zeros_like(shape_params)
        grad_color_params = torch.zeros_like(color_params)
        grad_x = torch.zeros(x.shape if ctx.needs_input_grad[0] else 0,
                             dtype=x.dtype,
                             device=x.device)
        _backend.stroke_backward(grad_shape_params, grad_color_params, grad_x, grad_alpha,
                                 grad_color, grad_sdf, x, viewdir, alpha_output, shape_params,
                                 color_params, sdf_id, color_id, sdf_delta, use_laplace_transform)
        if ctx.needs_input_grad[0]:
            grad_x = grad_x.reshape(*pre_shape, 3)
        else:
            grad_x = None
        return grad_x, None, grad_shape_params, grad_color_params, None, None, None, None, None


def get_stroke(shape_type: str, color_type: str):
    """Get the stroke function.
    
    Returns:
        stroke_fn (callable): Stroke function.
        dim_shape (int): Dimension of shape parameters.
        dim_color (int): Dimension of color parameters.
        shape_param_ranges (list): List of shape parameter ranges.
        color_param_ranges (list): List of color parameter ranges.
        shape_sampler (callable): Shape parameter sampler.
        color_sampler (callable): Color parameter sampler.
    """
    base_sdf_name, shape_param_ranges, shape_base_sampler, enable_translation, enable_rotation, \
        enable_singlescale, enable_multiscale = _sdf_dict[shape_type]
    color_id, color_param_ranges, color_sampler = _color_dict[color_type]
    sdf_id = _make_sdf_id(base_sdf_name, enable_translation, enable_rotation, enable_singlescale,
                          enable_multiscale)

    if enable_singlescale:
        shape_param_ranges += [(0.01, 0.5)]
    elif enable_multiscale:
        shape_param_ranges += [(0.01, 0.5)] * 3
    if enable_rotation:
        shape_param_ranges += [(-torch.pi, torch.pi)] * 3
    if enable_translation:
        shape_param_ranges += [(-1.0, 1.0)] * 3

    def shape_sampler(stroke_step, error_coord=None):
        trans_min = torch.tensor([-0.5, -0.5, -0.5])
        trans_max = torch.tensor([0.5, 0.5, 0.5])
        trans_range = torch.abs(trans_max - trans_min)
        scale_range = torch.square(trans_range).sum().sqrt()
        # scale_t = (1 - min(stroke_step / 500, 1.0))**2.0
        scale_t = math.exp(-stroke_step / 200)
        scale_min = 0.02 + 0.13 * scale_t
        scale_max = 0.04 + 0.21 * scale_t
        scale_min = scale_min * scale_range
        scale_max = scale_max * scale_range
        
        if error_coord is not None:
            sample_coord = error_coord
        else:
            sample_coord = trans_min + (trans_max - trans_min) * torch.rand(3)
            
        params = []
        if shape_base_sampler is not None:
            params.append(shape_base_sampler((scale_min, scale_max, sample_coord)))
        if enable_singlescale:
            params.append(torch.rand(1) * (scale_max - scale_min) + scale_min)
        elif enable_multiscale:
            params.append(torch.rand(3) * (scale_max - scale_min) + scale_min)
        if enable_rotation:
            params.append((torch.rand(3) * 2 - 1) * torch.pi)
        if enable_translation:
            params.append(sample_coord)

        if len(params) > 0:
            return torch.cat(params, dim=-1)
        else:
            return torch.empty(0)

    stroke_fn = lambda x, viewdir, shape_params, color_params, *args: _stroke_fn.apply(
        x, viewdir, shape_params, color_params, sdf_id, color_id, *args)
    dim_shape = len(shape_param_ranges)
    dim_color = len(color_param_ranges)
    return stroke_fn, dim_shape, dim_color, shape_param_ranges, color_param_ranges, shape_sampler, color_sampler


class _compositing_fn(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, alphas: torch.Tensor, colors: torch.Tensor, density_params: torch.Tensor):
        """Composite a batch of strokes."""
        assert alphas.ndim >= 2, 'alphas must have shape [..., num_strokes]'
        assert colors.ndim >= 3, 'colors must have shape [..., num_strokes, color_dim]'
        assert density_params.ndim == 1, 'density_params must have shape [num_strokes]'
        assert alphas.shape[:-1] == colors.shape[:-2], \
            'alphas and colors must have the same shape except the last two dimensions'
        assert alphas.shape[-1] == colors.shape[-2] == density_params.shape[0], \
            'alphas, colors and density_params must have the same number of strokes'

        pre_shape = alphas.shape[:-1]
        num_strokes = density_params.shape[0]
        alphas = alphas.contiguous().reshape(-1, num_strokes).float()
        colors = colors.contiguous().reshape(-1, num_strokes, colors.shape[-1]).float()
        density_params = density_params.contiguous().float()

        density_output = torch.empty(alphas.shape[0], dtype=alphas.dtype, device=alphas.device)
        color_output = torch.empty((colors.shape[0], colors.shape[-1]),
                                   dtype=colors.dtype,
                                   device=colors.device)
        _backend.compose_forward(density_output, color_output, alphas, colors, density_params)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            ctx.save_for_backward(alphas, colors, density_params)
            ctx.pre_shape = pre_shape

        density_output = density_output.reshape(*pre_shape)
        color_output = color_output.reshape(*pre_shape, -1)
        return density_output, color_output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_density: torch.Tensor, grad_color: torch.Tensor):
        alphas, colors, density_params = ctx.saved_tensors
        pre_shape = ctx.pre_shape
        num_strokes = density_params.shape[0]

        if grad_density is not None:
            grad_density = grad_density.reshape(-1)
        else:
            grad_density = torch.zeros(0, dtype=alphas.dtype, device=alphas.device)
        if grad_color is not None:
            grad_color = grad_color.reshape(-1, colors.shape[-1])
        else:
            grad_color = torch.zeros(0, dtype=colors.dtype, device=colors.device)

        grad_alphas = torch.zeros_like(alphas)
        grad_colors = torch.zeros_like(colors)
        grad_density_params = torch.zeros_like(density_params)
        _backend.compose_backward(grad_alphas, grad_colors, grad_density_params, grad_density,
                                  grad_color, alphas, colors, density_params)

        grad_alphas = grad_alphas.reshape(*pre_shape, num_strokes)
        grad_colors = grad_colors.reshape(*pre_shape, num_strokes, colors.shape[-1])
        return grad_alphas, grad_colors, grad_density_params


def compose_strokes(alphas: torch.Tensor, colors: torch.Tensor, density_params: torch.Tensor):
    """Composite a batch of strokes.

    Args:
        alphas (torch.Tensor): Alpha values of shape [..., num_strokes].
        colors (torch.Tensor): Color values of shape [..., num_strokes, color_dim].
        density_params (torch.Tensor): Density parameters of shape [num_strokes].
        
    Returns:
        density (torch.Tensor): Density values of shape [...].
        color (torch.Tensor): Color values of shape [..., color_dim].
    """
    return _compositing_fn.apply(alphas, colors, density_params)
