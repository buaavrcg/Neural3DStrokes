import accelerate
import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils._pytree import tree_map
from tqdm import tqdm
from source.utils import coord
from source.utils import stepfun
from source.utils import render
from source.utils import training as train_utils
from source.gridencoder import GridEncoder


@gin.configurable
class Model(nn.Module):
    """A mip-Nerf360 model containing all MLPs."""
    num_prop_samples: int = 64  # The number of samples for each proposal level.
    num_nerf_samples: int = 32  # The number of samples the final nerf level.
    num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).
    bg_intensity_range = (1., 1.)  # The range of background colors.
    stop_level_grad: bool = True  # If True, don't backprop across levels.
    use_viewdirs: bool = True  # If True, use view directions as input.
    raydist_fn = None  # The curve used for ray dists.
    power_transform_lambda: float = -1.5  # Lambda used in raydist power transformation.
    use_multi_samples: bool = False  # If True, use multiple samples in zipnerf.
    single_jitter: bool = True  # If True, jitter whole rays instead of samples.
    dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
    dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
    single_mlp: bool = False  # Use the NerfMLP for all rounds of sampling.
    single_prop: bool = False  # Use the same PropMLP for all rounds of sampling.
    resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
    opaque_background: bool = False  # If true, make the background opaque.
    std_scale: float = 0.5  # Scale the scale of the standard deviation.
    prop_desired_grid_size = [512, 2048]  # The desired grid size for each proposal level.

    def __init__(self, config=None, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.config = config

        # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
        # being regularized.
        self.nerf_mlp = NerfMLP()
        if self.single_mlp:
            self.prop_mlp = self.nerf_mlp
        elif self.single_prop:
            self.prop_mlp = PropMLP()
        else:
            for i in range(self.num_levels - 1):
                self.register_module(
                    f'prop_mlp_{i}',
                    PropMLP(grid_disired_resolution=self.prop_desired_grid_size[i]))

    def forward(self, batch, compute_extras):
        """The mip-NeRF Model.

        Args:
            batch: util.Rays, a pytree of ray origins, directions, and viewdirs.
            compute_extras: bool, if True, compute extra quantities besides color.

        Returns:
            renderings: list of rendering result of each layer, [*(rgb, distance, acc)]
            ray_history: list of ray history of each layer
        """
        device = batch['origins'].device
        rand = self.training  # Random for training, and deterministic for eval

        # Define the mapping from normalized to metric ray distance.
        _, s_to_t = coord.construct_ray_warps(self.raydist_fn, batch['near'], batch['far'],
                                              self.power_transform_lambda)

        # Initialize the range of (normalized) distances for each ray to [0, 1],
        # and assign that single interval a weight of 1. These distances and weights
        # will be repeatedly updated as we proceed through sampling levels.
        init_s_near, init_s_far = 0., 1.
        sdist = torch.cat([
            torch.full_like(batch['near'], init_s_near),
            torch.full_like(batch['far'], init_s_far)
        ], -1)
        weights = torch.ones_like(batch['near'])
        prod_num_samples = 1

        ray_history = []
        renderings = []
        for i_level in range(self.num_levels):
            is_prop = i_level < (self.num_levels - 1)
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

            # Dilate by some multiple of the expected span of each current interval,
            # with some bias added in.
            dilation = self.dilation_bias + self.dilation_multiplier * (
                init_s_far - init_s_near) / prod_num_samples

            # Record the product of the number of samples seen so far.
            prod_num_samples *= num_samples

            # After the first level (where dilation would be a no-op) optionally
            # dilate the interval weights along each ray slightly so that they're
            # overestimates, which can reduce aliasing.
            use_dilation = self.dilation_bias > 0 or self.dilation_multiplier > 0
            if i_level > 0 and use_dilation:
                sdist, weights = stepfun.max_dilate_weights(sdist,
                                                            weights,
                                                            dilation,
                                                            domain=(init_s_near, init_s_far),
                                                            renormalize=True)
                sdist = sdist[..., 1:-1]
                weights = weights[..., 1:-1]

            # A slightly more stable way to compute weights. If the distance
            # between adjacent intervals is zero then its weight is fixed to 0.
            logits_resample = torch.where(sdist[..., 1:] > sdist[..., :-1],
                                          torch.log(weights + self.resample_padding),
                                          torch.full_like(sdist[..., :-1], -torch.inf))

            # Draw sampled intervals from each ray's current weights.
            sdist = stepfun.sample_intervals(rand,
                                             sdist,
                                             logits_resample,
                                             num_samples,
                                             single_jitter=self.single_jitter,
                                             domain=(init_s_near, init_s_far))

            # Optimization will usually go nonlinear if you propagate gradients
            # through sampling.
            if self.stop_level_grad:
                sdist = sdist.detach()

            # Convert normalized distances to metric distances.
            tdist = s_to_t(sdist)

            # Cast our rays, by turning our distance intervals into Gaussians.
            coords, ts = render.cast_rays(tdist, batch['origins'], batch['directions'])

            # Push our Gaussians through one of our two MLPs.
            mlp = (self.prop_mlp if self.single_prop else
                   self.get_submodule(f'prop_mlp_{i_level}')) if is_prop else self.nerf_mlp
            ray_results = mlp(
                coords,
                viewdirs=batch['viewdirs'] if self.use_viewdirs else None,
            )
            if self.config.gradient_scaling:
                ray_results['rgb'], ray_results['density'] = train_utils.GradientScaler.apply(
                    ray_results['rgb'], ray_results['density'], ts.mean(dim=-1))

            # Get the alpha compositing weights used by volumetric rendering (and losses).
            weights = render.compute_alpha_weights(
                ray_results['density'],
                tdist,
                batch['directions'],
                opaque_background=self.opaque_background,
            )[0]

            # Define or sample the background color for each ray.
            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                # If the min and max of the range are equal, just take it.
                bg_rgbs = self.bg_intensity_range[0]
            elif not rand:
                # If rendering is deterministic, use the midpoint of the range.
                bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
            else:
                # Sample RGB values from the range for each ray.
                minval = self.bg_intensity_range[0]
                maxval = self.bg_intensity_range[1]
                bg_rand_t = torch.rand(weights.shape[:-1] + (3, ), device=device)
                bg_rgbs = bg_rand_t * (maxval - minval) + minval

            # Render each ray.
            rendering = render.volumetric_rendering(ray_results['rgb'],
                                                    weights,
                                                    tdist,
                                                    bg_rgbs,
                                                    batch['far'],
                                                    compute_extras,
                                                    extras={
                                                        k: v
                                                        for k, v in ray_results.items()
                                                        if k.startswith('normals')
                                                    })

            if compute_extras:
                # Collect some rays to visualize directly. By naming these quantities
                # with `ray_` they get treated differently downstream --- they're
                # treated as bags of rays, rather than image chunks.
                n = self.config.vis_num_rays
                rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
                rendering['ray_weights'] = (weights.reshape([-1, weights.shape[-1]])[:n, :])
                rgb = ray_results['rgb']
                rendering['ray_rgbs'] = (rgb.reshape((-1, ) + rgb.shape[-2:]))[:n, :, :]

            renderings.append(rendering)
            ray_results['sdist'] = sdist.clone()
            ray_results['weights'] = weights.clone()
            ray_history.append(ray_results)

        if compute_extras:
            # Because the proposal network doesn't produce meaningful colors, for
            # easier visualization we replace their colors with the final average
            # color.
            weights = [r['ray_weights'] for r in renderings]
            rgbs = [r['ray_rgbs'] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], dim=-2)
            avg_rgbs = [torch.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]]
            for i in range(len(avg_rgbs)):
                renderings[i]['ray_rgbs'] = avg_rgbs[i]

        return renderings, ray_history


class MLP(nn.Module):
    """A PosEnc MLP."""
    bottleneck_width: int = 256  # The width of the bottleneck vector.
    net_depth_viewdirs: int = 2  # The depth of the second part of ML.
    net_width_viewdirs: int = 256  # The width of the second part of MLP.
    skip_layer_dir: int = 0  # Add a skip connection to 2nd MLP after Nth layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
    use_directional_enc: bool = False  # If True, use IDE to encode directions.
    bottleneck_noise: float = 0.0  # Std. deviation of training noise added to bottleneck.
    density_bias: float = -1.  # Shift added to raw densities pre-activation.
    density_noise: float = 0.  # Standard deviation of training noise added to raw density.
    rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
    rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    disable_density_normals: bool = False  # If True don't compute normals.
    disable_rgb: bool = False  # If True don't output RGB.
    bbox_size: float = 4.  # The side length of the bounding box if warp is not used.
    warp_fn = 'contract'  # The warp function used to warp the input coordinates.

    # Configs for grid encoder
    grid_num_levels: int = 10
    grid_level_interval: int = 2
    grid_level_dim: int = 4
    grid_base_resolution: int = 16
    grid_disired_resolution: int = 8192
    grid_log2_hashmap_size: int = 21

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.grid_num_levels = int(
            np.log(self.grid_disired_resolution / self.grid_base_resolution) /
            np.log(self.grid_level_interval)) + 1
        self.encoder = GridEncoder(input_dim=3,
                                   num_levels=self.grid_num_levels,
                                   level_dim=self.grid_level_dim,
                                   base_resolution=self.grid_base_resolution,
                                   desired_resolution=self.grid_disired_resolution,
                                   log2_hashmap_size=self.grid_log2_hashmap_size,
                                   gridtype='hash',
                                   align_corners=False)
        last_dim = self.encoder.output_dim

        self.density_layer = nn.Sequential(
            nn.Linear(last_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1 if self.disable_rgb else self.bottleneck_width),
        )  # Hardcoded to a single channel.
        last_dim = 1 if self.disable_rgb else self.bottleneck_width

        # Precompute and define viewdir encoding function.
        self.dir_enc_fn = lambda d: coord.pos_enc(
            d, min_deg=0, max_deg=self.deg_view, with_identity=True)
        dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3)).shape[-1]

        if not self.disable_rgb:
            # Output of the first part of MLP.
            if self.bottleneck_width > 0:
                last_dim_rgb = self.bottleneck_width
            else:
                last_dim_rgb = 0

            last_dim_rgb += dim_dir_enc
            input_dim_rgb = last_dim_rgb
            for i in range(self.net_depth_viewdirs):
                lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                torch.nn.init.kaiming_uniform_(lin.weight)
                self.register_module(f"lin_second_stage_{i}", lin)
                last_dim_rgb = self.net_width_viewdirs
                if i == self.skip_layer_dir:
                    last_dim_rgb += input_dim_rgb
            self.rgb_layer = nn.Linear(last_dim_rgb, self.num_rgb_channels)

    def predict_density(self, coords, no_warp=False):
        """Helper function to output density."""
        # Encode input positions
        if self.warp_fn is None or no_warp:
            bound = self.bbox_size / 2
            coords = coords / bound
        elif self.warp_fn == 'contract':
            coords = coord.contract(coords)
            bound = 2.0
            coords = coords / bound  # contract [-2, 2] to [-1, 1]
        else:
            raise NotImplementedError(f'Unknown warp function {self.warp_fn}')

        features = self.encoder(coords, bound=bound)
        x = self.density_layer(features)
        raw_density = x[..., 0]  # Hardcoded to a single channel.
        # Add noise to regularize the density predictions if needed.
        if self.training and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn_like(raw_density)
        return raw_density, x, coords

    def forward(self, coords, viewdirs=None, no_warp=False):
        """Evaluate the MLP.

        Args:
            coords: [..., 3], coordinates.
            viewdirs: [..., 3], if not None, this variable will
                be part of the input to the second part of the MLP concatenated with the
                output vector of the first part of the MLP. If None, only the first part
                of the MLP will be used with input x. In the original paper, this
                variable is the view direction.
            no_warp: bool, if True, don't warp the input coordinates.

        Returns:
            rgb: [..., num_rgb_channels].
            density: [...].
            normals: [..., 3], or None.
        """
        if self.disable_density_normals:
            raw_density, x, coords_warped = self.predict_density(coords, no_warp=no_warp)
            raw_grad_density = None
            normals = None
        else:
            with torch.enable_grad():
                coords.requires_grad_(True)
                raw_density, x, coords_warped = self.predict_density(coords, no_warp=no_warp)
                d_output = torch.ones_like(raw_density,
                                           requires_grad=False,
                                           device=raw_density.device)
                raw_grad_density = torch.autograd.grad(outputs=raw_density,
                                                       inputs=coords,
                                                       grad_outputs=d_output,
                                                       create_graph=True,
                                                       retain_graph=True,
                                                       only_inputs=True)[0]
            raw_grad_density = raw_grad_density.mean(-2)
            # Compute normal vectors as negative normalized density gradient.
            # We normalize the gradient of raw (pre-activation) density because
            # it's the same as post-activation density, but is more numerically stable
            # when the activation function has a steep or flat gradient.
            normals = -torch.nn.functional.normalize(
                raw_grad_density, dim=-1, eps=torch.finfo(x.dtype).eps)

        # Apply bias and activation to raw density
        density = F.softplus(raw_density + self.density_bias)

        if self.disable_rgb:
            rgb = torch.zeros(density.shape + (3, ), device=density.device)
        else:
            if viewdirs is not None:
                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = x
                    # Add bottleneck noise.
                    if self.training and (self.bottleneck_noise > 0):
                        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)

                    x = [bottleneck]
                else:
                    x = []

                # Encode view directions.
                dir_enc = self.dir_enc_fn(viewdirs)
                dir_enc = torch.broadcast_to(dir_enc[..., None, :],
                                             bottleneck.shape[:-1] + (dir_enc.shape[-1], ))

                # Append view direction encoding to bottleneck vector.
                x.append(dir_enc)

                # Concatenate bottleneck, directional encoding, and GLO.
                x = torch.cat(x, dim=-1)
                # Output of the second part of MLP.
                inputs = x
                for i in range(self.net_depth_viewdirs):
                    x = self.get_submodule(f"lin_second_stage_{i}")(x)
                    x = F.relu(x)
                    if i == self.skip_layer_dir:
                        x = torch.cat([x, inputs], dim=-1)
            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            rgb = torch.sigmoid(self.rgb_premultiplier * self.rgb_layer(x) + self.rgb_bias)

            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        hash_levelwise_mean = None
        if self.training:
            # Compute the hash decay loss for this level.
            param_sq = torch.square(self.encoder.embeddings)
            hash_levelwise_mean = torch.zeros(self.encoder.num_levels,
                                              param_sq.shape[-1],
                                              device=param_sq.device)
            try:
                # Try use faster torch_scatter's scatter_coo first
                from torch_scatter import segment_coo
                hash_levelwise_mean = segment_coo( \
                    param_sq, self.encoder.idx, out=hash_levelwise_mean, reduce='mean')
            except:
                # Fall back to pytorch's scatter_reduce
                hash_levelwise_mean = hash_levelwise_mean.scatter_reduce_( \
                    0, self.encoder.idx.unsqueeze(1), param_sq, reduce='mean', include_self=False)

        return dict(coord=coords_warped,
                    density=density,
                    rgb=rgb,
                    raw_grad_density=raw_grad_density,
                    normals=normals,
                    hash_levelwise_mean=hash_levelwise_mean)


@gin.configurable
class NerfMLP(MLP):
    pass


@gin.configurable
class PropMLP(MLP):
    pass


@torch.no_grad()
def render_image(model: Model,
                 accelerator: accelerate.Accelerator,
                 batch,
                 config,
                 verbose=True,
                 return_weights=False):
    """Render all the pixels of an image (in test mode).

    Args:
        model: The rendering model.
        accelerator: used for DDP.
        batch: a `Rays` pytree, the rays to be rendered.
        config: A Config class.

    Returns:
        rgb: rendered color image.
        disp: rendered disparity image.
        acc: rendered accumulated weights per pixel.
    """
    model.eval()

    height, width = batch['origins'].shape[:2]
    num_rays = height * width
    batch = {k: v.reshape((num_rays, -1)) for k, v in batch.items() if v is not None}

    global_rank = accelerator.process_index
    chunks = []
    idx0s = tqdm(range(0, num_rays, config.render_chunk_size),
                 desc="Rendering chunk",
                 leave=False,
                 disable=not (accelerator.is_main_process and verbose))

    for i_chunk, idx0 in enumerate(idx0s):
        chunk_batch = tree_map(lambda r: r[idx0:idx0 + config.render_chunk_size], batch)
        actual_chunk_size = chunk_batch['origins'].shape[0]
        rays_remaining = actual_chunk_size % accelerator.num_processes
        if rays_remaining != 0:
            padding = accelerator.num_processes - rays_remaining
            chunk_batch = tree_map(lambda v: torch.cat([v, torch.zeros_like(v[-padding:])], dim=0),
                                   chunk_batch)
        else:
            padding = 0
        # After padding the number of chunk_rays is always divisible by host_count.
        rays_per_host = chunk_batch['origins'].shape[0] // accelerator.num_processes
        start, stop = global_rank * rays_per_host, (global_rank + 1) * rays_per_host
        chunk_batch = tree_map(lambda r: r[start:stop], chunk_batch)

        with accelerator.autocast():
            chunk_renderings, ray_history = model(chunk_batch, compute_extras=True)

        gather = lambda v: accelerator.gather(v.contiguous())[:-padding] \
            if padding > 0 else accelerator.gather(v.contiguous())
        # Unshard the renderings.
        chunk_renderings = tree_map(gather, chunk_renderings)

        # Gather the final pass for 2D buffers and all passes for ray bundles.
        chunk_rendering = chunk_renderings[-1]
        for k in chunk_renderings[0]:
            if k.startswith('ray_'):
                chunk_rendering[k] = [r[k] for r in chunk_renderings]

        if return_weights:
            chunk_rendering['weights'] = gather(ray_history[-1]['weights'])
            chunk_rendering['coord'] = gather(ray_history[-1]['coord'])
        chunks.append(chunk_rendering)

    # Concatenate all chunks within each leaf of a single pytree.
    rendering = {}
    for k in chunks[0].keys():
        if isinstance(chunks[0][k], list):
            rendering[k] = []
            for i in range(len(chunks[0][k])):
                rendering[k].append(torch.cat([item[k][i] for item in chunks]))
        else:
            rendering[k] = torch.cat([item[k] for item in chunks])

    for k, z in rendering.items():
        if not k.startswith('ray_'):
            # Reshape 2D buffers into original image shape.
            rendering[k] = z.reshape((height, width) + z.shape[1:])

    # After all of the ray bundles have been concatenated together, extract a
    # new random bundle (deterministically) from the concatenation that is the
    # same size as one of the individual bundles.
    keys = [k for k in rendering if k.startswith('ray_')]
    if keys:
        num_rays = rendering[keys[0]][0].shape[0]
        ray_idx = torch.randperm(num_rays)
        ray_idx = ray_idx[:config.vis_num_rays]
        for k in keys:
            rendering[k] = [r[ray_idx] for r in rendering[k]]

    model.train()
    return rendering
