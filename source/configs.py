import os
import gin
import torch
import dataclasses
import numpy as np
from typing import Any, Callable, Optional
from source.utils import misc

gin.add_config_file_search_path('configs/')

configurables = {
    'torch': [torch.reciprocal, torch.log, torch.log1p, torch.exp, torch.sqrt, torch.square],
}

for module, configurables in configurables.items():
    for configurable in configurables:
        gin.config.external_configurable(configurable, module=module)


@gin.configurable()
@dataclasses.dataclass
class Config:
    """Configuration flags for everything."""
    seed = 42  # Random seed.

    # Dataset configs
    data_dir: Optional[str] = "data/nerf_synthetic/lego"  # Input data directory.
    dataset_loader: str = 'blender'  # The type of dataset loader to use.
    batching: str = 'all_images'  # Batch composition, [single_image, all_images].
    batch_size: int = 2**14  # The number of rays/pixels in each batch.
    patch_size: int = 1  # Resolution of patches sampled for training batches.
    factor: int = 4  # The downsample factor of images, 0 for no downsampling.
    multiscale: bool = False  # use multiscale data for training.
    multiscale_levels: int = 4  # number of multiscale levels.
    forward_facing: bool = False  # Set to True for forward-facing LLFF captures.
    render_path: bool = False  # If True, render a path. Used only by LLFF.
    load_alphabetical: bool = True  # Load images in COLMAP vs alphabetical ordering (affects heldout test set).
    llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
    llff_use_all_images_for_training: bool = False  # If true, use all input images for training.
    llff_use_all_images_for_testing: bool = False  # If true, use all input images for testing.
    use_tiffs: bool = False  # If True, use 32-bit TIFFs. Used only by Blender.
    near: float = 2.  # Near plane distance.
    far: float = 6.  # Far plane distance.

    # Common configs
    exp_name : str = 'test'  # Experiment name
    render_chunk_size: int = 16384  # Chunk size for whole-image renderings.
    vis_num_rays: int = 16  # The number of rays to visualize.
    vis_decimate: int = 0  # Decimate images for tensorboard (ie, x[::d, ::d]) to conserve memory usage.

    # Train configs
    max_steps: int = 25000  # The number of optimization steps.
    early_exit_steps: Optional[int] = None  # Early stopping, for debugging.
    checkpoint_every: int = 5000  # The number of steps to save a checkpoint.
    resume_from_checkpoint: bool = True  # whether to resume from checkpoint.
    load_checkpoint: str = ''  # If not empty, load weights from this checkpoint.
    checkpoints_total_limit: int = 1
    gradient_scaling: bool = False  # If True, scale gradients as in https://gradient-scaling.github.io/.
    print_every: int = 100  # The number of steps between reports to tensorboard.
    train_render_every: int = 500  # Steps between test set renders when training
    data_loss_type: str = 'charb'  # What kind of loss to use ('mse' or 'charb').
    charb_padding: float = 0.001  # The padding used for Charbonnier loss.
    huber_delta: float = 1.0  # The threshold to change between delta-scaled L1 and L2 loss.
    data_loss_mult: float = 1.0  # Mult for the finest data term in the loss.
    data_coarse_loss_mult: float = 0.  # Multiplier for the coarser data terms.
    mask_loss_mult: float = 0.  # Multiplier for the mask loss.
    sinkhorn_loss_mult: float = 0.0  # Mult for the sinkohrn loss.
    sinkhorn_patch_size: int = 24  # Patch size for the sinkhorn loss.
    interlevel_loss_mult: float = 0.0  # Mult. for the loss on the proposal MLP.
    anti_interlevel_loss_mult: float = 0.01  # Mult. for the loss on the proposal MLP.
    pulse_width = [0.03, 0.003]  # Mult. for the loss on the proposal MLP.
    train_sample_multipler_init: float = 1.  # Initial sample multiplier.
    train_sample_final_frac: float = 0.9  # Train fraction to reach final sample multiplier.
    fix_shape_params: bool = False  # If True, fix the shape parameters of the stroke field.
    fix_color_params: bool = False  # If True, fix the color parameters of the stroke field.
    fix_density_params: bool = False  # If True, fix the densities of the stroke field.
    
    lr_init: float = 0.01  # The initial learning rate.
    lr_final: float = 0.003  # The final learning rate.
    lr_delay_steps: int = 0  # The number of "warmup" learning steps.
    lr_delay_mult: float = 1e-8  # How much sever the "warmup" should be.
    adam_beta1: float = 0.9  # Adam's beta2 hyperparameter.
    adam_beta2: float = 0.99  # Adam's beta2 hyperparameter.
    adam_eps: float = 1e-15  # Adam's epsilon hyperparameter.
    grad_max_norm: float = 0.  # Gradient clipping magnitude, disabled if == 0.
    grad_max_val: float = 0.  # Gradient clipping value, disabled if == 0.
    distortion_loss_mult: float = 0.005  # Multiplier on the distortion loss.
    opacity_loss_mult: float = 0.  # Multiplier on the distortion loss.
    hash_decay_mult: float = 0.1  # Mult. for the loss on the hash feature decay.
    error_loss_mult: float = 0.1  # Multiplier on the error loss.
    error_loss_lower_lambda: float = 10.0  # Multiplier on the lower error loss.
    density_reg_loss_mult: float = 0.  # Multiplier on the density regularization loss.
    style_loss_mult: float = 0.  # Multiplier on the perceptual style loss.
    style_target_image: str = ''  # A path to the target style image.
    style_transfer_shape: bool = False  # If True, transfer the style of the shape.
    clip_loss_mult: float = 0.  # Multiplier on the clip loss.
    clip_negative_mult: float = 0.3  # Multiplier on the negative clip prompts.
    clip_positive_prompt: str = ''  # The positive prompt of clip target, can be multiple sentences separated by ','.
    clip_negative_prompt: str = ''  # The negative prompt of clip target, can be multiple sentences separated by ','.
    transmittance_loss_mult: float = 0.  # Multiplier on the transmittance loss.
    transmittance_target: float = 0.88  # Target transmittance value.
    entropy_loss_mult: float = 0.  # Multiplier on the entropy loss.
    diffusion_loss_mult: float = 0.  # Multiplier on the SDS loss.
    diffusion_positive_prompt: str = ''  # The positive prompt of SDS loss.
    diffusion_negative_prompt: str = ''  # The negative prompt of SDS loss.
    diffusion_t_range: tuple[float, float] = (0.02, 0.98)  # The range of t values to use for SDS.
    diffusion_model_use_fp16: bool = True  # If True, use fp16 for the diffusion model.

    # Eval configs
    num_showcase_images: int = 5  # The number of test-set images to showcase.
    deterministic_showcase: bool = True  # If True, showcase the same images.
    eval_only_once: bool = True  # If True evaluate the model only once, ow loop.
    eval_save_output: bool = True  # If True save predicted images to disk.
    eval_save_ray_data: bool = False  # If True save individual ray traces.
    eval_render_interval: int = 1  # The interval between images saved to disk.
    eval_dataset_limit: int = np.iinfo(np.int32).max  # Num test images to eval.
    eval_quantize_metrics: bool = True  # If True, run metrics on 8-bit images.
    eval_crop_borders: int = 0  # Ignore c border pixels in eval (x[c:-c, c:-c]).
    eval_sample_multipler: float = 2.  # Multiplier for the number of samples.

    # Render configs
    render_progressive_strokes: bool = True  # If True, render strokes progressively.
    render_progressive_sample_multipler: float = 12.  # Multiplier for the number of samples.
    render_progressive_render_chunk_size_divisor: int = 16  # Divisor for render chunk size.
    render_factor: int = -1  # The downsample factor of rendered images, -1 for not used.
    render_video_fps: int = 30  # Framerate in frames-per-second.
    render_video_crf: int = 18  # Constant rate factor for ffmpeg video quality.
    render_path_frames: int = 120  # Number of frames in render path.
    z_variation: float = 0.  # How much height variation in render path.
    z_phase: float = 0.  # Phase offset for height variation in render path.
    render_dist_percentile: float = 0.5  # How much to trim from near/far planes.
    render_dist_curve_fn: Callable[..., Any] = np.log  # How depth is curved.
    render_path_file: Optional[str] = None  # Numpy render pose file to load.
    render_resolution: Optional[tuple[int, int]] = None  # Render resolution, as (width, height).
    render_focal: Optional[float] = None  # Render focal length.
    render_camtype: Optional[str] = None  # 'perspective', 'fisheye', or 'pano'.
    render_spherical: bool = False  # Render spherical 360 panoramas.
    render_save_async: bool = True  # Save to CNS using a separate thread.
    render_spline_keyframes: Optional[str] = None  # Text file containing names of images to be used as spline keyframes, OR directory containing those images.
    render_spline_n_interp: int = 30  # Num. frames to interpolate per keyframe.
    render_spline_degree: int = 5  # Polynomial degree of B-spline interpolation.
    render_spline_smoothness: float = .03  # B-spline smoothing factor, 0 for exact interpolation of keyframes.
    render_spline_interpolate_exposure: bool = False  # Interpolate per-frame exposure value from spline keyframes.


def load_config(rank: int, world_size: int) -> Config:
    config = Config()
    
    # Validate configs
    if config.batch_size % world_size != 0:
        raise ValueError(f"Batch size {config.batch_size} must be divisible by the number of processes {world_size}")
        
    # Set up temporary variables
    config.exp_path = os.path.join("exp", config.exp_name)
    config.ckpt_dir = os.path.join(config.exp_path, 'checkpoints')
    config.output_dir = os.path.join(config.exp_path, 'outputs')
    config.render_dir = os.path.join(config.exp_path, 'render')
    config.global_rank = rank  # Distributed process rank.
    config.world_size = world_size # Number of processes for distributed training.
    
    # Dump a copy of the config to the experiment directory.
    if rank == 0:
        os.makedirs(config.exp_path, exist_ok=True)
        dumped_cfg_path = os.path.join(config.exp_path, 'config.gin')
        with misc.open_file(dumped_cfg_path, 'w') as f:
            f.write(gin.config_str())
    
    return config