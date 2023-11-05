import os
import sys
import gin
import time
import torch
import logging
import functools
import accelerate
import numpy as np
from torch.utils import tensorboard
from torch.utils._pytree import tree_map
from torch.utils.data.dataloader import DataLoader
from source import models
from source import configs
from source import datasets
from source import checkpoints
from source import vis
from source.utils import image as image_utils
from source.utils import misc


def summarize_results(folder, scene_names, num_buckets):
    metric_names = ['psnrs', 'ssims', 'lpips']
    num_iters = 1000000
    precisions = [3, 4, 4, 4]

    results = []
    for scene_name in scene_names:
        test_preds_folder = os.path.join(folder, scene_name, 'test_preds')
        values = []
        for metric_name in metric_names:
            filename = os.path.join(folder, scene_name, 'test_preds',
                                    f'{metric_name}_{num_iters}.txt')
            with misc.open_file(filename) as f:
                v = np.array([float(s) for s in f.readline().split(' ')])
                values.append(np.mean(np.reshape(v, [-1, num_buckets]), 0))
        results.append(np.concatenate(values))
    avg_results = np.mean(np.array(results), 0)

    psnr, ssim, lpips = np.mean(np.reshape(avg_results, [-1, num_buckets]), 1)

    mse = np.exp(-0.1 * np.log(10.) * psnr)
    dssim = np.sqrt(1 - ssim)
    avg_avg = np.exp(np.mean(np.log(np.array([mse, dssim, lpips]))))

    s = []
    for i, v in enumerate(np.reshape(avg_results, [-1, num_buckets])):
        s.append(' '.join([f'{s:0.{precisions[i]}f}' for s in v]))
    s.append(f'{avg_avg:0.{precisions[-1]}f}')
    return ' | '.join(s)


def main():
    config = configs.load_config(rank=0, world_size=1)

    accelerator = accelerate.Accelerator()
    torch.backends.cudnn.benchmark = True  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.

    # setup logger
    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(config.exp_path, 'log_eval.txt'))
        ],
        level=logging.INFO,
    )
    sys.excepthook = misc.handle_exception
    logger = accelerate.logging.get_logger(__name__)
    logger.info(config)
    logger.info(accelerator.state, main_process_only=False)

    # Set random seed.
    accelerate.utils.set_seed(config.seed, device_specific=True)
    # setup model and optimizer
    model = models.Model(config=config)
    model.eval()

    dataset = datasets.load_dataset('test', config)
    dataloader = DataLoader(
        np.arange(len(dataset)),
        shuffle=False,
        batch_size=1,
        collate_fn=dataset.collate_fn,
    )
    tb_process_fn = lambda x: x.transpose(2, 0, 1) if len(x.shape) == 3 else x[None]

    # use accelerate to prepare.
    model, dataloader = accelerator.prepare(model, dataloader)

    # metric handler
    metric_harness = image_utils.MetricHarness()
    
    last_step = 0
    out_dir = os.path.join(config.exp_path, 'path_renders' if config.render_path else 'test_preds')
    path_fn = lambda x: os.path.join(out_dir, x)
    if not config.eval_only_once:
        summary_writer = tensorboard.SummaryWriter(os.path.join(config.exp_path, 'eval'))
        
    # Use more samples for evaluation.
    model.num_prop_samples *= config.eval_sample_multipler
    model.num_nerf_samples *= config.eval_sample_multipler
    
    # Evaluation loop
    while True:
        step = checkpoints.restore_checkpoint(config.ckpt_dir, accelerator, logger)
        if step <= last_step:
            logger.info(f'Checkpoint step {step} <= last step {last_step}, sleeping.')
            time.sleep(10)
            continue
        logger.info(f'Evaluating checkpoint at step {step}.')
        if config.eval_save_output and (not misc.isdir(out_dir)):
            misc.makedirs(out_dir)

        num_eval = min(dataset.size, config.eval_dataset_limit)
        perm = np.random.permutation(num_eval)
        showcase_indices = np.sort(perm[:config.num_showcase_images])
        metrics = []
        showcases = []
        render_times = []
        for idx, batch in enumerate(dataloader):
            batch = accelerate.utils.send_to_device(batch, accelerator.device)
            eval_start_time = time.time()
            if idx >= num_eval:
                logger.info(f'Skipping image {idx + 1}/{dataset.size}')
                continue
            logger.info(f'Evaluating image {idx + 1}/{dataset.size}')
            
            rendering = models.render_image(model, accelerator, batch, config)

            if not accelerator.is_main_process:  # Only record via host 0.
                continue

            render_times.append((time.time() - eval_start_time))
            logger.info(f'Rendered in {render_times[-1]:0.3f}s')

            rendering = tree_map(lambda x: x.detach().cpu().numpy()
                                 if x is not None else None, rendering)
            batch = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, batch)

            if not config.eval_only_once and idx in showcase_indices:
                showcase_idx = idx if config.deterministic_showcase else len(showcases)
                showcases.append((showcase_idx, rendering, batch))
            if not config.render_path:
                rgb = rendering['rgb']
                rgb_gt = batch['rgb']

                if config.eval_quantize_metrics:
                    # Ensures that the images written to disk reproduce the metrics.
                    rgb = np.round(rgb * 255) / 255

                if config.eval_crop_borders > 0:
                    crop_fn = lambda x, c=config.eval_crop_borders: x[c:-c, c:-c]
                    rgb = crop_fn(rgb)
                    rgb_gt = crop_fn(rgb_gt)

                metric = metric_harness(rgb, rgb_gt)
                for m, v in metric.items():
                    logger.info(f'{m:30s} = {v:.4f}')
                metrics.append(metric)

            if config.eval_save_output and (config.eval_render_interval > 0):
                if (idx % config.eval_render_interval) == 0:
                    misc.save_img_u8(rendering['rgb'], path_fn(f'color_{idx:03d}.png'))

                    for key in ['distance_mean', 'distance_median']:
                        if key in rendering:
                            misc.save_img_f32(rendering[key], path_fn(f'{key}_{idx:03d}.tiff'))

                    for key in ['normals']:
                        if key in rendering:
                            misc.save_img_u8(rendering[key] / 2. + 0.5,
                                             path_fn(f'{key}_{idx:03d}.png'))

                    misc.save_img_f32(rendering['acc'], path_fn(f'acc_{idx:03d}.tiff'))

        if (not config.eval_only_once) and accelerator.is_main_process:
            summary_writer.add_scalar('eval_median_render_time', np.median(render_times), step)
            for name in metrics[0]:
                scores = [m[name] for m in metrics]
                summary_writer.add_scalar('eval_metrics/' + name, np.mean(scores), step)
                summary_writer.add_histogram('eval_metrics/' + 'perimage_' + name, scores, step)

            for i, r, b in showcases:
                if config.vis_decimate > 1:
                    d = config.vis_decimate
                    decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
                else:
                    decimate_fn = lambda x: x
                r = tree_map(decimate_fn, r)
                b = tree_map(decimate_fn, b)
                visualizations = vis.visualize_suite(r, b)
                for k, v in visualizations.items():
                    summary_writer.add_image(f'output_{k}_{i}', tb_process_fn(v), step)
                if not config.render_path:
                    target = b['rgb']
                    summary_writer.add_image(f'true_color_{i}', tb_process_fn(target), step)
                    pred = visualizations['color']
                    residual = np.clip(pred - target + 0.5, 0, 1)
                    summary_writer.add_image(f'true_residual_{i}', tb_process_fn(residual), step)

        if (config.eval_save_output and (not config.render_path) and accelerator.is_main_process):
            with misc.open_file(path_fn(f'render_times_{step}.txt'), 'w') as f:
                f.write(' '.join([str(r) for r in render_times]))
            logger.info(f'metrics:')
            results = {}
            num_buckets = config.multiscale_levels if config.multiscale else 1
            for name in metrics[0]:
                with misc.open_file(path_fn(f'metric_{name}_{step}.txt'), 'w') as f:
                    ms = [m[name] for m in metrics]
                    f.write(' '.join([str(m) for m in ms]))
                    results[name] = ' | '.join(
                        list(map(str,
                                 np.mean(np.array(ms).reshape([-1, num_buckets]), 0).tolist())))
            with misc.open_file(path_fn(f'metric_avg_{step}.txt'), 'w') as f:
                for name in metrics[0]:
                    f.write(f'{name}: {results[name]}\n')
                    logger.info(f'{name}: {results[name]}')

            if config.eval_save_ray_data:
                for i, r, b in showcases:
                    rays = {k: v for k, v in r.items() if 'ray_' in k}
                    np.set_printoptions(threshold=sys.maxsize)
                    with misc.open_file(path_fn(f'ray_data_{step}_{i}.txt'), 'w') as f:
                        f.write(repr(rays))

        if config.eval_only_once:
            break
        if config.early_exit_steps is not None:
            num_steps = config.early_exit_steps
        else:
            num_steps = config.max_steps
        if int(step) >= num_steps:
            break
        last_step = step
    logger.info('Finish evaluation.')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', nargs='+', help="Path to gin config files")
    p.add_argument('-p', '--param', nargs='+', help="Command line parameter override")
    args = p.parse_args()

    gin.parse_config_files_and_bindings(args.config, args.param, skip_unknown=True)
    with gin.config_scope('eval'):
        main()