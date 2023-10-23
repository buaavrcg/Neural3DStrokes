import os
import sys
import gin
import time
import torch
import logging
import functools
import accelerate
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils import tensorboard
from torch.utils._pytree import tree_map, tree_flatten
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from source import models
from source import configs
from source import datasets
from source import checkpoints
from source import vis
from source import loss as loss_fn
from source.utils import training as train_utils
from source.utils import image as image_utils
from source.utils import misc


def train():
    cfg = configs.load_config(rank=0, world_size=1)
    misc.makedirs(cfg.exp_path)

    # accelerator for DDP
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
            logging.FileHandler(os.path.join(cfg.exp_path, 'log_train.txt'))
        ],
        level=logging.INFO,
    )
    sys.excepthook = misc.handle_exception
    logger = accelerate.logging.get_logger(__name__)
    logger.info(cfg)
    logger.info(accelerator.state, main_process_only=False)

    # Set random seed.
    accelerate.utils.set_seed(cfg.seed, device_specific=True)
    # setup model and optimizer
    model = models.Model(config=cfg)
    optimizer, lr_fn = train_utils.create_optimizer(cfg, model.parameters())

    # load dataset
    trainset = datasets.load_dataset('train', cfg)
    testset = datasets.load_dataset('test', cfg)
    trainloader = DataLoader(np.arange(len(trainset)),
                             num_workers=8,
                             sampler=train_utils.InfiniteSampler(trainset),
                             batch_size=1,
                             persistent_workers=True,
                             collate_fn=trainset.collate_fn)
    testloader = DataLoader(np.arange(len(testset)),
                            num_workers=4,
                            sampler=train_utils.InfiniteSampler(testset, shuffle=False),
                            batch_size=1,
                            persistent_workers=True,
                            collate_fn=testset.collate_fn)

    # use accelerate to prepare.
    model, trainloader, testloader, optimizer = \
        accelerator.prepare(model, trainloader, testloader, optimizer)

    if cfg.resume_from_checkpoint:
        init_step = checkpoints.restore_checkpoint(cfg.ckpt_dir, accelerator, logger)
    else:
        init_step = 0

    module = accelerator.unwrap_model(model)
    trainiter = iter(trainloader)
    testiter = iter(testloader)

    def tree_len(tree):
        tree_reduce = lambda fn, tree: functools.reduce(fn, tree_flatten(tree)[0], 0)
        tree_sum = lambda tree: tree_reduce(lambda x, y: x + y, tree)
        return tree_sum(tree_map(lambda z: np.prod(z.shape), tree))

    num_params = tree_len(list(model.parameters()))
    logger.info(f'Number of parameters being optimized: {num_params}')

    # metric handler
    metric_harness = image_utils.MetricHarness()

    # tensorboard
    if accelerator.is_main_process:
        summary_writer = tensorboard.SummaryWriter(cfg.exp_path)

    logger.info("Begin training...")
    step = init_step + 1
    total_time = 0
    total_steps = 0
    reset_stats = True
    num_steps = cfg.max_steps
    model.step_update(cur_step=step, max_step=num_steps)
    with logging_redirect_tqdm():
        for step in tqdm(range(init_step + 1, num_steps + 1),
                         desc='Training',
                         initial=init_step,
                         total=num_steps,
                         disable=not accelerator.is_main_process):
            batch = next(trainiter)
            if reset_stats and accelerator.is_main_process:
                stats_buffer = []
                train_start_time = time.time()
                reset_stats = False

            # use lr_fn to control learning rate
            learning_rate = lr_fn(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            with accelerator.autocast():
                renderings, ray_history = model(batch, compute_extras=False)

            # apply loss functions
            loss, stats = apply_loss(batch, renderings, ray_history, module, cfg)

            # accelerator automatically handle the scale
            accelerator.backward(loss)
            # clip gradient by max/norm/nan
            train_utils.clip_gradients(model, accelerator, cfg)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model.step_update(cur_step=step, max_step=num_steps)

            # Log training summaries. This is put behind a host_id check because in
            # multi-host evaluation, all hosts need to run inference even though we
            # only use host 0 to record results.
            if accelerator.is_main_process:
                stats_buffer.append(stats)
                if step == init_step + 1 or step % cfg.print_every == 0:
                    total_time, total_steps = log_training(train_start_time, stats_buffer,
                                                           learning_rate, cfg, summary_writer,
                                                           logger, step, total_time, total_steps)
                    # Reset everything we are tracking between summarizations.
                    reset_stats = True

                if step > 0 and step % cfg.checkpoint_every == 0:
                    checkpoints.save_checkpoint(cfg.ckpt_dir, accelerator, step,
                                                cfg.checkpoints_total_limit)

            # Test-set evaluation.
            if cfg.train_render_every > 0 and step % cfg.train_render_every == 0:
                train_eval(next(testiter), model, metric_harness, accelerator, cfg, summary_writer,
                           logger, step)

    if accelerator.is_main_process and cfg.max_steps > init_step:
        logger.info(f'Saving last checkpoint at step {step} to {cfg.ckpt_dir}')
        checkpoints.save_checkpoint(cfg.ckpt_dir, accelerator, step, cfg.checkpoints_total_limit)
    logger.info('Finish training.')


def apply_loss(batch, renderings, ray_history, module, cfg) -> tuple[torch.Tensor, dict]:
    losses = {}

    # supervised by data
    data_loss, stats = loss_fn.compute_data_loss(batch, renderings, ray_history, cfg)
    losses['data'] = data_loss

    # interlevel loss in MipNeRF360
    if cfg.interlevel_loss_mult > 0 and not module.single_mlp:
        losses['interlevel'] = loss_fn.interlevel_loss(ray_history, cfg)

    # interlevel loss in ZipNeRF360
    if cfg.anti_interlevel_loss_mult > 0 and not module.single_mlp:
        losses['anti_interlevel'] = loss_fn.anti_interlevel_loss(ray_history, cfg)

    # distortion loss
    if cfg.distortion_loss_mult > 0:
        losses['distortion'] = loss_fn.distortion_loss(ray_history, cfg)

    # opacity loss
    if cfg.opacity_loss_mult > 0:
        losses['opacity'] = loss_fn.opacity_reg_loss(renderings, cfg)

    # hash grid l2 weight decay
    if cfg.hash_decay_mult > 0:
        losses['hash_decay'] = loss_fn.hash_decay_loss(ray_history, cfg)
        
    # error field loss
    if cfg.error_loss_mult > 0:
        losses['error'] = loss_fn.error_loss(batch, renderings, cfg)

    loss = sum(losses.values())
    stats['loss'] = loss.item()
    stats['losses'] = tree_map(lambda x: x.item(), losses)

    stats['psnrs'] = image_utils.mse_to_psnr(stats['mses'])
    stats['psnr'] = stats['psnrs'][-1]

    return loss, stats


def log_training(train_start_time, stats_buffer, learning_rate, cfg, summary_writer, logger, step,
                 total_time, total_steps):
    elapsed_time = time.time() - train_start_time
    steps_per_sec = cfg.print_every / elapsed_time
    rays_per_sec = cfg.batch_size * steps_per_sec

    # A robust approximation of total training time, in case of pre-emption.
    total_time += int(round(1000 * elapsed_time))
    total_steps += cfg.print_every
    approx_total_time = int(round(step * total_time / total_steps))

    # Transpose and stack stats_buffer along axis 0.
    fs = [misc.flatten_dict(s, sep='/') for s in stats_buffer]
    stats_stacked = {k: np.stack([f[k] for f in fs]) for k in fs[0].keys()}

    # Split every statistic that isn't a vector into a set of statistics.
    stats_split = {}
    for k, v in stats_stacked.items():
        if v.ndim not in [1, 2] and v.shape[0] != len(stats_buffer):
            raise ValueError('statistics must be of size [n], or [n, k].')
        if v.ndim == 1:
            stats_split[k] = v
        elif v.ndim == 2:
            for i, vi in enumerate(tuple(v.T)):
                stats_split[f'{k}/{i}'] = vi

    # Summarize the entire histogram of each statistic.
    for k, v in stats_split.items():
        summary_writer.add_histogram('train_' + k, v, step)

    # Take the mean and max of each statistic since the last summary.
    avg_stats = {k: np.mean(v) for k, v in stats_split.items()}
    max_stats = {k: np.max(v) for k, v in stats_split.items()}

    summ_fn = lambda s, v: summary_writer.add_scalar(s, v, step)

    # Summarize the mean and max of each statistic.
    for k, v in avg_stats.items():
        summ_fn(f'train_avg_{k}', v)
    for k, v in max_stats.items():
        summ_fn(f'train_max_{k}', v)

    summ_fn('train_learning_rate', learning_rate)
    summ_fn('train_steps_per_sec', steps_per_sec)
    summ_fn('train_rays_per_sec', rays_per_sec)

    summary_writer.add_scalar('train_avg_psnr_timed', avg_stats['psnr'], total_time // 1000)
    summary_writer.add_scalar('train_avg_psnr_timed_approx', avg_stats['psnr'],
                              approx_total_time // 1000)

    avg_loss = avg_stats['loss']
    avg_psnr = avg_stats['psnr']
    str_losses = {  # Grab each "losses_{x}" field and print it as "x[:4]".
        k[7:11]: (f'{v:0.5f}' if 1e-4 <= v < 10 else f'{v:0.1e}')
        for k, v in avg_stats.items() if k.startswith('losses/')
    }
    logger.info(f'{step}' + f'/{cfg.max_steps:d}:' + f'loss={avg_loss:0.5f},' +
                f'psnr={avg_psnr:.3f},' + f'lr={learning_rate:0.2e} | ' +
                ','.join([f'{k}={s}' for k, s in str_losses.items()]) + f',{rays_per_sec:0.0f} r/s')

    return total_time, total_steps


def train_eval(test_batch, model, metric_harness, accelerator, cfg, summary_writer, logger, step):
    # We reuse the same random number generator from the optimization step
    # here on purpose so that the visualization matches what happened in training.
    eval_start_time = time.time()

    # render a single image with all distributed processes
    rendering = models.render_image(model, accelerator, test_batch, cfg)

    # move to numpy
    rendering = tree_map(lambda x: x.detach().cpu().numpy(), rendering)
    test_batch = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, test_batch)
    if not accelerator.is_main_process:
        return

    # Log eval summaries on host 0.
    eval_time = time.time() - eval_start_time
    num_rays = np.prod(test_batch['directions'].shape[:-1])
    rays_per_sec = num_rays / eval_time
    summary_writer.add_scalar('test_rays_per_sec', rays_per_sec, step)

    metric_start_time = time.time()
    metric = metric_harness(rendering['rgb'], test_batch['rgb'])
    logger.info(f'Eval {step}: {eval_time:0.3f}s, {rays_per_sec:0.0f} rays/sec')
    logger.info(f'Metrics computed in {(time.time() - metric_start_time):0.3f}s')
    for name, val in metric.items():
        if not np.isnan(val):
            logger.info(f'{name} = {val:.4f}')
            summary_writer.add_scalar('train_metrics/' + name, val, step)

    if cfg.vis_decimate > 1:
        d = cfg.vis_decimate
        decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
    else:
        decimate_fn = lambda x: x
    rendering = tree_map(decimate_fn, rendering)
    test_batch = tree_map(decimate_fn, test_batch)
    vis_start_time = time.time()
    vis_suite = vis.visualize_suite(rendering, test_batch)
    with tqdm.external_write_mode():
        logger.info(f'Visualized in {(time.time() - vis_start_time):0.3f}s')

    # function to convert image for tensorboard
    tb_process_fn = lambda x: x.transpose(2, 0, 1) if len(x.shape) == 3 else x[None]
    summary_writer.add_image('test_true_color', tb_process_fn(test_batch['rgb']), step)
    if 'alphas' in test_batch:
        summary_writer.add_image('test_true_alpha', tb_process_fn(test_batch['alphas'][..., None]), step)
    for k, v in vis_suite.items():
        summary_writer.add_image('test_output_' + k, tb_process_fn(v), step)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', nargs='+', help="Path to gin config files")
    p.add_argument('-p', '--param', nargs='+', help="Command line parameter override")
    args = p.parse_args()

    gin.parse_config_files_and_bindings(args.config, args.param, skip_unknown=True)
    with gin.config_scope('train'):
        train()