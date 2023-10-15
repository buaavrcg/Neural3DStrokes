import torch
import numpy as np
from source import configs


class GradientScaler(torch.autograd.Function):
    """Near plane gradient scaling in https://gradient-scaling.github.io/"""
    @staticmethod
    def forward(ctx, colors, sigmas, ray_dist):
        ctx.save_for_backward(ray_dist)
        return colors, sigmas

    @staticmethod
    def backward(ctx, grad_output_colors, grad_output_sigmas):
        (ray_dist, ) = ctx.saved_tensors
        scaling = torch.square(ray_dist).clamp(0, 1)
        return grad_output_colors * scaling[..., None], grad_output_sigmas * scaling, None


class InfiniteSampler(torch.utils.data.Sampler):
    """
    Sampler for torch.utils.data.DataLoader that loops over the dataset
    indefinitely, shuffling items as it goes.
    """
    def __init__(self,
                 data_source: torch.utils.data.Dataset,
                 rank=0,
                 num_replicas=1,
                 shuffle=True,
                 seed=0,
                 window_size=0.5):
        super().__init__(data_source)
        assert len(data_source) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        self.dataset = data_source
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        try:
            while True:
                i = idx % order.size
                if idx % self.num_replicas == self.rank:
                    yield order[i]
                if window >= 2:
                    j = (i - rnd.randint(window)) % order.size
                    order[i], order[j] = order[j], order[i]
                idx += 1
        except GeneratorExit:
            pass


def clip_gradients(model, accelerator, config : configs.Config):
    """Clips gradients of MLP based on norm and max value."""
    if config.grad_max_norm > 0 and accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), config.grad_max_norm)

    if config.grad_max_val > 0 and accelerator.sync_gradients:
        accelerator.clip_grad_value_(model.parameters(), config.grad_max_val)

    for param in model.parameters():
        if param.grad is not None:
            param.grad.nan_to_num_()


def log_lerp(t, v0, v1):
    """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
    if v0 <= 0 or v1 <= 0:
        raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
    lv0 = np.log(v0)
    lv1 = np.log(v1)
    return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)


def learning_rate_decay(step, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1):
    """Continuous learning rate decay function.

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    Args:
        step: int, the current optimization step.
        lr_init: float, the initial learning rate.
        lr_final: float, the final learning rate.
        max_steps: int, the number of steps during optimization.
        lr_delay_steps: int, the number of steps to delay the full learning rate.
        lr_delay_mult: float, the multiplier on the rate when delaying it.

    Returns:
        lr: the learning for current step 'step'.
    """
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.
    return delay_rate * log_lerp(step / max_steps, lr_init, lr_final)


def create_optimizer(config: configs.Config, params):
    """Creates optax optimizer for model training."""
    adam_kwargs = {
        'betas': [config.adam_beta1, config.adam_beta2],
        'eps': config.adam_eps,
    }
    lr_kwargs = {
        'max_steps': config.max_steps,
        'lr_delay_steps': config.lr_delay_steps,
        'lr_delay_mult': config.lr_delay_mult,
        'lr_init': config.lr_init,
        'lr_final': config.lr_final,
    }

    lr_fn_main = lambda step: learning_rate_decay(step, **lr_kwargs)
    optimizer = torch.optim.Adam(params, lr=config.lr_init, **adam_kwargs)

    return optimizer, lr_fn_main
