import logging
import os
import torch
import numpy as np
import collections
from PIL import Image
from scipy.stats import norm


class Timing:
    """
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")


def handle_exception(exc_type, exc_value, exc_traceback):
    logging.error("Error!", exc_info=(exc_type, exc_value, exc_traceback))


def nan_sum(x):
    return (torch.isnan(x) | torch.isinf(x)).sum()


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def open_file(pth, mode='r'):
    return open(pth, mode=mode)


def file_exists(pth):
    return os.path.exists(pth)


def listdir(pth):
    return os.listdir(pth)


def isdir(pth):
    return os.path.isdir(pth)


def makedirs(pth):
    os.makedirs(pth, exist_ok=True)


def load_img(pth):
    """Load an image and cast to float32."""
    image = np.array(Image.open(pth), dtype=np.float32)
    return image


def save_img_u8(img, pth):
    """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
        pth, 'PNG')


def save_img_f32(depthmap, pth, p=0.5):
    """Save an image (probably a depthmap) to disk as a float32 TIFF."""
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(pth, 'TIFF')


def safe_normalize(x : np.ndarray, axis=-1, eps=1e-8):
    """Normalize a tensor by its L2 norm."""
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def sample_normal_dist(n: int, low: float, high: float, target_ratio=0.95):
    """Sample from a normal distribution."""
    # Use the percent point function (inverse of cdf) to find the z-scores for the bounds
    lower_z = norm.ppf((1 - target_ratio) / 2)
    upper_z = norm.ppf(1 - (1 - target_ratio) / 2)
    
    # Calculate the mean and std deviation of the distribution that 
    # would place low and high at the corresponding z-scores
    mean = (low * upper_z - high * lower_z) / (upper_z - lower_z)
    std = (high - low) / (upper_z - lower_z)

    return np.random.normal(mean, std, n)
