from functools import wraps
import torch

__all__ = [
    'mappify',
    'batch_to_device',
]


def mappify(func):
    @wraps(func)
    def inner(X, **kwargs):
        return [func(x, **kwargs) for x in X]
    return inner


def batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        res = [batch_to_device(x, device) for x in batch]
        return res if isinstance(batch, list) else tuple(res)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
