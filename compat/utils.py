# Copyright (c) 2024, pUniFind Team
# General utility functions

import torch
from typing import Dict, Any, Optional, Union, List


def move_to_cuda(
    sample: Union[Dict[str, Any], torch.Tensor, List],
    device: Optional[torch.device] = None,
) -> Union[Dict[str, Any], torch.Tensor, List]:
    """
    Move a sample to CUDA device.

    Args:
        sample: Sample to move (can be dict, tensor, or list)
        device: Target device (defaults to current CUDA device)

    Returns:
        Sample on CUDA device
    """
    if device is None:
        device = torch.cuda.current_device()

    if isinstance(sample, dict):
        return {
            key: move_to_cuda(value, device)
            for key, value in sample.items()
        }
    elif isinstance(sample, list):
        return [move_to_cuda(item, device) for item in sample]
    elif isinstance(sample, tuple):
        return tuple(move_to_cuda(item, device) for item in sample)
    elif isinstance(sample, torch.Tensor):
        return sample.to(device=device, non_blocking=True)
    else:
        return sample


def move_to_cpu(
    sample: Union[Dict[str, Any], torch.Tensor, List],
) -> Union[Dict[str, Any], torch.Tensor, List]:
    """
    Move a sample to CPU.

    Args:
        sample: Sample to move

    Returns:
        Sample on CPU
    """
    if isinstance(sample, dict):
        return {key: move_to_cpu(value) for key, value in sample.items()}
    elif isinstance(sample, list):
        return [move_to_cpu(item) for item in sample]
    elif isinstance(sample, tuple):
        return tuple(move_to_cpu(item) for item in sample)
    elif isinstance(sample, torch.Tensor):
        return sample.cpu()
    else:
        return sample


def apply_to_sample(
    f,
    sample: Union[Dict[str, Any], torch.Tensor, List],
):
    """
    Apply a function to all tensors in a sample.

    Args:
        f: Function to apply
        sample: Sample containing tensors

    Returns:
        Sample with function applied to all tensors
    """
    if isinstance(sample, dict):
        return {key: apply_to_sample(f, value) for key, value in sample.items()}
    elif isinstance(sample, list):
        return [apply_to_sample(f, item) for item in sample]
    elif isinstance(sample, tuple):
        return tuple(apply_to_sample(f, item) for item in sample)
    elif isinstance(sample, torch.Tensor):
        return f(sample)
    else:
        return sample


def item(tensor: Union[torch.Tensor, float, int]) -> Union[float, int]:
    """
    Get the Python scalar value from a tensor.

    Args:
        tensor: Tensor or scalar

    Returns:
        Python scalar
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.item()
    return tensor


def clip_grad_norm_(
    params,
    max_norm: float,
    aggregate_norm_fn=None,
) -> torch.Tensor:
    """
    Clip gradient norm.

    Args:
        params: Model parameters
        max_norm: Maximum gradient norm
        aggregate_norm_fn: Optional function to aggregate norms

    Returns:
        Total gradient norm
    """
    if isinstance(params, torch.Tensor):
        params = [params]

    params = list(filter(lambda p: p.grad is not None, params))

    if len(params) == 0:
        return torch.tensor(0.0)

    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return total_norm


def fill_with_neg_inf(t: torch.Tensor) -> torch.Tensor:
    """
    Fill a tensor with -inf.

    Args:
        t: Input tensor

    Returns:
        Tensor filled with -inf
    """
    return t.float().fill_(float('-inf')).type_as(t)


def get_activation_fn(name: str):
    """
    Get an activation function by name.

    Args:
        name: Name of the activation function

    Returns:
        Activation function
    """
    if name == 'relu':
        return torch.nn.functional.relu
    elif name == 'gelu':
        return torch.nn.functional.gelu
    elif name == 'tanh':
        return torch.tanh
    elif name == 'sigmoid':
        return torch.sigmoid
    else:
        raise ValueError(f"Unknown activation function: {name}")
