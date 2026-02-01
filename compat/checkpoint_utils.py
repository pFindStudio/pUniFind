# Copyright (c) 2024, pUniFind Team
# Checkpoint utilities for loading model weights

import torch
import os
from typing import Dict, Any, Optional


def load_checkpoint_to_cpu(
    path: str,
    arg_overrides: Optional[Dict[str, Any]] = None,
    load_on_all_ranks: bool = False,
) -> Dict[str, Any]:
    """
    Load a checkpoint to CPU (compatible with both Windows and Linux).

    Args:
        path: Path to the checkpoint file
        arg_overrides: Optional dictionary of argument overrides
        load_on_all_ranks: Whether to load on all ranks (for distributed)

    Returns:
        Dictionary containing the checkpoint state
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint to CPU to avoid GPU memory issues
    state = torch.load(path, map_location=torch.device("cpu"))

    # Handle arg_overrides if provided
    if arg_overrides is not None and "args" in state:
        args = state["args"]
        for key, value in arg_overrides.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return state


def save_checkpoint(
    state: Dict[str, Any],
    path: str,
) -> None:
    """
    Save a checkpoint to disk.

    Args:
        state: Dictionary containing the checkpoint state
        path: Path to save the checkpoint
    """
    torch.save(state, path)
