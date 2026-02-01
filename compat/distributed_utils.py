# Copyright (c) 2024, pUniFind Team
# Distributed utilities for multi-GPU inference
# Supports both Windows (gloo backend) and Linux (nccl backend)

import os
import sys
import pickle
import torch
import torch.distributed as dist
from typing import List, Any, Optional, Callable


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_initialized()


def get_world_size() -> int:
    """Get the total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank of the current process."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_data_parallel_world_size() -> int:
    """Get the world size for data parallelism."""
    return get_world_size()


def get_data_parallel_rank() -> int:
    """Get the rank for data parallelism."""
    return get_rank()


def get_data_parallel_group():
    """Get the process group for data parallelism."""
    if is_distributed():
        return dist.group.WORLD
    return None


def infer_init_method(args) -> None:
    """
    Infer the distributed initialization method from environment variables.
    Supports both Windows and Linux.
    """
    if args.distributed_init_method is not None:
        return

    # Check for common distributed environment variables
    if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
        args.distributed_init_method = "env://"
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.distributed_init_method = "env://"


def distributed_init(args) -> int:
    """
    Initialize distributed training.

    Args:
        args: Arguments containing distributed settings

    Returns:
        The rank of the current process
    """
    if is_distributed():
        return get_rank()

    if args.distributed_world_size == 1:
        return 0

    # Determine backend based on platform and CUDA availability
    if sys.platform == "win32":
        # Windows does not support NCCL, use gloo
        backend = "gloo"
    else:
        # Linux: use NCCL for GPU, gloo for CPU
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Initialize the process group
    if hasattr(args, 'distributed_init_method') and args.distributed_init_method:
        dist.init_process_group(
            backend=backend,
            init_method=args.distributed_init_method,
            world_size=args.distributed_world_size,
            rank=args.distributed_rank,
        )
    else:
        dist.init_process_group(backend=backend)

    args.distributed_rank = get_rank()
    args.device_id = args.distributed_rank % torch.cuda.device_count() if torch.cuda.is_available() else 0

    return args.distributed_rank


def all_gather_list(
    data: List[Any],
    group: Optional[Any] = None,
    max_size: int = 16384,
) -> List[Any]:
    """
    Gather arbitrary data from all processes.

    Args:
        data: List of data to gather
        group: Process group (optional)
        max_size: Maximum buffer size for gathering

    Returns:
        List of gathered data from all processes
    """
    if not is_distributed():
        return data

    world_size = get_world_size()
    rank = get_rank()

    # Serialize data
    buffer = pickle.dumps(data)
    size = len(buffer)

    # Create tensors for size synchronization
    size_tensor = torch.tensor([size], dtype=torch.long)
    if torch.cuda.is_available():
        size_tensor = size_tensor.cuda()

    # Gather sizes from all processes
    size_list = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    if torch.cuda.is_available():
        size_list = [s.cuda() for s in size_list]

    dist.all_gather(size_list, size_tensor, group=group)

    max_size = max(int(s.item()) for s in size_list)

    # Pad buffer to max size
    buffer_tensor = torch.zeros(max_size, dtype=torch.uint8)
    buffer_tensor[:size] = torch.tensor(list(buffer), dtype=torch.uint8)
    if torch.cuda.is_available():
        buffer_tensor = buffer_tensor.cuda()

    # Gather all buffers
    buffer_list = [torch.zeros(max_size, dtype=torch.uint8) for _ in range(world_size)]
    if torch.cuda.is_available():
        buffer_list = [b.cuda() for b in buffer_list]

    dist.all_gather(buffer_list, buffer_tensor, group=group)

    # Deserialize gathered data
    result = []
    for i, (buf, s) in enumerate(zip(buffer_list, size_list)):
        size_i = int(s.item())
        buf_bytes = bytes(buf[:size_i].cpu().numpy().tolist())
        result.extend(pickle.loads(buf_bytes))

    return result


def barrier(group: Optional[Any] = None) -> None:
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier(group=group)


def call_main(args, main_func: Callable, **kwargs) -> None:
    """
    Call the main function with distributed support.

    Args:
        args: Arguments to pass to main function
        main_func: The main function to call
        **kwargs: Additional keyword arguments
    """
    # Initialize distributed if needed
    if hasattr(args, 'distributed_world_size') and args.distributed_world_size > 1:
        if not is_distributed():
            distributed_init(args)

    # Set device
    if torch.cuda.is_available() and hasattr(args, 'device_id'):
        torch.cuda.set_device(args.device_id)

    # Call main function
    main_func(args, **kwargs)


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Reduce a tensor across all processes.

    Args:
        tensor: Tensor to reduce
        world_size: Number of processes

    Returns:
        Reduced tensor (averaged)
    """
    if not is_distributed():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt
