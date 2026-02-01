# Copyright (c) 2024, pUniFind Team
# Task framework for managing datasets, models, and losses

import argparse
from typing import Dict, Any, Optional, Type, Callable
from torch.utils.data import Dataset, DataLoader
import torch

# Global task registry
TASK_REGISTRY: Dict[str, Type["UnicoreTask"]] = {}
TASK_DATACLASS_REGISTRY: Dict[str, Any] = {}


def register_task(name: str) -> Callable:
    """
    Decorator to register a task.

    Args:
        name: Name of the task

    Returns:
        Decorator function
    """
    def decorator(cls):
        TASK_REGISTRY[name] = cls
        return cls
    return decorator


def setup_task(args: argparse.Namespace, **kwargs) -> "UnicoreTask":
    """
    Setup a task from arguments.

    Args:
        args: Parsed arguments
        **kwargs: Additional keyword arguments

    Returns:
        Initialized task instance
    """
    # Ensure task registries are initialized
    if len(TASK_REGISTRY) == 0:
        try:
            # Import the tasks package which auto-imports all task modules
            import PepMS.tasks
        except ImportError as e:
            print(f"Warning: Failed to import PepMS.tasks: {e}")

    task_name = getattr(args, 'task', None)
    if task_name is None:
        raise ValueError("No task specified")

    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_REGISTRY.keys())}")

    task_cls = TASK_REGISTRY[task_name]
    return task_cls.setup_task(args, **kwargs)


class UnicoreTask:
    """
    Base class for tasks.

    Tasks define how to load datasets, build models, and compute losses.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.datasets: Dict[str, Dataset] = {}

    @classmethod
    def setup_task(cls, args: argparse.Namespace, **kwargs) -> "UnicoreTask":
        """
        Setup the task from arguments.

        Args:
            args: Parsed arguments
            **kwargs: Additional keyword arguments

        Returns:
            Initialized task instance
        """
        return cls(args)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """
        Add task-specific arguments to the parser.

        Args:
            parser: ArgumentParser instance
        """
        pass

    def load_dataset(
        self,
        split: str,
        combine: bool = False,
        epoch: int = 1,
        **kwargs,
    ) -> None:
        """
        Load a dataset split.

        Args:
            split: Name of the split (e.g., 'train', 'valid')
            combine: Whether to combine with other splits
            epoch: Current epoch number
            **kwargs: Additional keyword arguments
        """
        raise NotImplementedError

    def dataset(self, split: str) -> Dataset:
        """
        Get a dataset by split name.

        Args:
            split: Name of the split

        Returns:
            Dataset instance
        """
        if split not in self.datasets:
            raise KeyError(f"Dataset not found: {split}")
        return self.datasets[split]

    def build_model(self, args: argparse.Namespace) -> torch.nn.Module:
        """
        Build the model.

        Args:
            args: Parsed arguments

        Returns:
            Model instance
        """
        from . import models
        return models.build_model(args, self)

    def build_loss(self, args: argparse.Namespace) -> torch.nn.Module:
        """
        Build the loss function.

        Args:
            args: Parsed arguments

        Returns:
            Loss module instance
        """
        # Default: return a dummy loss that does nothing
        return DummyLoss()

    def get_batch_iterator(
        self,
        dataset: Dataset,
        batch_size: int,
        ignore_invalid_inputs: bool = True,
        required_batch_size_multiple: int = 1,
        seed: int = 1,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 0,
        data_buffer_size: int = 10,
        **kwargs,
    ) -> "EpochBatchIterator":
        """
        Get a batch iterator for the dataset.

        Args:
            dataset: Dataset to iterate over
            batch_size: Batch size
            ignore_invalid_inputs: Whether to ignore invalid inputs
            required_batch_size_multiple: Batch size multiple requirement
            seed: Random seed
            num_shards: Number of shards for distributed training
            shard_id: Shard ID for current process
            num_workers: Number of data loading workers
            data_buffer_size: Data buffer size

        Returns:
            EpochBatchIterator instance
        """
        # Adjust batch size to be a multiple of required_batch_size_multiple
        if required_batch_size_multiple > 1:
            batch_size = max(
                1,
                batch_size // required_batch_size_multiple * required_batch_size_multiple
            )

        # Use the dataset's collater method if available
        collate_fn = getattr(dataset, 'collater', None)

        return EpochBatchIterator(
            dataset=dataset,
            batch_size=batch_size,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            seed=seed,
            collate_fn=collate_fn,
        )


class DummyLoss(torch.nn.Module):
    """Dummy loss module for inference."""

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.tensor(0.0)


class EpochBatchIterator:
    """
    Iterator for batches within an epoch.
    Supports distributed training by sharding the dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 0,
        seed: int = 1,
        collate_fn: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers
        self.seed = seed
        self.collate_fn = collate_fn
        self._current_epoch = 1

    def __len__(self) -> int:
        """Get the number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def next_epoch_itr(
        self,
        shuffle: bool = True,
        fix_batches_to_gpus: bool = False,
    ) -> DataLoader:
        """
        Get an iterator for the next epoch.

        Args:
            shuffle: Whether to shuffle the data
            fix_batches_to_gpus: Whether to fix batches to GPUs

        Returns:
            DataLoader instance
        """
        # Create a sampler for distributed training
        if self.num_shards > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=self.num_shards,
                rank=self.shard_id,
                shuffle=shuffle,
            )
        else:
            if shuffle:
                generator = torch.Generator()
                generator.manual_seed(self.seed + self._current_epoch)
                sampler = torch.utils.data.RandomSampler(
                    self.dataset,
                    generator=generator,
                )
            else:
                sampler = torch.utils.data.SequentialSampler(self.dataset)

        self._current_epoch += 1

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
