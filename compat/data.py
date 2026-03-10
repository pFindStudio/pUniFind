# Copyright (c) 2024, pUniFind Team
# Data utilities for dataset management

import random
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, List, Union


class UnicoreDataset(Dataset):
    """
    Base class for all datasets.
    """

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def collater(self, samples: List[Any]) -> Any:
        """
        Merge a list of samples into a batch.

        Args:
            samples: List of samples

        Returns:
            Batched samples
        """
        if len(samples) == 0:
            return {}
        return torch.utils.data.default_collate(samples)

    @property
    def sizes(self) -> List[int]:
        """Get the sizes of all samples."""
        return [1] * len(self)

    def num_tokens(self, index: int) -> int:
        """Get the number of tokens for sample at index."""
        return 1

    def size(self, index: int) -> int:
        """Get the size of sample at index."""
        return 1

    def ordered_indices(self) -> List[int]:
        """Get ordered indices for iterating."""
        return list(range(len(self)))

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch."""
        pass


class BaseWrapperDataset(UnicoreDataset):
    """
    Base class for dataset wrappers.
    """

    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def sizes(self) -> List[int]:
        if hasattr(self.dataset, 'sizes'):
            return self.dataset.sizes
        return [1] * len(self)

    def collater(self, samples: List[Any]) -> Any:
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        return super().collater(samples)


class EpochShuffleDataset(BaseWrapperDataset):
    """
    Dataset wrapper that shuffles data each epoch.
    """

    def __init__(
        self,
        dataset: Dataset,
        size: Optional[int] = None,
        seed: int = 1,
    ):
        super().__init__(dataset)
        self.seed = seed
        self._size = size if size is not None else len(dataset)
        self._current_epoch = 0
        self._shuffled_indices = list(range(len(dataset)))

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self._shuffled_indices[index]]

    def __len__(self) -> int:
        return self._size

    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch and reshuffle the data.

        Args:
            epoch: Current epoch number
        """
        self._current_epoch = epoch
        # Use a fixed seed combined with epoch for reproducibility
        rng = random.Random(self.seed + epoch)
        self._shuffled_indices = list(range(len(self.dataset)))
        rng.shuffle(self._shuffled_indices)

    def ordered_indices(self) -> List[int]:
        return list(range(len(self)))


class NestedDictionaryDataset(UnicoreDataset):
    """
    Dataset that returns nested dictionaries.
    Useful for organizing inputs and targets.
    """

    def __init__(
        self,
        defn: Dict[str, Any],
        sizes: Optional[List[int]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            defn: Nested dictionary definition where leaf values are datasets
            sizes: Optional list of sizes
        """
        super().__init__()
        self.defn = defn
        self._sizes = sizes

        # Find the first dataset to determine length
        self._len = None
        self._find_length(defn)

    def _find_length(self, obj: Any) -> None:
        """Find the length from nested datasets."""
        if self._len is not None:
            return

        if isinstance(obj, dict):
            for v in obj.values():
                self._find_length(v)
        elif hasattr(obj, '__len__'):
            self._len = len(obj)

    def _get_nested(self, index: int, obj: Any) -> Any:
        """Get item at index from nested structure."""
        if isinstance(obj, dict):
            return {k: self._get_nested(index, v) for k, v in obj.items()}
        elif hasattr(obj, '__getitem__'):
            return obj[index]
        else:
            return obj

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._get_nested(index, self.defn)

    def __len__(self) -> int:
        return self._len if self._len is not None else 0

    @property
    def sizes(self) -> List[int]:
        if self._sizes is not None:
            return self._sizes
        return [1] * len(self)

    def collater(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate samples into a batch.

        Args:
            samples: List of sample dictionaries

        Returns:
            Batched dictionary
        """
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}

        return self._collate_nested(samples, self.defn)

    def _collate_nested(
        self,
        samples: List[Any],
        defn: Any,
    ) -> Any:
        """Recursively collate nested structures."""
        if isinstance(defn, dict):
            result = {}
            for k, v in defn.items():
                items = [s[k] if isinstance(s, dict) and k in s else None for s in samples]
                result[k] = self._collate_nested(items, v)
            return result
        elif hasattr(defn, 'collater'):
            return defn.collater(samples)
        else:
            # Try default collation
            try:
                # Handle lists of tensors
                if all(isinstance(s, torch.Tensor) for s in samples):
                    return torch.stack(samples)
                # Handle lists of scalars
                elif all(isinstance(s, (int, float)) for s in samples):
                    return torch.tensor(samples)
                # Handle lists of strings
                elif all(isinstance(s, str) for s in samples):
                    return samples
                # Handle lists of lists
                elif all(isinstance(s, list) for s in samples):
                    return samples
                else:
                    return samples
            except Exception:
                return samples
