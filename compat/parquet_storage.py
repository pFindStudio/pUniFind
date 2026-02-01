"""
Parquet-based storage module to replace LMDB.

This module provides cross-platform compatible storage using Parquet format,
which works well on both Windows and Linux, unlike LMDB which has poor Windows support.

The data is stored as gzip-compressed pickled objects in a binary column,
maintaining compatibility with the existing data format.
"""

import gzip
import os
import pickle
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetWriter:
    """
    Writer class for creating Parquet files from spectrum data.

    Replaces LMDB write operations with Parquet-based storage.
    Data is stored as gzip-compressed pickled binary in a Parquet column.
    """

    def __init__(self, path: str, batch_size: int = 1000):
        """
        Initialize ParquetWriter.

        Args:
            path: Path to the Parquet file (should end with .parquet)
            batch_size: Number of records to accumulate before writing to disk
        """
        self.path = path
        if not path.endswith('.parquet'):
            self.path = path.replace('.lmdb', '.parquet')
        self.batch_size = batch_size
        self._buffer: List[Dict[str, Any]] = []
        self._keys: List[bytes] = []
        self._written = False
        self._total_count = 0

        # Remove existing file if present
        if os.path.exists(self.path):
            os.remove(self.path)

    def put(self, key: bytes, value: bytes) -> None:
        """
        Add a key-value pair to the storage.

        Args:
            key: Key as bytes (typically index encoded as ASCII)
            value: Value as bytes (typically gzip-compressed pickled data)
        """
        self._buffer.append({
            'key': key,
            'data': value
        })
        self._keys.append(key)

        if len(self._buffer) >= self.batch_size:
            self._flush()

    def commit(self) -> None:
        """Commit any buffered data to disk."""
        if self._buffer:
            self._flush()

    def _flush(self) -> None:
        """Flush the buffer to disk."""
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)
        table = pa.Table.from_pandas(df)

        if not self._written:
            pq.write_table(table, self.path, compression='snappy')
            self._written = True
        else:
            # Append to existing file by reading, concatenating, and rewriting
            existing_table = pq.read_table(self.path)
            combined_table = pa.concat_tables([existing_table, table])
            pq.write_table(combined_table, self.path, compression='snappy')

        self._total_count += len(self._buffer)
        self._buffer = []

    def close(self) -> None:
        """Close the writer and ensure all data is written."""
        self.commit()

    def get_keys(self) -> List[bytes]:
        """Return all keys that have been written."""
        return self._keys

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ParquetReader:
    """
    Reader class for reading Parquet files created by ParquetWriter.

    Replaces LMDB read operations with Parquet-based storage.
    Provides similar interface to LMDB environment for compatibility.
    """

    def __init__(self, path: str, cache_size: int = 1000):
        """
        Initialize ParquetReader.

        Args:
            path: Path to the Parquet file
            cache_size: Size of the LRU cache for data access
        """
        self.path = path
        if not path.endswith('.parquet'):
            self.path = path.replace('.lmdb', '.parquet')

        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Parquet file not found: {self.path}")

        # Read the entire table into memory for fast access
        self._table = pq.read_table(self.path)
        self._df = self._table.to_pandas()

        # Build key-to-index mapping for O(1) lookup
        self._key_to_idx = {
            key: idx for idx, key in enumerate(self._df['key'].values)
        }
        self._keys = list(self._df['key'].values)
        self._cache_size = cache_size

    def keys(self) -> List[bytes]:
        """Return all keys in the storage."""
        return self._keys

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self._keys)

    def get(self, key: bytes) -> Optional[bytes]:
        """
        Get value by key.

        Args:
            key: Key as bytes

        Returns:
            Value as bytes if found, None otherwise
        """
        idx = self._key_to_idx.get(key)
        if idx is None:
            return None
        return self._df.iloc[idx]['data']

    def get_by_index(self, idx: int) -> bytes:
        """
        Get value by integer index.

        Args:
            idx: Integer index

        Returns:
            Value as bytes
        """
        return self._df.iloc[idx]['data']

    def __getitem__(self, key_or_idx: Union[bytes, int]) -> bytes:
        """
        Get value by key or index.

        Args:
            key_or_idx: Either a bytes key or integer index

        Returns:
            Value as bytes
        """
        if isinstance(key_or_idx, int):
            return self.get_by_index(key_or_idx)
        return self.get(key_or_idx)

    def close(self) -> None:
        """Close the reader and free resources."""
        self._df = None
        self._table = None
        self._key_to_idx = None
        self._keys = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ParquetDataset:
    """
    Dataset class providing LMDB-like interface for Parquet storage.

    This class is designed to be a drop-in replacement for LMDB-based datasets.
    It provides the same interface (connect_db, __len__, __getitem__) but uses
    Parquet files instead of LMDB.
    """

    def __init__(self, db_path: str, decompress: bool = True):
        """
        Initialize ParquetDataset.

        Args:
            db_path: Path to the Parquet file (or .lmdb path which will be converted)
            decompress: Whether to decompress and unpickle data on access
        """
        self.db_path = db_path
        if not db_path.endswith('.parquet'):
            self.db_path = db_path.replace('.lmdb', '.parquet')

        if not os.path.isfile(self.db_path):
            raise FileNotFoundError(f"Parquet file not found: {self.db_path}")

        self.decompress = decompress
        self._reader = ParquetReader(self.db_path)
        self._keys = self._reader.keys()

    def connect_db(self, path: Optional[str] = None) -> 'ParquetDataset':
        """
        Connect to database (for LMDB interface compatibility).

        Args:
            path: Optional path to database (uses self.db_path if not provided)

        Returns:
            Self for chaining
        """
        return self

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: int) -> Any:
        """
        Get item by index.

        Args:
            idx: Integer index

        Returns:
            Decompressed and unpickled data if decompress=True,
            otherwise raw bytes
        """
        data = self._reader.get(self._keys[idx])
        if self.decompress and data is not None:
            try:
                data = pickle.loads(gzip.decompress(data))
            except Exception:
                # Try without decompression (some data may not be compressed)
                try:
                    data = pickle.loads(data)
                except Exception:
                    pass
        return data

    def get_by_key(self, key: bytes) -> Any:
        """
        Get item by key.

        Args:
            key: Key as bytes

        Returns:
            Decompressed and unpickled data if decompress=True,
            otherwise raw bytes
        """
        data = self._reader.get(key)
        if self.decompress and data is not None:
            try:
                data = pickle.loads(gzip.decompress(data))
            except Exception:
                try:
                    data = pickle.loads(data)
                except Exception:
                    pass
        return data

    def keys(self) -> List[bytes]:
        """Return all keys."""
        return self._keys

    def close(self) -> None:
        """Close the dataset."""
        self._reader.close()


class MultiParquetDataset:
    """
    Dataset class for handling multiple Parquet files (splits).

    Replaces MultiZipLMDB2Dataset and similar classes that handle
    multiple LMDB files with split data.
    """

    def __init__(
        self,
        db_path: str,
        splits: List[str],
        key_dict: Optional[Dict] = None
    ):
        """
        Initialize MultiParquetDataset.

        Args:
            db_path: Base path containing split files
            splits: List of split names (e.g., ['new1', 'new2', ...])
            key_dict: Optional dictionary mapping keys to (split, key) pairs
        """
        self.db_path = db_path
        self.splits = splits
        self.key_dict = key_dict

        # Load all split readers
        self._readers: Dict[str, ParquetReader] = {}
        for split in splits:
            split_path = os.path.join(db_path, f"{split}.parquet")
            if os.path.isfile(split_path):
                self._readers[split] = ParquetReader(split_path)

        # Build unified key list
        if key_dict is not None:
            self._keys = list(key_dict.keys())
        else:
            self._keys = []
            for split, reader in self._readers.items():
                for key in reader.keys():
                    self._keys.append((split, key))

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: int) -> Any:
        """
        Get item by index.

        Args:
            idx: Integer index

        Returns:
            Decompressed and unpickled data
        """
        if self.key_dict is not None:
            # Use key_dict to find the split and key
            key_info = self._keys[idx]
            if isinstance(key_info, tuple):
                split, key = key_info
            else:
                # For weighted sampling, need to get from key_dict
                import random
                pep_key = key_info
                split_keys = self.key_dict[pep_key]
                split, key = random.choice(list(split_keys.items()))
                if isinstance(key, list):
                    key = random.choice(key)
        else:
            split, key = self._keys[idx]

        data = self._readers[split].get(key)
        if data is not None:
            try:
                data = pickle.loads(gzip.decompress(data))
            except Exception:
                try:
                    data = pickle.loads(data)
                except Exception:
                    pass
        return data

    def close(self) -> None:
        """Close all readers."""
        for reader in self._readers.values():
            reader.close()


def convert_lmdb_to_parquet(lmdb_path: str, parquet_path: Optional[str] = None) -> str:
    """
    Convert an existing LMDB file to Parquet format.

    Args:
        lmdb_path: Path to the LMDB file
        parquet_path: Optional output path (defaults to replacing .lmdb with .parquet)

    Returns:
        Path to the created Parquet file
    """
    import lmdb as lmdb_lib

    if parquet_path is None:
        parquet_path = lmdb_path.replace('.lmdb', '.parquet')

    # Open LMDB
    env = lmdb_lib.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=1,
    )

    # Create Parquet writer
    with ParquetWriter(parquet_path) as writer:
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                writer.put(key, value)

    env.close()
    return parquet_path


def get_storage_path(original_path: str) -> str:
    """
    Convert a path from .lmdb to .parquet extension.

    Args:
        original_path: Original path (may have .lmdb extension)

    Returns:
        Path with .parquet extension
    """
    if original_path.endswith('.lmdb'):
        return original_path.replace('.lmdb', '.parquet')
    elif not original_path.endswith('.parquet'):
        return original_path + '.parquet'
    return original_path
