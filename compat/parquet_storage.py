"""
Parquet-based storage module to replace LMDB.

This module provides cross-platform compatible storage using Parquet format,
which works well on both Windows and Linux, unlike LMDB which has poor Windows support.

The data is stored as pickled objects in a binary column.
Compression is handled by Parquet's native snappy/zstd compression, which is faster than gzip.
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

# Use faster pickle protocol
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


class ParquetWriter:
    """
    Writer class for creating Parquet files from spectrum data.

    Replaces LMDB write operations with Parquet-based storage.
    Data is stored as pickled binary in a Parquet column.
    Uses PyArrow's streaming ParquetWriter to write row groups incrementally,
    avoiding the O(n^2) read-concat-rewrite pattern.
    """

    def __init__(self, path: str, batch_size: int = 5000, use_gzip: bool = False):
        """
        Initialize ParquetWriter.

        Args:
            path: Path to the Parquet file (should end with .parquet)
            batch_size: Number of records to accumulate before flushing as a row group
            use_gzip: Whether to use gzip compression (default: False for better performance)
        """
        self.path = path
        if not path.endswith('.parquet'):
            self.path = path.replace('.lmdb', '.parquet')
        self.batch_size = batch_size
        self.use_gzip = use_gzip
        self._buffer_keys: List[bytes] = []
        self._buffer_data: List[bytes] = []
        self._keys: List[bytes] = []
        self._total_count = 0
        self._writer: Optional[pq.ParquetWriter] = None

        # Remove existing file if present
        if os.path.exists(self.path):
            os.remove(self.path)

        # Define schema once
        self._schema = pa.schema([
            ('key', pa.binary()),
            ('data', pa.binary()),
        ])

    def put(self, key: bytes, value: bytes) -> None:
        """
        Add a key-value pair to the storage.

        Args:
            key: Key as bytes (typically index encoded as ASCII)
            value: Value as bytes (pickled data, optionally gzip-compressed)
        """
        self._buffer_keys.append(key)
        self._buffer_data.append(value)
        self._keys.append(key)

        if len(self._buffer_keys) >= self.batch_size:
            self._flush()

    def put_object(self, key: bytes, obj: Any) -> None:
        """
        Add an object to the storage (will be pickled automatically).

        This method avoids gzip compression for better performance.

        Args:
            key: Key as bytes
            obj: Python object to store
        """
        # Use highest protocol for fastest serialization
        value = pickle.dumps(obj, protocol=PICKLE_PROTOCOL)
        self.put(key, value)

    def commit(self) -> None:
        """Commit any buffered data to disk."""
        if self._buffer_keys:
            self._flush()

    def _flush(self) -> None:
        """Flush the buffer to disk as a new row group (O(1) per flush)."""
        if not self._buffer_keys:
            return

        table = pa.table(
            {'key': self._buffer_keys, 'data': self._buffer_data},
            schema=self._schema,
        )

        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.path, self._schema,
                compression='zstd', compression_level=3,
            )

        self._writer.write_table(table)

        self._total_count += len(self._buffer_keys)
        self._buffer_keys = []
        self._buffer_data = []

    def close(self) -> None:
        """Close the writer and ensure all data is written."""
        self.commit()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

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

        # Extract data column as numpy array for O(1) access without pandas iloc overhead
        self._data_array = self._df['data'].values

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
        return self._data_array[idx]

    def get_by_index(self, idx: int) -> bytes:
        """
        Get value by integer index.

        Args:
            idx: Integer index

        Returns:
            Value as bytes
        """
        return self._data_array[idx]

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
        self._data_array = None
        self._key_to_idx = None
        self._keys = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _fast_unpickle(data: bytes) -> Any:
    """
    Fast unpickle with fallback for gzip-compressed data.

    Tries pickle.loads first (fastest), falls back to gzip.decompress if needed.
    This is optimized for the common case where data is not gzip-compressed.
    """
    if data is None:
        return None

    # Fast path: try direct unpickle first (most common case for new data)
    try:
        return pickle.loads(data)
    except Exception:
        pass

    # Fallback: try gzip decompression (for legacy data)
    try:
        return pickle.loads(gzip.decompress(data))
    except Exception:
        return None


class ParquetDataset:
    """
    Dataset class providing LMDB-like interface for Parquet storage.

    This class is designed to be a drop-in replacement for LMDB-based datasets.
    It provides the same interface (connect_db, __len__, __getitem__) but uses
    Parquet files instead of LMDB.

    Performance optimizations:
    - Larger LRU cache (512 instead of 16) for better cache hit rate
    - Fast unpickle path that tries direct pickle first, gzip fallback second
    - Pre-loaded data in memory via Parquet reader
    """

    # Class-level cache size configuration
    CACHE_SIZE = 512  # Increased from 16 for better hit rate

    def __init__(self, db_path: str, decompress: bool = True, cache_size: int = None):
        """
        Initialize ParquetDataset.

        Args:
            db_path: Path to the Parquet file (or .lmdb path which will be converted)
            decompress: Whether to decompress and unpickle data on access
            cache_size: LRU cache size (default: 512)
        """
        self.db_path = db_path
        if not db_path.endswith('.parquet'):
            self.db_path = db_path.replace('.lmdb', '.parquet')

        if not os.path.isfile(self.db_path):
            raise FileNotFoundError(f"Parquet file not found: {self.db_path}")

        self.decompress = decompress
        self._reader = ParquetReader(self.db_path)
        self._keys = self._reader.keys()

        # Configure cache size
        self._cache_size = cache_size or self.CACHE_SIZE
        # Clear any existing cache and set new size
        self._getitem_cached = lru_cache(maxsize=self._cache_size)(self._getitem_impl)

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

    def _getitem_impl(self, idx: int) -> Any:
        """
        Internal implementation of __getitem__ (cached via lru_cache).
        """
        data = self._reader.get(self._keys[idx])
        if self.decompress and data is not None:
            return _fast_unpickle(data)
        return data

    def __getitem__(self, idx: int) -> Any:
        """
        Get item by index.

        Args:
            idx: Integer index

        Returns:
            Decompressed and unpickled data if decompress=True,
            otherwise raw bytes
        """
        return self._getitem_cached(idx)

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

    @lru_cache(maxsize=512)  # Increased from 16 for better cache hit rate
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
            # Use fast unpickle with gzip fallback
            return _fast_unpickle(data)
        return data

    def close(self) -> None:
        """Close all readers."""
        for reader in self._readers.values():
            reader.close()


def check_parquet_up_to_date(
    parquet_path: str,
    keys_path: str,
    source_files: List[str],
) -> bool:
    """
    Check if an existing parquet file is up-to-date and can be reused.

    Validates:
    1. Parquet file and keys pkl file both exist
    2. Parquet row count matches keys count
    3. All source files are older than the parquet file (no new data)

    Args:
        parquet_path: Path to the parquet file
        keys_path: Path to the keys pkl file
        source_files: List of source file paths (mgf, qry.res, etc.)

    Returns:
        True if parquet is up-to-date and can be reused
    """
    import logging
    logger = logging.getLogger("pUniFind.parquet_check")

    # 1. Check files exist
    if not os.path.isfile(parquet_path):
        logger.info(f"Parquet file not found: {parquet_path}")
        return False
    if not os.path.isfile(keys_path):
        logger.info(f"Keys file not found: {keys_path}")
        return False

    try:
        # 2. Read parquet metadata to get row count (without loading data)
        parquet_meta = pq.read_metadata(parquet_path)
        parquet_num_rows = parquet_meta.num_rows

        # Load keys and check count
        with open(keys_path, "rb") as f:
            keys = pickle.load(f)

        keys_count = len(keys)

        if parquet_num_rows != keys_count:
            logger.info(
                f"Row count mismatch: parquet has {parquet_num_rows} rows, "
                f"keys file has {keys_count} keys. Re-preprocessing."
            )
            return False

        if parquet_num_rows == 0:
            logger.info("Parquet file is empty. Re-preprocessing.")
            return False

        # 3. Check source files haven't been modified after parquet was created
        parquet_mtime = os.path.getmtime(parquet_path)
        source_count = 0
        for src in source_files:
            if os.path.isfile(src):
                source_count += 1
                if os.path.getmtime(src) > parquet_mtime:
                    logger.info(
                        f"Source file {src} is newer than parquet. Re-preprocessing."
                    )
                    return False

        if source_count == 0:
            logger.info("No source files found. Re-preprocessing.")
            return False

        logger.info(
            f"Parquet is up-to-date: {parquet_num_rows} records from "
            f"{source_count} source files."
        )
        return True

    except Exception as e:
        logger.warning(f"Error checking parquet: {e}. Re-preprocessing.")
        return False


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
