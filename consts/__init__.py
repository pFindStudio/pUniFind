"""
Constants and pre-computed data for pUniFind.

This package contains:
- tokenize1.pkl: Tokenizer for peptide sequences
- modification_meta_dict.pkl: Metadata for post-translational modifications
- modification_name_dict.pkl: Names of modifications
"""

import os
from pathlib import Path

# Get the directory containing this __init__.py
_CONSTS_DIR = Path(__file__).parent


def get_consts_path():
    """Return the path to the consts directory."""
    return _CONSTS_DIR


def get_tokenize1_path():
    """Return the path to tokenize1.pkl."""
    return _CONSTS_DIR / "tokenize1.pkl"


def get_modification_meta_dict_path():
    """Return the path to modification_meta_dict.pkl."""
    return _CONSTS_DIR / "modification_meta_dict.pkl"


def get_modification_name_dict_path():
    """Return the path to modification_name_dict.pkl."""
    return _CONSTS_DIR / "modification_name_dict.pkl"
