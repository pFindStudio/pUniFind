# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json

import numpy as np


def str_hash(text: str):
    hash = 0
    for ch in text:
        hash = (hash * 281 ^ ord(ch) * 997) & 0xFFFFFFFF
    return hash


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds, key=None):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return

    def check_seed(s):
        # Use isinstance to properly handle int subclasses and numpy integer types
        assert isinstance(s, (int, np.integer)), f"Seed must be an integer, got {type(s)}: {s}"

    check_seed(seed)
    # Filter out None values from addl_seeds
    addl_seeds = tuple(s for s in addl_seeds if s is not None)
    if len(addl_seeds) > 0:
        for s in addl_seeds:
            check_seed(s)
        seed = int(hash((seed, *addl_seeds)) % 1e8)
        if key is not None:
            seed = int(hash((seed, str_hash(key))) % 1e8)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def load_json_file(file_path):
    """读取JSON文件并返回内容"""
    with open(file_path, "r") as f:
        return json.load(f)


def create_reverse_mapping(original_dict):
    """创建反向映射字典"""
    reverse_dict = {}
    for key, values in original_dict.items():
        for value in values:
            reverse_dict[value] = key
    return reverse_dict


mass2mod = load_json_file("./PepMS/data/modification_mc_dict.json")
mod2mass = create_reverse_mapping(mass2mod)
