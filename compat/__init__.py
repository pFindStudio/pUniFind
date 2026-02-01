# Copyright (c) 2024, pUniFind Team
# This module replaces unicore functionality with cross-platform compatible implementations
# Supports both Windows and Linux for multi-GPU inference

from .checkpoint_utils import load_checkpoint_to_cpu
from .distributed_utils import (
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_data_parallel_group,
    all_gather_list,
    call_main,
    infer_init_method,
)
from .options import (
    get_validation_parser,
    add_model_args,
    parse_args_and_arch,
)
from .tasks import UnicoreTask, register_task, setup_task, TASK_REGISTRY
from .models import BaseUnicoreModel, register_model, register_model_architecture, build_model, MODEL_REGISTRY
from .data import (
    UnicoreDataset,
    BaseWrapperDataset,
    EpochShuffleDataset,
    NestedDictionaryDataset,
)
from .modules import LayerNorm, SelfMultiheadAttention, softmax_dropout
from .utils import move_to_cuda
from .progress_bar import progress_bar

# Initialize registries by importing model and task definitions
from . import init_registry

__all__ = [
    # checkpoint_utils
    'load_checkpoint_to_cpu',
    # distributed_utils
    'get_data_parallel_world_size',
    'get_data_parallel_rank',
    'get_data_parallel_group',
    'all_gather_list',
    'call_main',
    'infer_init_method',
    # options
    'get_validation_parser',
    'add_model_args',
    'parse_args_and_arch',
    # tasks
    'UnicoreTask',
    'register_task',
    'setup_task',
    'TASK_REGISTRY',
    # models
    'BaseUnicoreModel',
    'register_model',
    'register_model_architecture',
    'build_model',
    'MODEL_REGISTRY',
    # data
    'UnicoreDataset',
    'BaseWrapperDataset',
    'EpochShuffleDataset',
    'NestedDictionaryDataset',
    # modules
    'LayerNorm',
    'SelfMultiheadAttention',
    'softmax_dropout',
    # utils
    'move_to_cuda',
    # progress_bar
    'progress_bar',
]
