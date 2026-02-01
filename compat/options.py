# Copyright (c) 2024, pUniFind Team
# Command-line argument parsing utilities

import argparse
import sys
from typing import Optional


def get_parser(
    desc: str = "pUniFind",
    default_task: str = "score_inference",
) -> argparse.ArgumentParser:
    """
    Get a basic argument parser.

    Args:
        desc: Description for the parser
        default_task: Default task name

    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description=desc, allow_abbrev=False)

    # Required arguments
    parser.add_argument(
        "data",
        nargs="?",
        help="Path to data directory",
        default=".",
    )

    # Common arguments
    parser.add_argument(
        "--task",
        default=default_task,
        help="Task to run",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision",
    )

    # Additional common arguments that may be passed
    parser.add_argument("--user-dir", default=None, type=str)
    parser.add_argument("--ddp-backend", default="c10d", type=str)
    parser.add_argument("--loss", default="cross_entropy", type=str)

    return parser


def add_dataset_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add dataset-related arguments."""
    group = parser.add_argument_group("Dataset")

    group.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Number of data loading workers",
    )
    group.add_argument(
        "--data-buffer-size",
        default=10,
        type=int,
        help="Data buffer size",
    )
    group.add_argument(
        "--batch-size",
        "--max-sentences",
        type=int,
        default=32,
        help="Batch size",
    )
    group.add_argument(
        "--required-batch-size-multiple",
        default=1,
        type=int,
        help="Batch size must be a multiple of this value",
    )

    return parser


def add_distributed_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add distributed training arguments."""
    group = parser.add_argument_group("Distributed")

    group.add_argument(
        "--distributed-world-size",
        default=1,
        type=int,
        help="Total number of GPUs/processes",
    )
    group.add_argument(
        "--distributed-rank",
        default=0,
        type=int,
        help="Rank of current process",
    )
    group.add_argument(
        "--distributed-init-method",
        default=None,
        help="Distributed init method",
    )
    group.add_argument(
        "--device-id",
        "--local_rank",
        default=0,
        type=int,
        help="GPU device ID",
    )
    group.add_argument(
        "--distributed-backend",
        default="nccl",
        help="Distributed backend (nccl for Linux, gloo for Windows)",
    )

    return parser


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add model-specific arguments."""
    group = parser.add_argument_group("Model")

    group.add_argument(
        "--arch",
        "-a",
        default="pep_ms",
        help="Model architecture",
    )

    # Add all model arguments from MDProteinModel.add_args
    group.add_argument("--node-dim", default=512, type=int)
    group.add_argument("--num-head", default=8, type=int)
    group.add_argument("--dim-forward", default=1024, type=int)
    group.add_argument("--num-layers", default=12, type=int)
    group.add_argument("--cutoff-peptide", default=100, type=int)
    group.add_argument("--cutoff-spectra", default=200, type=int)
    group.add_argument("--debug", default=0, type=int)
    group.add_argument("--use-nce", default=0, type=int)
    group.add_argument("--use-ins", default=0, type=int)
    group.add_argument("--use-rt", default=0, type=int)
    group.add_argument("--pred-nce", default=0, type=int)
    group.add_argument("--pred-ins", default=0, type=int)
    group.add_argument("--pred-rt", default=0, type=int)
    group.add_argument("--inference", default=0, type=int)
    group.add_argument("--sample-test", default=1000, type=int)
    group.add_argument("--denovo-pred", default=0, type=int)
    group.add_argument("--use-clip", default=0, type=int)
    group.add_argument("--use-hard-clip", default=0, type=int)
    group.add_argument("--spectrum-pred", default=0, type=int)
    group.add_argument("--triplet-loss-weight", default=1, type=float)
    group.add_argument("--denovo-loss-weight", default=1, type=float)
    group.add_argument("--spectrum-loss-weight", default=1, type=float)
    group.add_argument("--peak-loss-weight", default=0.25, type=float)
    group.add_argument("--num-res-loss-weight", default=0.25, type=float)
    group.add_argument("--clip-loss-weight", default=0.1, type=float)
    group.add_argument("--modification-loss-weight", default=1, type=float)
    group.add_argument("--random-mask-prob", default=0, type=float)
    group.add_argument("--intensity-pred", default=0, type=int)
    group.add_argument("--intensity-prob", default=0.1, type=float)
    group.add_argument("--denoise-pred", default=0, type=int)
    group.add_argument("--denoise-prob", default=0.1, type=float)
    group.add_argument("--dataset-name", default="", type=str)
    group.add_argument("--enforce-triplet-loss", default=0, type=int)
    group.add_argument("--use-rope", default=0, type=int)
    group.add_argument("--noise-peak-pred", default=0, type=int)
    group.add_argument("--multi-class-noise-peak-pred", default=1, type=int)
    group.add_argument("--max-charges", default=2, type=int)
    group.add_argument("--half-spectrum-pred", default=0, type=int)
    group.add_argument("--from-scratch", default=0, type=int)
    group.add_argument("--modification-pred", default=0, type=int)
    group.add_argument("--modification-cls-pred", default=0, type=int)
    group.add_argument("--modification-mass-pred", default=0, type=int)
    group.add_argument("--norm-first", default=0, type=int)
    group.add_argument("--massive", default=0, type=int)
    group.add_argument("--res-num-pred", default=0, type=int)
    group.add_argument("--res-seq-len-pred", default=0, type=int)
    group.add_argument("--res-type-pred", default=0, type=int)
    group.add_argument("--comp-res-num-pred", default=0, type=int)
    group.add_argument("--no-mod", default=0, type=int)
    group.add_argument("--train-pfind", default=0, type=int)
    group.add_argument("--shift", default=0, type=int)
    group.add_argument("--even-op", default=3, type=int)
    group.add_argument("--pair-dim", default=64, type=int)
    group.add_argument("--use-unimol-encoder", default=0, type=int)
    group.add_argument("--head-clip", default=0, type=int)
    group.add_argument("--non-ar", default=1, type=int)
    group.add_argument("--max-prec-charge", default=10, type=int)
    group.add_argument("--hard-neg", default=1, type=int)
    group.add_argument("--num-round-pred", default=3, type=int)
    group.add_argument("--tokenize-mod", default=1, type=int)
    group.add_argument("--joint-encoding", default=0, type=int)
    group.add_argument("--joint-encoding-layers", default=4, type=int)
    group.add_argument("--num-pept-layers", default=9, type=int)
    group.add_argument("--modification-meta-dict-path", default=r"./consts/modification_meta_dict.pkl", type=str)
    group.add_argument("--tokenize1-pkl-path", default=r"./consts/tokenize1.pkl", type=str)
    group.add_argument("--start-id", default=3, type=int)
    group.add_argument("--unfreeze-step", default=-1, type=int)
    group.add_argument("--tof", default=0, type=int)
    group.add_argument("--only-decoy", default=0, type=int)
    group.add_argument("--use-rankloss", default=1, type=int)
    group.add_argument("--rank-encoding-layers", default=4, type=int)
    group.add_argument("--rank-loss-weight", default=10, type=int)
    group.add_argument("--special-rank-loss", default=0, type=int)
    group.add_argument("--weighted", default=0, type=int)
    group.add_argument("--dropout", default=0.18, type=float)
    group.add_argument("--ckpt-file", default="", type=str)

    return parser


def add_logging_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add logging arguments."""
    group = parser.add_argument_group("Logging")

    group.add_argument(
        "--log-format",
        default=None,
        choices=["json", "none", "simple", "tqdm"],
        help="Log format",
    )
    group.add_argument(
        "--log-interval",
        default=100,
        type=int,
        help="Log interval",
    )
    group.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable progress bar",
    )

    return parser


def add_validation_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add validation arguments."""
    group = parser.add_argument_group("Validation")

    group.add_argument(
        "--valid-subset",
        default="valid",
        help="Validation subset name(s), comma-separated",
    )
    group.add_argument(
        "--results-path",
        default="",
        type=str,
        help="Path to save results",
    )
    group.add_argument(
        "--run-name",
        default="run",
        type=str,
        help="Name of the run",
    )
    group.add_argument(
        "--fdr-thread",
        default=0.1,
        type=float,
        help="FDR threshold",
    )
    group.add_argument(
        "--range-pred",
        default=5,
        type=int,
        help="Range for prediction",
    )
    group.add_argument(
        "--pkl-path",
        default="",
        type=str,
        help="Path to pkl file",
    )
    group.add_argument(
        "--check",
        default=0,
        type=int,
        help="Check mode",
    )

    return parser


def get_validation_parser(desc: str = "pUniFind Validation") -> argparse.ArgumentParser:
    """
    Get a parser for validation/inference.

    Args:
        desc: Description for the parser

    Returns:
        ArgumentParser instance
    """
    parser = get_parser(desc)
    add_dataset_args(parser)
    add_distributed_args(parser)
    add_logging_args(parser)
    add_validation_args(parser)

    return parser


def parse_args_and_arch(
    parser: argparse.ArgumentParser,
    input_args: Optional[list] = None,
    parse_known: bool = False,
) -> argparse.Namespace:
    """
    Parse command-line arguments and architecture-specific arguments.

    Args:
        parser: ArgumentParser instance
        input_args: Optional list of arguments (defaults to sys.argv)
        parse_known: Whether to parse only known arguments

    Returns:
        Parsed arguments namespace
    """
    # Add model arguments if architecture is specified
    if input_args is None:
        input_args = sys.argv[1:]

    # Note: add_model_args should be called by the caller before parse_args_and_arch
    # to avoid duplicate argument definitions

    # Parse all arguments - use parse_known_args to be more flexible
    args, remaining = parser.parse_known_args(input_args)

    if remaining and not parse_known:
        # Log unknown arguments but don't fail
        print(f"Warning: Unknown arguments ignored: {remaining}")

    # Set default values for distributed
    import torch
    if not hasattr(args, 'distributed_world_size') or args.distributed_world_size is None:
        args.distributed_world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if not hasattr(args, 'distributed_rank'):
        args.distributed_rank = 0

    return args
