#!/usr/bin/env python
"""
pUniFind Command Line Interface

Usage:
    pUniFind denovo <project_path> <batch_size> [options]
    pUniFind rescore <project_path> <batch_size> [options]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_pytorch_installation():
    """Check if PyTorch is installed and provide helpful error message if not."""
    try:
        import torch
        return True
    except ImportError:
        pass
    except OSError as e:
        # DLL loading error on Windows
        error_msg = str(e)
        print("=" * 60)
        print("ERROR: PyTorch DLL loading failed")
        print("=" * 60)
        print(f"\nError details: {error_msg}")
        print("\nThis usually means PyTorch installation is corrupted or")
        print("incompatible with your system. Please reinstall PyTorch:")
        print("\n1. Uninstall current PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print("\n2. Reinstall based on your system:")
        print("   - CPU only:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("   - CUDA 11.8:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("   - CUDA 12.1:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("\nVisit https://pytorch.org/get-started/locally/ for more options.")
        print("=" * 60)
        sys.exit(1)

    # ImportError case
    print("=" * 60)
    print("ERROR: PyTorch is not installed")
    print("=" * 60)
    print("\npUniFind requires PyTorch. Please install it first:")
    print("\n  For CPU only:")
    print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("\n  For CUDA 11.8 (recommended for most GPUs):")
    print("    pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("\n  For CUDA 12.1:")
    print("    pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("\nVisit https://pytorch.org/get-started/locally/ for more options.")
    print("=" * 60)
    sys.exit(1)


def get_package_root():
    """Get the root directory of the installed package."""
    return Path(__file__).parent.parent


def get_default_paths():
    """Get default paths for model weights and const files."""
    pkg_root = get_package_root()

    # Try to import consts module for paths
    try:
        from consts import get_tokenize1_path, get_modification_meta_dict_path
        tokenize1_path = get_tokenize1_path()
        modification_meta_dict_path = get_modification_meta_dict_path()
    except ImportError:
        # Fallback to direct path resolution
        tokenize1_path = pkg_root / "consts" / "tokenize1.pkl"
        modification_meta_dict_path = pkg_root / "consts" / "modification_meta_dict.pkl"

    return {
        "weight_path": pkg_root / "ckpts" / "checkpoint_rank.pt",
        "tokenize1_pkl_path": tokenize1_path,
        "modification_meta_dict_path": modification_meta_dict_path,
    }


def get_gpu_count():
    """Get the number of available GPUs."""
    # First ensure PyTorch is properly installed
    check_pytorch_installation()

    import torch
    if torch.cuda.is_available():
        return torch.cuda.device_count()

    # Fallback: try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().split("\n") if l])
    except FileNotFoundError:
        pass

    return 1  # Default to 1 if can't detect


def build_torchrun_args(n_gpu, master_port, master_ip, world_size, world_rank):
    """Build torchrun command arguments.

    Uses 'python -m torch.distributed.run' instead of 'torchrun' to ensure
    we use the Python interpreter from the current environment (e.g., conda).
    This avoids issues where system-installed torchrun might conflict with
    conda environment packages.
    """
    return [
        sys.executable,  # Use current Python interpreter (conda env)
        "-m", "torch.distributed.run",
        f"--nproc_per_node={n_gpu}",
        f"--master_port={master_port}",
        f"--nnodes={world_size}",
        f"--node_rank={world_rank}",
        f"--master_addr={master_ip}",
    ]


def build_common_args(args, task_type):
    """Build common arguments for both denovo and rescore tasks."""
    defaults = get_default_paths()
    pkg_root = get_package_root()

    # Use absolute paths for project
    project_path = Path(args.project_path).resolve()

    # Setup paths
    mgf_path = project_path / "mgfs"
    qry_res_path = project_path / "result"
    results_path = project_path / "pUniFind_result"
    data_path = project_path / "pUniFind_data_tmp"

    # Create directories
    results_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)
    (data_path / "pkls").mkdir(parents=True, exist_ok=True)

    # Use user-specified or default weight path
    weight_path = args.ckpt if args.ckpt else defaults["weight_path"]
    tokenize1_pkl_path = args.tokenize1_pkl_path if args.tokenize1_pkl_path else defaults["tokenize1_pkl_path"]
    modification_meta_dict_path = args.modification_meta_dict_path if args.modification_meta_dict_path else defaults["modification_meta_dict_path"]

    # Get project name from path
    project_name = project_path.name

    # Task-specific settings
    if task_type == "denovo":
        task = "denovo_inference"
        denovo_pred = 1
        spectrum_pred = 1
        modification_pred = 1
        res_seq_len_pred = 1
        default_batch_size = 128
    else:  # rescore
        task = "score_inference"
        denovo_pred = 0
        spectrum_pred = 0
        modification_pred = 0
        res_seq_len_pred = 0
        default_batch_size = 256

    batch_size = args.batch_size if args.batch_size else default_batch_size

    # Determine workflow script path
    if task_type == "denovo":
        workflow_script = pkg_root / "official_denovo_workflow.py"
    else:
        workflow_script = pkg_root / "official_score_workflow.py"

    cmd_args = [
        str(workflow_script),
        str(data_path),
        f"--valid-subset={project_name}",
        f"--results-path={results_path}",
        f"--tmp-data-path={data_path / f'{project_name}.lmdb'}",
        f"--project-name={project_name}",
        f"--weight-path={weight_path}",
        f"--qry-res-path={qry_res_path}",
        f"--mgf-path={mgf_path}",
        f"--num-proc={args.num_proc}",
        "--data-buffer-size=32",
        f"--num-workers={args.num_workers}",
        "--ddp-backend=c10d",
        f"--batch-size={batch_size}",
        f"--task={task}",
        "--loss=cross_entropy",
        "--arch=pep_ms",
        f"--node-dim={args.node_dim}",
        f"--num-layers={args.num_layers}",
        f"--num-head={args.num_head}",
        f"--use-nce={args.use_nce}",
        f"--use-ins={args.use_ins}",
        f"--use-clip={args.use_clip}",
        f"--use-hard-clip={args.use_hard_clip}",
        f"--denovo-pred={denovo_pred}",
        f"--spectrum-pred={spectrum_pred}",
        "--denoise-pred=0",
        "--intensity-pred=0",
        "--dataset-name=_",
        f"--max-charges={args.max_charges}",
        "--use-rope=0",
        "--noise-peak-pred=0",
        "--half-spectrum-pred=0",
        "--from-scratch=1",
        "--norm-first=0",
        f"--modification-pred={modification_pred}",
        "--modification-cls-pred=0",
        "--multi-class-noise-peak-pred=1",
        "--res-num-pred=0",
        "--res-type-pred=0",
        f"--res-seq-len-pred={res_seq_len_pred}",
        f"--cutoff-spectra={args.cutoff_spectra}",
        "--comp-res-num-pred=0",
        "--train-pfind=1",
        "--hard-neg=1",
        "--shift=1",
        "--head-clip=1",
        f"--fdr-thread={args.fdr_threshold}",
        f"--pkl-path={data_path}",
        f"--joint-encoding-layers={args.joint_encoding_layers}",
        f"--num-pept-layers={args.num_pept_layers}",
        "--inference=1",
        "--check=0",
        "--joint-encoding=1",
        f"--tof={args.tof}",
        f"--tokenize1-pkl-path={tokenize1_pkl_path}",
        f"--modification-meta-dict-path={modification_meta_dict_path}",
    ]

    # Add denovo-specific args
    if task_type == "denovo":
        cmd_args.extend([
            f"--max-length={args.max_length}",
            f"--min-length={args.min_length}",
            f"--range-pred={args.range_pred}",
        ])

    return cmd_args


def run_denovo(args):
    """Run de novo peptide sequencing workflow."""
    print("=" * 40)
    print("pUniFind De Novo Peptide Sequencing")
    print("=" * 40)

    pkg_root = get_package_root()

    # Get GPU count
    n_gpu = args.n_gpu if args.n_gpu else get_gpu_count()

    # Build torchrun command
    torchrun_args = build_torchrun_args(
        n_gpu=n_gpu,
        master_port=args.master_port,
        master_ip=args.master_ip,
        world_size=args.world_size,
        world_rank=args.world_rank,
    )

    # Build workflow arguments
    workflow_args = build_common_args(args, "denovo")

    # Full command
    cmd = torchrun_args + workflow_args

    print(f"Project path: {args.project_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using {n_gpu} GPU(s)")
    print()

    # Set environment and run
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pkg_root) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, env=env, cwd=str(pkg_root))
    return result.returncode


def run_rescore(args):
    """Run database search rescoring workflow."""
    print("=" * 40)
    print("pUniFind Database Search Rescoring")
    print("=" * 40)

    pkg_root = get_package_root()

    # Get GPU count
    n_gpu = args.n_gpu if args.n_gpu else get_gpu_count()

    # Build torchrun command
    torchrun_args = build_torchrun_args(
        n_gpu=n_gpu,
        master_port=args.master_port,
        master_ip=args.master_ip,
        world_size=args.world_size,
        world_rank=args.world_rank,
    )

    # Build workflow arguments
    workflow_args = build_common_args(args, "rescore")

    # Full command
    cmd = torchrun_args + workflow_args

    print(f"Project path: {args.project_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using {n_gpu} GPU(s)")
    print()

    # Set environment and run
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pkg_root) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, env=env, cwd=str(pkg_root))
    return result.returncode


def add_common_arguments(parser):
    """Add common arguments shared by denovo and rescore."""
    # Required arguments
    parser.add_argument("project_path", type=str, help="Path to project directory (can be any path)")
    parser.add_argument("batch_size", type=int, help="Batch size for inference")

    # Optional model/data paths
    parser.add_argument("--ckpt", "--weight-path", type=str, default=None, dest="ckpt",
                        help="Path to model checkpoint (default: <install_dir>/ckpts/checkpoint_rank.pt)")
    parser.add_argument("--tokenize1-pkl-path", type=str, default=None, help="Path to tokenizer pickle")
    parser.add_argument("--modification-meta-dict-path", type=str, default=None, help="Path to modification metadata")

    # Distributed training settings
    parser.add_argument("--n-gpu", type=int, default=None, help="Number of GPUs (auto-detect if not specified)")
    parser.add_argument("--master-port", type=int, default=10077, help="Master port for distributed training")
    parser.add_argument("--master-ip", type=str, default="127.0.0.1", help="Master IP address")
    parser.add_argument("--world-size", type=int, default=1, help="Number of nodes")
    parser.add_argument("--world-rank", type=int, default=0, help="Rank of current node")

    # Model hyperparameters
    parser.add_argument("--node-dim", type=int, default=512, help="Node dimension")
    parser.add_argument("--num-layers", type=int, default=9, help="Number of encoder layers")
    parser.add_argument("--num-pept-layers", type=int, default=9, help="Number of peptide layers")
    parser.add_argument("--num-head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--joint-encoding-layers", type=int, default=4, help="Number of joint encoding layers")

    # Processing settings
    parser.add_argument("--num-proc", type=int, default=16, help="Number of processes for data preprocessing")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers (increase for better GPU utilization)")
    parser.add_argument("--fdr-threshold", type=float, default=0.1, help="FDR threshold")
    parser.add_argument("--max-charges", type=int, default=4, help="Maximum charge state")
    parser.add_argument("--cutoff-spectra", type=int, default=300, help="Spectrum cutoff")
    parser.add_argument("--tof", type=int, default=0, help="TOF mode")

    # Feature flags
    parser.add_argument("--use-nce", type=int, default=1, help="Use NCE")
    parser.add_argument("--use-ins", type=int, default=1, help="Use instrument info")
    parser.add_argument("--use-clip", type=int, default=1, help="Use CLIP")
    parser.add_argument("--use-hard-clip", type=int, default=0, help="Use hard CLIP")


def main():
    """Main entry point for pUniFind CLI."""
    parser = argparse.ArgumentParser(
        prog="pUniFind",
        description="pUniFind: A unified deep learning framework for de novo peptide sequencing and database search rescoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    pUniFind denovo /path/to/project 32
    pUniFind rescore /path/to/project 64
    pUniFind denovo /path/to/project 32 --ckpt /path/to/checkpoint.pt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # De novo subcommand
    denovo_parser = subparsers.add_parser(
        "denovo",
        help="Run de novo peptide sequencing",
        description="De novo peptide sequencing from MS/MS spectra"
    )
    add_common_arguments(denovo_parser)
    denovo_parser.add_argument("--max-length", type=int, default=40, help="Maximum peptide length")
    denovo_parser.add_argument("--min-length", type=int, default=6, help="Minimum peptide length")
    denovo_parser.add_argument("--range-pred", type=int, default=5, help="Range prediction")

    # Rescore subcommand
    rescore_parser = subparsers.add_parser(
        "rescore",
        help="Run database search rescoring",
        description="Rescore peptide-spectrum matches from database search results"
    )
    add_common_arguments(rescore_parser)

    # Version
    parser.add_argument("--version", action="version", version="pUniFind 1.0.0")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "denovo":
        sys.exit(run_denovo(args))
    elif args.command == "rescore":
        sys.exit(run_rescore(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
