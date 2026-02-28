"""
End-to-end tests for pUniFind denovo and rescore workflows.

These tests run the full pipeline on demo data and compare output
against golden reference files in pUniFind_result_label/.

Requirements:
    - GPU with CUDA
    - pUniFind installed (pip install -e .)
    - Model checkpoint at ckpts/checkpoint_rank.pt

Usage:
    pytest tests/test_e2e.py -v -s
    pytest tests/test_e2e.py -v -s -k denovo   # run only denovo test
    pytest tests/test_e2e.py -v -s -k score     # run only score test
"""

import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Repo root: tests/ is one level below
REPO_ROOT = Path(__file__).resolve().parent.parent

# Numeric tolerance for floating-point comparison
RTOL = 1e-3  # relative tolerance
ATOL = 1e-4  # absolute tolerance


def _is_close(a: str, b: str) -> bool:
    """Check if two tokens are close enough (numeric tolerance or exact match)."""
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    if math.isnan(fa) and math.isnan(fb):
        return True
    return math.isclose(fa, fb, rel_tol=RTOL, abs_tol=ATOL)


def _compare_files(generated: Path, golden: Path, sep: str = None) -> list:
    """Compare two text files with numeric tolerance.

    Each line is split by `sep` (None = any whitespace for .spectra,
    ',' for .csv). Numeric tokens are compared with tolerance,
    non-numeric tokens must match exactly.

    Returns a list of (line_number, field_idx, gen_token, gold_token)
    for the first mismatched field per line. Empty list = files match.
    """
    gen_lines = generated.read_text().splitlines()
    gold_lines = golden.read_text().splitlines()

    diffs = []

    if len(gen_lines) != len(gold_lines):
        diffs.append((0, -1,
                       f"<{len(gen_lines)} lines>",
                       f"<{len(gold_lines)} lines>"))

    for i in range(min(len(gen_lines), len(gold_lines))):
        gen_tokens = gen_lines[i].split(sep)
        gold_tokens = gold_lines[i].split(sep)

        if len(gen_tokens) != len(gold_tokens):
            diffs.append((i + 1, -1, gen_lines[i][:120], gold_lines[i][:120]))
            continue

        for j, (gt, gd) in enumerate(zip(gen_tokens, gold_tokens)):
            if not _is_close(gt, gd):
                diffs.append((i + 1, j, gt, gd))
                break  # report first bad field per line

    return diffs


def _print_diffs(diffs: list, max_show: int = 10):
    """Print the first N diffs in a human-readable format."""
    print(f"\n{'='*80}")
    print(f"MISMATCH: {len(diffs)} lines differ")
    print(f"{'='*80}")
    for line_no, field_idx, gen, gold in diffs[:max_show]:
        print(f"\n--- Line {line_no}, field {field_idx} ---")
        print(f"  Generated: {gen[:200]}")
        print(f"  Expected:  {gold[:200]}")
    if len(diffs) > max_show:
        print(f"\n  ... and {len(diffs) - max_show} more mismatched lines")
    print(f"{'='*80}\n")


def _cleanup_dirs(project_path: Path):
    """Remove generated directories from a project."""
    for dirname in ("pUniFind_result", "pUniFind_data_tmp"):
        d = project_path / dirname
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


@pytest.mark.slow
def test_denovo_demo():
    """Test de novo peptide sequencing on HLA_denovo_demo data."""
    project_path = REPO_ROOT / "projects" / "HLA_denovo_demo"
    golden_file = project_path / "pUniFind_result_label" / "HLA_denovo_demo_5_merged.csv"
    generated_file = project_path / "pUniFind_result" / "HLA_denovo_demo_5_merged.csv"

    assert golden_file.exists(), f"Golden reference not found: {golden_file}"

    # Clean up any leftover files from previous runs
    _cleanup_dirs(project_path)

    try:
        # Run pUniFind denovo
        result = subprocess.run(
            ["pUniFind", "denovo", str(project_path), "64"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
        )

        # Print stdout/stderr for debugging
        if result.stdout:
            print(result.stdout[-2000:])
        if result.stderr:
            print(result.stderr[-2000:])

        assert result.returncode == 0, (
            f"pUniFind denovo exited with code {result.returncode}\n"
            f"stderr: {result.stderr[-1000:]}"
        )

        # Check output file exists
        assert generated_file.exists(), (
            f"Output file not generated: {generated_file}"
        )

        # Compare against golden reference (CSV: comma-separated)
        diffs = _compare_files(generated_file, golden_file, sep=",")
        if diffs:
            _print_diffs(diffs)
            pytest.fail(
                f"De novo output differs from golden reference: "
                f"{len(diffs)} lines mismatch"
            )

    finally:
        _cleanup_dirs(project_path)


@pytest.mark.slow
def test_score_demo():
    """Test database search rescoring on HLA_score_demo data."""
    project_path = REPO_ROOT / "projects" / "HLA_score_demo"
    golden_file = project_path / "pUniFind_result_label" / "HLA_score_demofdr0.01_pUniFind.spectra"
    generated_file = project_path / "pUniFind_result" / "HLA_score_demofdr0.01_pUniFind.spectra"

    assert golden_file.exists(), f"Golden reference not found: {golden_file}"

    # Clean up any leftover files from previous runs
    _cleanup_dirs(project_path)

    try:
        # Run pUniFind rescore
        result = subprocess.run(
            ["pUniFind", "rescore", str(project_path), "64"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
        )

        # Print stdout/stderr for debugging
        if result.stdout:
            print(result.stdout[-2000:])
        if result.stderr:
            print(result.stderr[-2000:])

        assert result.returncode == 0, (
            f"pUniFind rescore exited with code {result.returncode}\n"
            f"stderr: {result.stderr[-1000:]}"
        )

        # Check output file exists
        assert generated_file.exists(), (
            f"Output file not generated: {generated_file}"
        )

        # Compare against golden reference (TSV: tab-separated)
        diffs = _compare_files(generated_file, golden_file, sep="\t")
        if diffs:
            _print_diffs(diffs)
            pytest.fail(
                f"Score output differs from golden reference: "
                f"{len(diffs)} lines mismatch"
            )

    finally:
        _cleanup_dirs(project_path)
