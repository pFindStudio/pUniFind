"""
pUniFind: A unified deep learning framework for de novo peptide sequencing and database search rescoring.

Usage:
    pUniFind denovo <project_path> <batch_size> [options]
    pUniFind rescore <project_path> <batch_size> [options]
"""

__version__ = "1.0.0"
__author__ = "pUniFind Team"

from punifind.cli import main

__all__ = ["main", "__version__"]
