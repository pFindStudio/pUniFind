from typing import Tuple, Optional
import numpy as np
import torch


def preprocess_spectrum_fast(
    mz_array: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    precursor_charge: int,
    min_mz: float = 50.5,
    max_mz: float = 4500.0,
    min_intensity: float = 0.01,
    n_peaks: int = 300,
    remove_precursor_tol_ppm: float = 20.0,
    remove_precursor_tol_da: float = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fast spectrum preprocessing without spectrum_utils dependency.

    This is a vectorized implementation that replaces the slower spectrum_utils operations:
    - set_mz_range
    - remove_precursor_peak
    - filter_intensity
    - scale_intensity (root)

    Parameters
    ----------
    mz_array : np.ndarray
        m/z values
    intensity : np.ndarray
        Intensity values
    precursor_mz : float
        Precursor m/z
    precursor_charge : int
        Precursor charge
    min_mz, max_mz : float
        m/z range filter
    min_intensity : float
        Minimum relative intensity threshold
    n_peaks : int
        Maximum number of peaks to keep
    remove_precursor_tol_ppm : float
        PPM tolerance for precursor peak removal (used if remove_precursor_tol_da is None)
    remove_precursor_tol_da : float, optional
        Da tolerance for precursor peak removal (overrides ppm if set)

    Returns
    -------
    mz_array, intensity : Tuple[np.ndarray, np.ndarray] or (None, None) if empty
    """
    if len(mz_array) == 0:
        return None, None

    # Ensure numpy arrays
    if isinstance(mz_array, torch.Tensor):
        mz_array = mz_array.numpy()
    if isinstance(intensity, torch.Tensor):
        intensity = intensity.numpy()

    mz_array = mz_array.astype(np.float64)
    intensity = intensity.astype(np.float32)

    # 1. Filter by m/z range (vectorized)
    mask = (mz_array >= min_mz) & (mz_array <= max_mz)
    mz_array = mz_array[mask]
    intensity = intensity[mask]

    if len(mz_array) == 0:
        return None, None

    # 2. Remove precursor peak (PPM or Da tolerance, vectorized)
    if remove_precursor_tol_da is not None:
        tol = remove_precursor_tol_da
    else:
        tol = precursor_mz * remove_precursor_tol_ppm / 1e6
    mask = ~((mz_array >= precursor_mz - tol) & (mz_array <= precursor_mz + tol))
    mz_array = mz_array[mask]
    intensity = intensity[mask]

    if len(mz_array) == 0:
        return None, None

    # 3. Filter by intensity (relative threshold + top N peaks)
    max_int = intensity.max()
    if max_int > 0:
        mask = intensity >= min_intensity * max_int
        mz_array = mz_array[mask]
        intensity = intensity[mask]

    if len(mz_array) == 0:
        return None, None

    # Keep top N peaks by intensity
    if len(mz_array) > n_peaks:
        top_indices = np.argsort(intensity)[-n_peaks:]
        top_indices = np.sort(top_indices)  # Keep m/z order
        mz_array = mz_array[top_indices]
        intensity = intensity[top_indices]

    # 4. Scale intensity (root + normalize)
    intensity = np.sqrt(intensity)
    norm = np.linalg.norm(intensity)
    if norm > 0:
        intensity = intensity / norm

    return mz_array, intensity


def sqrt_and_norm(
    mz_array: torch.Tensor,
    int_array: torch.Tensor,
    precursor_mz: float,
    precursor_charge: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Square root the intensities and scale to unit norm.

    Parameters
    ----------
    mz_array : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    int_array : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    precursor_mz : float
        The precursor m/z.
    precursor_charge : int
        The precursor charge.

    Returns
    -------
    mz_array : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    int_array : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    """
    int_array = torch.sqrt(int_array)
    int_array /= torch.linalg.norm(int_array)
    return mz_array, int_array


def remove_precursor_peak(
    mz_array: torch.Tensor,
    int_array: torch.Tensor,
    precursor_mz: float,
    tol: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Square root the intensities and scale to unit norm.

    Parameters
    ----------
    mz_array : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    int_array : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    precursor_mz : float
        The precursor m/z.
    precursor_charge : int
        The precursor charge.
    tol : float, optional
        The tolerance to remove the precursor peak in m/z.

    Returns
    -------
    mz_array : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    int_array : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    """
    bounds = (precursor_mz - tol, precursor_mz + tol)
    outside = (mz_array >= bounds[0]) | (mz_array <= bounds[1])
    return mz_array[outside], int_array[outside]
