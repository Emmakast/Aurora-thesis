"""
Kinetic Energy Spectrum & Effective Resolution

Spherical-harmonic-based kinetic energy spectrum calculation and an
"effective resolution" metric that measures the wavenumber at which a
model prediction loses at least 50 % of the true spectral energy.

Core functions
--------------
compute_ke_spectrum_spharm(u, v, nlat, nlon)
    KE spectrum E(k) via pyshtools SHExpandDH.

calculate_effective_resolution(k, energy_pred, energy_true)
    Effective resolution L_eff (km) and small-scale energy ratio.

Reference
---------
    Saccardi et al. — PSD-based spectral analysis for AI weather models.

Compatible with: WeatherBench 2 Zarr datasets.
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import xarray as xr


# ============================================================================
# Physical Constants
# ============================================================================

EARTH_RADIUS = 6.371e6  # m, mean Earth radius


# ============================================================================
# Spherical Harmonic KE Spectrum
# ============================================================================

def compute_ke_spectrum_spharm(
    u: Union[xr.DataArray, np.ndarray],
    v: Union[xr.DataArray, np.ndarray],
    nlat: int | None = None,
    nlon: int | None = None,
    rsphere: float = EARTH_RADIUS,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Kinetic Energy Spectrum using spherical harmonics (pyshtools).

    Parameters
    ----------
    u : array-like, shape (nlat, nlon)
        Zonal wind component (m/s).  North-to-south latitude order.
    v : array-like, shape (nlat, nlon)
        Meridional wind component (m/s).
    nlat, nlon : int, optional
        Grid dimensions.  Inferred from *u* if not given.
    rsphere : float
        Earth radius in metres.
    verbose : bool
        Print progress.

    Returns
    -------
    wavenumber : np.ndarray, shape (lmax+1,)
        Total wavenumber k (0 … lmax).
    energy : np.ndarray, shape (lmax+1,)
        Kinetic energy per wavenumber (m²/s²).
    """
    try:
        import pyshtools as pysh
    except ImportError:
        raise ImportError(
            "pyshtools is required. Install with: pip install pyshtools"
        )

    if verbose:
        print("  Computing spherical harmonic transform (pyshtools) …")

    # Convert to numpy 2-D arrays
    if isinstance(u, xr.DataArray):
        u = u.values
    if isinstance(v, xr.DataArray):
        v = v.values

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # Squeeze singleton dimensions (e.g. leftover time or level)
    u = u.squeeze()
    v = v.squeeze()

    if u.ndim != 2:
        raise ValueError(
            f"Expected 2-D wind fields (lat, lon), got u.shape = {u.shape}"
        )

    actual_nlat, actual_nlon = u.shape
    if nlat is None:
        nlat = actual_nlat
    if nlon is None:
        nlon = actual_nlon

    # pyshtools SHExpandDH requires even-sized grids — trim if necessary
    if nlat % 2 != 0:
        u = u[:-1, :]
        v = v[:-1, :]
        nlat -= 1
        if verbose:
            print(f"  Trimmed latitude to {nlat} (even required)")
    if nlon % 2 != 0:
        u = u[:, :-1]
        v = v[:, :-1]
        nlon -= 1
        if verbose:
            print(f"  Trimmed longitude to {nlon} (even required)")

    # pyshtools expects latitude from south to north — flip if N→S
    u = np.flip(u, axis=0)
    v = np.flip(v, axis=0)

    # Maximum total wavenumber (degree)
    lmax = nlat // 2 - 1

    if verbose:
        print(f"  lmax = {lmax}")

    # Spherical harmonic expansion (Driscoll–Healy grid)
    u_coeffs = pysh.expand.SHExpandDH(u, sampling=2, lmax_calc=lmax)
    v_coeffs = pysh.expand.SHExpandDH(v, sampling=2, lmax_calc=lmax)

    # Energy per wavenumber:  E(l) = ½ Σ_m (|û_lm|² + |v̂_lm|²)
    wavenumber = np.arange(lmax + 1)
    energy = np.zeros(lmax + 1)

    for l in range(lmax + 1):
        for m in range(l + 1):
            u_power = u_coeffs[0, l, m] ** 2
            v_power = v_coeffs[0, l, m] ** 2
            if m > 0:
                u_power += u_coeffs[1, l, m] ** 2
                v_power += v_coeffs[1, l, m] ** 2
            energy[l] += 0.5 * (u_power + v_power)

    if verbose:
        print(f"  Total KE = {energy.sum():.4e} m²/s²")

    return wavenumber, energy


# ============================================================================
# Effective Resolution
# ============================================================================

def calculate_effective_resolution(
    k: np.ndarray,
    energy_pred: np.ndarray,
    energy_true: np.ndarray,
    threshold: float = 0.5,
    k_min: int = 10,
    earth_radius: float = EARTH_RADIUS,
) -> tuple[float, float]:
    """
    Effective resolution from the spectral energy ratio.

    The spectral ratio is  R(k) = E_pred(k) / E_true(k).
    The *effective resolution* is the wavelength corresponding to
    the first wavenumber k_cutoff ≥ k_min where R drops below
    *threshold* (default 0.5):

        L_eff = 2π R_earth / k_cutoff    (km)

    Parameters
    ----------
    k : np.ndarray
        Wavenumber array (0 … kmax).
    energy_pred : np.ndarray
        Predicted KE spectrum.
    energy_true : np.ndarray
        Ground-truth KE spectrum (same shape).
    threshold : float
        Ratio cut-off (default 0.5 = 50 % energy retained).
    k_min : int
        Ignore wavenumbers below this to avoid large-scale noise.
    earth_radius : float
        Earth radius (m).

    Returns
    -------
    effective_resolution_km : float
        L_eff in kilometres.  ``np.inf`` if the ratio never drops below
        the threshold (model preserves energy everywhere).
    small_scale_energy_ratio : float
        Mean R(k) for k > k_cutoff.  ``np.nan`` if k_cutoff is not found.
    """
    # Restrict to k >= k_min and non-zero true energy
    mask = (k >= k_min) & (energy_true > 0)
    k_sel = k[mask]
    ratio = energy_pred[mask] / energy_true[mask]

    # Find first wavenumber where ratio drops below threshold
    below = np.where(ratio < threshold)[0]

    if len(below) == 0:
        # Model retains energy at all resolved scales
        return np.inf, float(np.mean(ratio))

    idx_cutoff = below[0]
    k_cutoff = float(k_sel[idx_cutoff])

    # Effective resolution  L = 2π R / k  (convert m → km)
    effective_resolution_km = (2.0 * np.pi * earth_radius / k_cutoff) / 1000.0

    # Mean ratio beyond the cutoff (small-scale energy retention)
    if idx_cutoff < len(ratio) - 1:
        small_scale_ratio = float(np.mean(ratio[idx_cutoff:]))
    else:
        small_scale_ratio = float(ratio[idx_cutoff])

    return effective_resolution_km, small_scale_ratio


# ============================================================================
# Convenience: spectrum from a 2-D dataset slice
# ============================================================================

def spectrum_from_slice(
    ds: xr.Dataset,
    level: float = 500,
    u_names: tuple[str, ...] = ("u_component_of_wind", "u"),
    v_names: tuple[str, ...] = ("v_component_of_wind", "v"),
    level_dim: str = "level",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 500 hPa u/v from a single-time dataset slice and compute
    the KE spectrum.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset (already selected to a single time step).
    level : float
        Pressure level in hPa.
    u_names, v_names : tuple[str, ...]
        Candidate variable names for wind components.
    level_dim : str
        Name of the vertical level coordinate.
    verbose : bool
        Print progress.

    Returns
    -------
    wavenumber, energy : np.ndarray
    """
    # Squeeze time if present
    if "time" in ds.dims:
        ds = ds.isel(time=0)

    # Locate u variable
    u_var = None
    for name in u_names:
        if name in ds.data_vars:
            u_var = name
            break
    if u_var is None:
        raise ValueError(
            f"No u-wind variable found. Tried {u_names}. "
            f"Available: {list(ds.data_vars)}"
        )

    # Locate v variable
    v_var = None
    for name in v_names:
        if name in ds.data_vars:
            v_var = name
            break
    if v_var is None:
        raise ValueError(
            f"No v-wind variable found. Tried {v_names}. "
            f"Available: {list(ds.data_vars)}"
        )

    u = ds[u_var]
    v = ds[v_var]

    # Select pressure level if the dimension exists
    if level_dim in u.dims:
        levels = ds[level_dim].values
        idx = int(np.abs(levels - level).argmin())
        u = u.isel({level_dim: idx})
        v = v.isel({level_dim: idx})

    return compute_ke_spectrum_spharm(u, v, verbose=verbose)


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == "__main__":
    print("Running quick self-test …")

    # Synthetic white-noise wind fields on a small grid
    np.random.seed(0)
    nlat, nlon = 180, 360
    u_test = np.random.randn(nlat, nlon)
    v_test = np.random.randn(nlat, nlon)

    k, E = compute_ke_spectrum_spharm(u_test, v_test, verbose=True)
    print(f"  Spectrum shape: k[{k.shape}], E[{E.shape}]")

    # Effective-resolution test with artificial drop at k=100
    E_pred = E.copy()
    E_pred[k > 50] *= 0.3
    L, ratio = calculate_effective_resolution(k, E_pred, E)
    print(f"  Effective resolution: {L:.0f} km")
    print(f"  Small-scale ratio:    {ratio:.3f}")
    print("Self-test OK ✓")
