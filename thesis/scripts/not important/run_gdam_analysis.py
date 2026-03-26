#!/usr/bin/env python
"""
GDAM Conservation Batch Analysis — WeatherBench 2 Zarr Edition

Streams Aurora predictions and ERA5 ground-truth data directly from the
public WeatherBench 2 Google Cloud Zarr buckets, computes Global Dry Air
Mass for selected dates in 2022, and saves a comparison CSV.

Data Sources:
  Aurora : gs://weatherbench2/datasets/aurora/2022-1440x721.zarr
  ERA5   : gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr

Static fields (surface geopotential, land-sea mask) are taken from the
ERA5 store at time index 0 (they are time-invariant).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

# Import conservation functions from the companion module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from gdam_conservation import calculate_global_conservation, derive_surface_pressure


# ============================================================================
# Configuration
# ============================================================================

AURORA_ZARR = "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr"
ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"

OUTPUT_CSV = Path.home() / "aurora_thesis" / "results" / "gdam_wb2_conservation.csv"

# Default test dates (2022 init times at 00:00 UTC)
DEFAULT_INIT_DATES = [
    "2022-01-01T00",
    "2022-04-01T00",
    "2022-07-01T00",
    "2022-10-01T00",
]

# Lead times to evaluate (hours)
DEFAULT_LEAD_HOURS = [12, 24, 72]

# Variable-name mapping for the two stores.  The first match wins.
SP_NAMES = ("surface_pressure", "sp", "ps")
MSL_NAMES = ("mean_sea_level_pressure", "msl")
Q_NAMES = ("specific_humidity", "q")
T2M_NAMES = ("2m_temperature", "t2m")
GEOPOTENTIAL_NAMES = ("geopotential_at_surface", "z", "orography")

# 3-D dynamic variable names (on pressure levels)
T_NAMES = ("temperature", "t")
U_NAMES = ("u_component_of_wind", "u")
V_NAMES = ("v_component_of_wind", "v")
PHI_NAMES = ("geopotential", "z")  # dynamic 3-D field on levels


# ============================================================================
# Zarr I/O helpers
# ============================================================================

def open_zarr_anonymous(url: str) -> xr.Dataset:
    """Open a public GCS Zarr store without authentication."""
    return xr.open_zarr(url, storage_options={"token": "anon"})


def _first_var(ds: xr.Dataset, candidates: tuple[str, ...]) -> Optional[str]:
    """Return the first variable name found in *ds*, or None."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


# ============================================================================
# Static field loader
# ============================================================================

def load_static_fields(ds_era5: xr.Dataset) -> xr.Dataset:
    """
    Extract time-invariant surface geopotential (and LSM if present)
    from the ERA5 Zarr store at the first available time step.

    Returns
    -------
    xr.Dataset
        Dataset with at least one of the geopotential variables and,
        optionally, ``lsm``.
    """
    # Pick the first time to get invariant fields
    ds0 = ds_era5.isel(time=0)

    keep_vars = []
    for name in GEOPOTENTIAL_NAMES:
        if name in ds0.data_vars:
            keep_vars.append(name)
            break
    if "lsm" in ds0.data_vars:
        keep_vars.append("lsm")
    if "land_sea_mask" in ds0.data_vars:
        keep_vars.append("land_sea_mask")

    if not keep_vars:
        raise ValueError(
            "Could not find surface geopotential in ERA5 Zarr. "
            f"Tried {GEOPOTENTIAL_NAMES}. "
            f"Available: {list(ds0.data_vars)}"
        )

    ds_static = ds0[keep_vars].load()  # small — load eagerly
    return ds_static


# ============================================================================
# Core per-timestamp GDAM computation
# ============================================================================

def _find_q_name(ds: xr.Dataset) -> str:
    """Locate the specific-humidity variable."""
    name = _first_var(ds, Q_NAMES)
    if name is None:
        raise ValueError(
            f"No specific-humidity variable found. "
            f"Tried {Q_NAMES}. Available: {list(ds.data_vars)}"
        )
    return name


def _has_surface_pressure(ds: xr.Dataset) -> Optional[str]:
    """Return the SP variable name if present, else None."""
    return _first_var(ds, SP_NAMES)


def compute_conservation_for_slice(
    ds_slice: xr.Dataset,
    ds_static: xr.Dataset,
    verbose: bool = False,
) -> dict:
    """
    Compute conservation metrics for a single-timestep dataset slice.

    If surface pressure is missing, it is derived from MSL via the
    hypsometric equation using *ds_static*.

    Returns a dict with keys:
        dry_mass_eg, water_mass_kg, total_energy_J
    """
    # Squeeze time if still present
    if "time" in ds_slice.dims:
        ds_slice = ds_slice.isel(time=0)

    # Determine variable names
    sp_name = _has_surface_pressure(ds_slice)
    q_name = _find_q_name(ds_slice)

    # If SP is absent, derive it and inject into the dataset
    if sp_name is None:
        sp_derived = derive_surface_pressure(ds_slice, ds_static)
        ds_slice = ds_slice.assign(surface_pressure=sp_derived)
        sp_name = "surface_pressure"

    # Resolve dynamic 3-D variable names
    t_name = _first_var(ds_slice, T_NAMES) or "temperature"
    u_name = _first_var(ds_slice, U_NAMES) or "u_component_of_wind"
    v_name = _first_var(ds_slice, V_NAMES) or "v_component_of_wind"
    phi_name = _first_var(ds_slice, PHI_NAMES) or "geopotential"

    result = calculate_global_conservation(
        ds=ds_slice,
        ds_static=ds_static,
        ps_name=sp_name,
        tcwv_name=None,      # compute from q
        q_name=q_name,
        t_name=t_name,
        u_name=u_name,
        v_name=v_name,
        phi_name=phi_name,
        verbose=verbose,
    )
    return {
        "dry_mass_eg": result.dry_mass_exagram,
        "water_mass_kg": result.water_mass_kg,
        "total_energy_J": result.total_energy_joules,
    }


# ============================================================================
# Batch analysis
# ============================================================================

def run_batch_analysis(
    init_dates: list[str],
    lead_hours: list[int],
    output_csv: Path = OUTPUT_CSV,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run GDAM conservation analysis for the given init-dates / lead-times.

    Parameters
    ----------
    init_dates : list[str]
        Initialization timestamps as ISO-like strings, e.g. "2022-01-01T00".
    lead_hours : list[int]
        Lead times in hours to evaluate (e.g. [12, 24, 72]).
    output_csv : Path
        Where to write the results CSV.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Results with one row per (init_time, lead_time).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("  GDAM CONSERVATION — WeatherBench 2 Zarr Streaming")
        print("=" * 70)
        print(f"  Aurora Zarr : {AURORA_ZARR}")
        print(f"  ERA5 Zarr   : {ERA5_ZARR}")
        print(f"  Output CSV  : {output_csv}")
        print(f"  Init dates  : {init_dates}")
        print(f"  Lead hours  : {lead_hours}")
        print("=" * 70)

    # ---- Open stores (lazy) ----
    if verbose:
        print("\n  Opening Zarr stores (anonymous) …")
    ds_aurora_full = open_zarr_anonymous(AURORA_ZARR)
    ds_era5_full = open_zarr_anonymous(ERA5_ZARR)

    if verbose:
        print(f"  Aurora vars : {list(ds_aurora_full.data_vars)[:15]}")
        print(f"  ERA5 vars   : {list(ds_era5_full.data_vars)[:15]}")

    # ---- Load static fields once ----
    if verbose:
        print("  Loading static fields (geopotential, LSM) …")
    ds_static = load_static_fields(ds_era5_full)
    if verbose:
        print(f"  Static vars : {list(ds_static.data_vars)}")

    # ---- Loop over init-dates × lead-times ----
    results: list[dict] = []
    total = len(init_dates) * len(lead_hours)
    n = 0

    for init_str in init_dates:
        init_time = np.datetime64(init_str)
        if verbose:
            print(f"\n  [Init: {init_time}]")

        for lh in lead_hours:
            n += 1
            lead_td = np.timedelta64(lh, "h")
            valid_time = init_time + lead_td

            if verbose:
                print(f"    +{lh:3d}h → {valid_time}  ({n}/{total}) ", end="", flush=True)

            row = {
                "init_time": str(init_time),
                "lead_time": f"{lh}h",
                "era5_mass_eg": None,
                "aurora_mass_eg": None,
                "residual_eg": None,
                "relative_error_percent": None,
                "era5_water_kg": None,
                "aurora_water_kg": None,
                "water_residual_kg": None,
                "era5_energy_J": None,
                "aurora_energy_J": None,
                "energy_residual_J": None,
                "status": "OK",
            }

            try:
                # Select valid-time slice for Aurora
                ds_aurora_t = ds_aurora_full.sel(time=valid_time, method="nearest")
                # Select valid-time slice for ERA5
                ds_era5_t = ds_era5_full.sel(time=valid_time, method="nearest")

                # Load into memory (single timestep, manageable size)
                ds_aurora_t = ds_aurora_t.load()
                ds_era5_t = ds_era5_t.load()

                # Compute conservation metrics for both
                era5_metrics = compute_conservation_for_slice(ds_era5_t, ds_static, verbose=False)
                aurora_metrics = compute_conservation_for_slice(ds_aurora_t, ds_static, verbose=False)

                # Dry-air mass
                residual = aurora_metrics["dry_mass_eg"] - era5_metrics["dry_mass_eg"]
                rel_error = (
                    100.0 * residual / era5_metrics["dry_mass_eg"]
                    if era5_metrics["dry_mass_eg"] != 0 else float("nan")
                )
                row["era5_mass_eg"] = era5_metrics["dry_mass_eg"]
                row["aurora_mass_eg"] = aurora_metrics["dry_mass_eg"]
                row["residual_eg"] = residual
                row["relative_error_percent"] = rel_error

                # Water mass
                row["era5_water_kg"] = era5_metrics["water_mass_kg"]
                row["aurora_water_kg"] = aurora_metrics["water_mass_kg"]
                row["water_residual_kg"] = aurora_metrics["water_mass_kg"] - era5_metrics["water_mass_kg"]

                # Total energy
                row["era5_energy_J"] = era5_metrics["total_energy_J"]
                row["aurora_energy_J"] = aurora_metrics["total_energy_J"]
                row["energy_residual_J"] = aurora_metrics["total_energy_J"] - era5_metrics["total_energy_J"]

                if verbose:
                    print(f"✓  Δm = {residual:+.6f} Eg  ({rel_error:+.4f}%)")

            except Exception as exc:
                row["status"] = f"Error: {str(exc)[:80]}"
                if verbose:
                    print(f"⚠  {exc}")

            results.append(row)

    # ---- Save CSV ----
    df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    if verbose:
        print(f"\n  ✓ Results saved → {output_csv}")

    # ---- Summary ----
    if verbose:
        _print_summary(df)

    return df


# ============================================================================
# Summary printer
# ============================================================================

def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    valid = df[df["status"] == "OK"]
    print(f"  Total rows:  {len(df)}")
    print(f"  Successful:  {len(valid)}")

    if len(valid) == 0:
        print("  ⚠  No successful calculations.")
        return

    # Dry-air mass
    print(f"\n  [Dry Air Mass]")
    print(f"  ERA5 mass  (mean): {valid['era5_mass_eg'].mean():.6f} Eg")
    print(f"  Aurora mass (mean): {valid['aurora_mass_eg'].mean():.6f} Eg")
    print(f"  Mean residual    : {valid['residual_eg'].mean():+.6f} Eg")
    print(f"  Max |residual|   : {valid['residual_eg'].abs().max():.6f} Eg")
    print(f"  Mean rel. error  : {valid['relative_error_percent'].mean():+.6f}%")
    print(f"  Max |rel. error| : {valid['relative_error_percent'].abs().max():.6f}%")

    # Water mass
    if "era5_water_kg" in valid.columns and valid["era5_water_kg"].notna().any():
        print(f"\n  [Water Mass]")
        print(f"  ERA5 water  (mean): {valid['era5_water_kg'].mean():.4e} kg")
        print(f"  Aurora water (mean): {valid['aurora_water_kg'].mean():.4e} kg")
        print(f"  Mean residual     : {valid['water_residual_kg'].mean():+.4e} kg")

    # Total energy
    if "era5_energy_J" in valid.columns and valid["era5_energy_J"].notna().any():
        print(f"\n  [Total Energy]")
        print(f"  ERA5 energy  (mean): {valid['era5_energy_J'].mean():.4e} J")
        print(f"  Aurora energy (mean): {valid['aurora_energy_J'].mean():.4e} J")
        print(f"  Mean residual      : {valid['energy_residual_J'].mean():+.4e} J")

    # Per-lead-time breakdown
    print("\n  Per-lead-time (dry mass):")
    for lt, grp in valid.groupby("lead_time"):
        print(
            f"    {lt:>4s}  mean Δ = {grp['residual_eg'].mean():+.6f} Eg  "
            f"({grp['relative_error_percent'].mean():+.6f}%)"
        )

    print("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GDAM conservation batch analysis (WB2 Zarr streaming)"
    )
    parser.add_argument(
        "--init-dates",
        nargs="+",
        default=DEFAULT_INIT_DATES,
        help="Initialization timestamps, e.g. 2022-01-01T00 2022-07-01T00",
    )
    parser.add_argument(
        "--lead-hours",
        nargs="+",
        type=int,
        default=DEFAULT_LEAD_HOURS,
        help="Lead times in hours (default: 12 24 72)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else OUTPUT_CSV

    run_batch_analysis(
        init_dates=args.init_dates,
        lead_hours=args.lead_hours,
        output_csv=output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
