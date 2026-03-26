#!/usr/bin/env python
"""
KE Effective Resolution Batch Analysis — WeatherBench 2 Zarr Edition

Streams Aurora predictions and ERA5 ground-truth data from public WB2
Zarr buckets, computes KE spectra at 500 hPa via spherical harmonics,
and outputs a CSV of effective resolution and small-scale energy ratio.

Data Sources:
  Aurora : gs://weatherbench2/datasets/aurora/2022-1440x721.zarr
  ERA5   : gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Companion module (same directory)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from calc_ke_spectrum import (
    calculate_effective_resolution,
    spectrum_from_slice,
)


# ============================================================================
# Configuration
# ============================================================================

AURORA_ZARR = "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr"
ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"

OUTPUT_CSV = Path.home() / "aurora_thesis" / "results" / "ke_effective_resolution.csv"

# Default: every day in January 2022 (small for a first run)
DEFAULT_DATES = [f"2022-01-{d:02d}" for d in range(1, 32)]

LEVEL_HPA = 500


# ============================================================================
# Zarr helpers
# ============================================================================

def open_zarr_anonymous(url: str) -> xr.Dataset:
    """Open a public GCS Zarr store without authentication."""
    return xr.open_zarr(url, storage_options={"token": "anon"})


# ============================================================================
# Batch analysis
# ============================================================================

def run_batch_analysis(
    dates: list[str],
    level: float = LEVEL_HPA,
    output_csv: Path = OUTPUT_CSV,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute effective resolution for each date.

    Parameters
    ----------
    dates : list[str]
        ISO date strings, e.g. ["2022-01-01", "2022-02-15"].
    level : float
        Pressure level in hPa (default 500).
    output_csv : Path
        Output CSV path.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
    """
    if verbose:
        print("\n" + "=" * 70)
        print("  KE EFFECTIVE RESOLUTION — WeatherBench 2 Zarr Streaming")
        print("=" * 70)
        print(f"  Aurora : {AURORA_ZARR}")
        print(f"  ERA5   : {ERA5_ZARR}")
        print(f"  Level  : {level} hPa")
        print(f"  Dates  : {len(dates)}")
        print(f"  Output : {output_csv}")
        print("=" * 70)

    # Open stores lazily
    if verbose:
        print("\n  Opening Zarr stores (anonymous) …")
    ds_aurora_full = open_zarr_anonymous(AURORA_ZARR)
    ds_era5_full = open_zarr_anonymous(ERA5_ZARR)

    if verbose:
        print(f"  Aurora vars: {list(ds_aurora_full.data_vars)[:12]}")
        print(f"  ERA5 vars  : {list(ds_era5_full.data_vars)[:12]}")

    results: list[dict] = []

    for i, date_str in enumerate(dates, 1):
        valid_time = np.datetime64(date_str)

        if verbose:
            print(f"\n  [{i}/{len(dates)}] {valid_time} ", end="", flush=True)

        row = {
            "valid_time": str(valid_time),
            "effective_resolution_km": None,
            "small_scale_energy_ratio": None,
            "status": "OK",
        }

        try:
            # Select nearest time step
            ds_aurora_t = ds_aurora_full.sel(time=valid_time, method="nearest").load()
            ds_era5_t = ds_era5_full.sel(time=valid_time, method="nearest").load()

            # Compute spectra at the requested pressure level
            k_aurora, e_aurora = spectrum_from_slice(
                ds_aurora_t, level=level, verbose=False
            )
            k_era5, e_era5 = spectrum_from_slice(
                ds_era5_t, level=level, verbose=False
            )

            # Align lengths (both grids should be identical, but be safe)
            n = min(len(k_aurora), len(k_era5))
            k = k_aurora[:n]
            e_aurora = e_aurora[:n]
            e_era5 = e_era5[:n]

            # Effective resolution
            L_eff, ratio = calculate_effective_resolution(k, e_aurora, e_era5)

            row["effective_resolution_km"] = L_eff
            row["small_scale_energy_ratio"] = ratio

            if verbose:
                if np.isinf(L_eff):
                    print("✓  L_eff = ∞ (no cutoff)")
                else:
                    print(f"✓  L_eff = {L_eff:.0f} km, ratio = {ratio:.3f}")

        except Exception as exc:
            row["status"] = f"Error: {str(exc)[:80]}"
            if verbose:
                print(f"⚠  {exc}")

        results.append(row)

    # Save CSV
    df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if verbose:
        _print_summary(df)
        print(f"\n  ✓ Results saved → {output_csv}")

    return df


# ============================================================================
# Summary
# ============================================================================

def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    valid = df[df["status"] == "OK"]
    print(f"  Total dates:  {len(df)}")
    print(f"  Successful:   {len(valid)}")

    if len(valid) == 0:
        print("  ⚠  No successful calculations.")
        return

    L = valid["effective_resolution_km"]
    finite = L[np.isfinite(L)]

    if len(finite) > 0:
        print(f"\n  Effective resolution (finite only, n={len(finite)}):")
        print(f"    Mean : {finite.mean():.0f} km")
        print(f"    Median: {finite.median():.0f} km")
        print(f"    Min  : {finite.min():.0f} km")
        print(f"    Max  : {finite.max():.0f} km")
    else:
        print("  All dates have L_eff = ∞ (no spectral cutoff found)")

    ratio = valid["small_scale_energy_ratio"]
    print(f"\n  Small-scale energy ratio:")
    print(f"    Mean : {ratio.mean():.3f}")
    print(f"    Min  : {ratio.min():.3f}")
    print(f"    Max  : {ratio.max():.3f}")

    print("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KE Effective Resolution analysis (WB2 Zarr streaming)"
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=DEFAULT_DATES,
        help="Dates to process, e.g. 2022-01-01 2022-07-15 (default: all of Jan 2022)",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=LEVEL_HPA,
        help=f"Pressure level in hPa (default: {LEVEL_HPA})",
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
        dates=args.dates,
        level=args.level,
        output_csv=output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
