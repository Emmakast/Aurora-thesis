#!/usr/bin/env python
"""
Level Ablation Study — 3-Level Harmonized Conservation Metrics

Compares Aurora, Pangu-Weather, and HRES using ONLY Aurora's 3 pressure
levels for column-integrated conservation metrics (dry mass, water mass,
total energy).  This isolates the effect of vertical discretisation from
model-specific physics.

Output goes to results/ablation_3levels/ to avoid overwriting main results.

Usage
-----
  python run_physics_evaluation_level_ablation.py --model aurora
  python run_physics_evaluation_level_ablation.py --model pangu --workers 16
  python run_physics_evaluation_level_ablation.py --model aurora --dates 2022-01-01
"""

from __future__ import annotations

import argparse
import calendar
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Companion library (same directory — NOT copied)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from physics_metrics import (
    compute_conservation_scalars,
    compute_drift_percentages,
    compute_drift_slope,
    derive_surface_pressure,
    get_grid_cell_area,
    _find_var,
    _detect_level_dim,
    _detect_pred_td_dim,
    SP_NAMES,
    MSL_NAMES,
    T_NAMES,
    ZSFC_NAMES,
    Q_NAMES,
    GRAVITY,
    R_DRY,
    LAPSE_RATE,
    PHI_NAMES,
)

warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*prediction_timedelta.*")


# ============================================================================
# Configuration
# ============================================================================

ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

# Model Zarr URLs — all must contain 2022 data
MODEL_ZARRS = {
    "aurora": "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr",
    "pangu":  "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
    "hres":   "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
}

# We use a single lead time (10d) whose drift window [6h → 240h]
# covers the full forecast range — but limit to MAX_TIMESTEPS steps.
LEAD_TIMES = [
    ("10d", np.timedelta64(240, "h")),
]

DRIFT_WINDOW_END = {
    240: np.timedelta64(240, "h"),
}

MAX_TIMESTEPS = 10   # cap on number of timesteps in the drift window
DEFAULT_WORKERS = 8
YEAR = 2022

OUTPUT_DIR = Path.home() / "aurora_thesis" / "thesis" / "results" / "ablation_3levels"


# ============================================================================
# Zarr I/O  (identical to main script)
# ============================================================================

def open_zarr_anonymous(url: str) -> xr.Dataset:
    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    rename = {}
    for v in ds.data_vars:
        if v != v.strip():
            rename[v] = v.strip()
    for d in ds.dims:
        if d != d.strip():
            rename[d] = d.strip()
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def load_static_fields(ds_era5: xr.Dataset) -> xr.Dataset:
    static_vars = {}

    def _extract_static(ds, name):
        var = ds[name]
        if "time" in var.dims:
            var = var.isel(time=0, drop=True)
        return var

    for name in ("geopotential_at_surface", "z", "orography"):
        if name in ds_era5.data_vars:
            static_vars[name] = _extract_static(ds_era5, name)
            break

    for name in ("land_sea_mask", "lsm"):
        if name in ds_era5.data_vars:
            static_vars[name] = _extract_static(ds_era5, name)
            break

    if not static_vars:
        raise ValueError(f"No static fields found. Available: {list(ds_era5.data_vars)[:20]}")

    return xr.Dataset(static_vars)


def _get_ps(ds, ds_static, level_dim="level"):
    """Return surface pressure, deriving from MSL if needed."""
    sp_name = _find_var(ds, SP_NAMES)
    if sp_name is not None:
        sp = ds[sp_name]
        sp.attrs["derivation_method"] = "direct_sp"
        return sp

    try:
        sp = derive_surface_pressure(ds, ds_static)
        sp.attrs["derivation_method"] = "hypsometric_msl"
        return sp
    except (ValueError, KeyError):
        pass

    # Barometric fallback
    eff_level_dim = level_dim if level_dim in ds.dims else _detect_level_dim(ds)
    if eff_level_dim not in ds.dims:
        raise ValueError(f"Cannot derive surface pressure: no SP, MSL, or levels. Vars: {list(ds.data_vars)}")

    levels = ds[eff_level_dim].values
    idx_bot = int(levels.argmax())
    p_bot = float(levels[idx_bot]) * 100.0

    phi_name = _find_var(ds, PHI_NAMES)
    t_name = _find_var(ds, T_NAMES)
    if phi_name is None or t_name is None:
        raise ValueError(f"Cannot derive SP from levels: need geopotential and temperature. Vars: {list(ds.data_vars)}")

    phi_bot = ds[phi_name].isel({eff_level_dim: idx_bot})
    T_bot = ds[t_name].isel({eff_level_dim: idx_bot})

    z_sfc_name = _find_var(ds_static, ZSFC_NAMES)
    if z_sfc_name is None:
        raise ValueError(f"No z_sfc in static fields: {list(ds_static.data_vars)}")
    z_sfc = ds_static[z_sfc_name]
    for tdim in ("time", "valid_time"):
        if tdim in z_sfc.dims:
            z_sfc = z_sfc.isel({tdim: 0}, drop=True)

    lat_name = "latitude"
    if lat_name in z_sfc.dims and lat_name in phi_bot.dims:
        n_target = phi_bot.sizes[lat_name]
        n_static = z_sfc.sizes[lat_name]
        if n_static != n_target:
            z_sfc = z_sfc.interp({lat_name: phi_bot[lat_name]}, method="nearest")

    phi_sfc = z_sfc
    sp = p_bot * np.exp((phi_sfc - phi_bot) / (R_DRY * T_bot))
    sp = sp.astype(np.float64)
    sp.name = "surface_pressure"
    sp.attrs = {"units": "Pa", "long_name": "Surface pressure (bottom-level estimate)",
                "derivation_method": "barometric_fallback"}
    return sp


# ============================================================================
# Grid helpers
# ============================================================================

def _grids_match(ds_a, ds_b, lat_name="latitude", lon_name="longitude", atol=1e-6):
    if ds_a.sizes.get(lat_name, 0) != ds_b.sizes.get(lat_name, 0):
        return False
    if ds_a.sizes.get(lon_name, 0) != ds_b.sizes.get(lon_name, 0):
        return False
    if not np.allclose(ds_a[lat_name].values, ds_b[lat_name].values, atol=atol):
        return False
    if not np.allclose(ds_a[lon_name].values, ds_b[lon_name].values, atol=atol):
        return False
    return True


# ============================================================================
# Aurora level discovery
# ============================================================================

def discover_aurora_levels() -> np.ndarray:
    """Open the Aurora Zarr and return its pressure levels (hPa)."""
    print("  Discovering Aurora pressure levels …")
    ds = open_zarr_anonymous(MODEL_ZARRS["aurora"])
    level_dim = _detect_level_dim(ds)
    levels = ds[level_dim].values
    print(f"  Aurora levels ({level_dim}): {levels}")
    ds.close()
    return levels


# ============================================================================
# Single-date evaluation (conservation only, level-harmonized)
# ============================================================================

def _evaluate_one(
    prediction_zarr: str,
    era5_zarr: str,
    date_str: str,
    lead_label: str,
    lead_td: np.timedelta64,
    counter: int,
    total: int,
    aurora_levels: np.ndarray,
    verbose: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Conservation-only evaluation for one (date × lead_time), with level
    harmonization.  All models are sliced to aurora_levels before computing
    column integrals.
    """
    ds_pred_full = open_zarr_anonymous(prediction_zarr)
    ds_era5_full = open_zarr_anonymous(era5_zarr)
    ds_static = load_static_fields(ds_era5_full)
    z_sfc_name = _find_var(ds_static, ZSFC_NAMES)
    z_sfc = ds_static[z_sfc_name]
    area = get_grid_cell_area(ds_era5_full.isel(time=0, drop=True))

    lead_hours = int(lead_td / np.timedelta64(1, "h"))
    init_time = np.datetime64(date_str, "ns")

    rows: list[dict] = []
    ts_rows: list[dict] = []
    _n_levels: int | None = None
    _sp_method: str = "none"

    def _append(metric_name, model_val, era5_val=None):
        rows.append({
            "date": date_str,
            "lead_time_hours": lead_hours,
            "metric_name": metric_name,
            "model_value": model_val,
            "era5_value": era5_val,
            "n_levels": _n_levels,
            "sp_method": _sp_method,
        })

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    _log(f"  [{counter}/{total}] init={init_time}  lead={lead_label} ({lead_hours}h)")

    try:
        # ---- Drift window [6h → 240h], capped at MAX_TIMESTEPS ----
        td_start = np.timedelta64(6, "h")
        td_end = DRIFT_WINDOW_END.get(lead_hours, lead_td)

        ds_pred_init = ds_pred_full.sel(time=init_time, method="nearest")
        pred_td_dim = _detect_pred_td_dim(ds_pred_init) or "prediction_timedelta"

        ds_pred_window = ds_pred_init.sel({pred_td_dim: slice(td_start, td_end)})

        # Keep only needed variables
        _CONS_VARS = {
            "temperature", "geopotential",
            "u_component_of_wind", "v_component_of_wind",
            "specific_humidity", "q",
            "mean_sea_level_pressure", "msl",
            "surface_pressure", "sp",
            "2m_temperature", "t2m",
        }
        drop_vars = [v for v in ds_pred_window.data_vars if v.strip() not in _CONS_VARS]
        if drop_vars:
            ds_pred_window = ds_pred_window.drop_vars(drop_vars)

        if "time" in ds_pred_window.dims:
            ds_pred_window = ds_pred_window.isel(time=0)

        # Handle ensemble dimension
        for ens_dim in ("number", "realization", "sample"):
            if ens_dim in ds_pred_window.dims:
                n_ens = ds_pred_window.sizes[ens_dim]
                if n_ens > 10:
                    ds_pred_window = ds_pred_window.mean(dim=ens_dim)
                else:
                    ds_pred_window = ds_pred_window.isel({ens_dim: 0})

        avail_tds = ds_pred_window[pred_td_dim].values
        if len(avail_tds) < 2:
            _log(f"    [{counter}] ⚠ <2 timesteps — skipping")
            _append("ALL_METRICS_FAILED", None, None)
            return rows, ts_rows

        # Subsample to MAX_TIMESTEPS (evenly spaced)
        if len(avail_tds) > MAX_TIMESTEPS:
            indices = np.linspace(0, len(avail_tds) - 1, MAX_TIMESTEPS, dtype=int)
            avail_tds = avail_tds[indices]
            _log(f"    [{counter}] Subsampled to {MAX_TIMESTEPS} timesteps")

        _log(f"    [{counter}] Processing {len(avail_tds)} timesteps")

        pred_level_dim = _detect_level_dim(ds_pred_window)
        _n_levels = len(aurora_levels)

        # ---- LEVEL HARMONIZATION ----
        # Slice the prediction to Aurora's levels BEFORE loading
        pred_levels = ds_pred_window[pred_level_dim].values
        if not np.array_equal(np.sort(pred_levels), np.sort(aurora_levels)):
            _log(f"    [{counter}] Harmonizing levels: {pred_levels} → {aurora_levels}")
            ds_pred_window = ds_pred_window.sel(
                {pred_level_dim: aurora_levels}, method="nearest"
            )

        # Process in mini-batches
        batch_size = 5
        hours_pred = []
        dry_vals, water_vals, energy_vals = [], [], []

        # Interpolate static fields to model grid if needed
        ds_static_model = ds_static
        z_sfc_model = z_sfc
        area_pred = area

        for batch_start in range(0, len(avail_tds), batch_size):
            batch_tds = avail_tds[batch_start : batch_start + batch_size]
            ds_batch = ds_pred_window.sel({pred_td_dim: batch_tds}).load()

            for td_val in batch_tds:
                snap = ds_batch.sel({pred_td_dim: td_val})

                # Ensure (lat, lon) ordering
                if "latitude" in snap.dims and "longitude" in snap.dims:
                    sdims = list(snap.dims)
                    si_lon = sdims.index("longitude")
                    si_lat = sdims.index("latitude")
                    if si_lon < si_lat:
                        sdims[si_lon], sdims[si_lat] = sdims[si_lat], sdims[si_lon]
                        snap = snap.transpose(*sdims)

                # On first iteration, check grid alignment
                if batch_start == 0 and td_val is batch_tds[0]:
                    model_grid_matches = _grids_match(snap, ds_era5_full)
                    if not model_grid_matches:
                        _log(f"    [{counter}] Model grid differs — interpolating statics")
                        interp_coords = {}
                        if "latitude" in snap.dims and "latitude" in ds_static.dims:
                            interp_coords["latitude"] = snap["latitude"]
                        if "longitude" in snap.dims and "longitude" in ds_static.dims:
                            interp_coords["longitude"] = snap["longitude"]
                        if interp_coords:
                            ds_static_model = ds_static.interp(interp_coords, method="nearest")
                            # Ensure (lat,lon) order on statics too
                            if "latitude" in ds_static_model.dims and "longitude" in ds_static_model.dims:
                                sd = list(ds_static_model.dims)
                                si = sd.index("longitude")
                                sl = sd.index("latitude")
                                if si < sl:
                                    sd[si], sd[sl] = sd[sl], sd[si]
                                    ds_static_model = ds_static_model.transpose(*sd)
                            z_sfc_name_l = _find_var(ds_static_model, ZSFC_NAMES)
                            z_sfc_model = ds_static_model[z_sfc_name_l]
                        area_pred = get_grid_cell_area(snap)

                # Derive surface pressure
                try:
                    ps_snap = _get_ps(snap, ds_static_model, level_dim=pred_level_dim)
                except Exception:
                    ps_snap = None

                _sp_method = (ps_snap.attrs.get("derivation_method", "unknown")
                              if ps_snap is not None else "none")

                # Conservation scalars (at harmonized levels)
                if ps_snap is not None:
                    dry, water, energy = compute_conservation_scalars(
                        snap, ps_snap, area_pred, z_sfc=z_sfc_model,
                        level_dim=pred_level_dim,
                    )
                else:
                    dry, water, energy = float("nan"), float("nan"), float("nan")

                h = float(td_val / np.timedelta64(1, "h"))
                hours_pred.append(h)
                dry_vals.append(dry)
                water_vals.append(water)
                energy_vals.append(energy)

                ts_rows.append({
                    "date": date_str,
                    "forecast_hour": h,
                    "dry_mass_Eg": dry,
                    "water_mass_kg": water,
                    "total_energy_J": energy,
                })

                del snap, ps_snap
            del ds_batch

        hours_pred = np.array(hours_pred)
        dry_vals = np.array(dry_vals)
        water_vals = np.array(water_vals)
        energy_vals = np.array(energy_vals)

        _log(f"    [{counter}] Drift: {len(hours_pred)} steps "
             f"[{hours_pred[0]:.0f}h–{hours_pred[-1]:.0f}h]")

        # ---- Compute drift (model-only, without ERA5 baseline) ----
        slope_dry = compute_drift_slope(hours_pred, dry_vals)
        dry_ref = float(dry_vals[0])
        slope_water = compute_drift_slope(hours_pred, water_vals)
        water_ref = float(water_vals[0])
        slope_energy = compute_drift_slope(hours_pred, energy_vals)
        energy_ref = float(energy_vals[0])

        drift = {
            "dry_mass_drift_pct_per_day": (
                (slope_dry / dry_ref) * 100.0
                if dry_ref != 0 and np.isfinite(slope_dry) else float("nan")
            ),
            "water_mass_drift_pct_per_day": (
                (slope_water / water_ref) * 100.0
                if water_ref != 0 and np.isfinite(slope_water) else float("nan")
            ),
            "total_energy_drift_pct_per_day": (
                (slope_energy / energy_ref) * 100.0
                if energy_ref != 0 and np.isfinite(slope_energy) else float("nan")
            ),
        }

        for metric_name, val in drift.items():
            _append(metric_name, val)

        _log(f"    [{counter}] ✓ Drift: "
             f"dry={drift['dry_mass_drift_pct_per_day']:+.6g}%/day  "
             f"water={drift['water_mass_drift_pct_per_day']:+.6g}%/day  "
             f"energy={drift['total_energy_drift_pct_per_day']:+.6g}%/day")

    except Exception as exc:
        _log(f"    [{counter}] ⚠ Evaluation failed: {exc}")
        import traceback
        traceback.print_exc()
        _append("ALL_METRICS_FAILED", None, None)

    return rows, ts_rows


# ============================================================================
# Batch runner
# ============================================================================

def run_evaluation(
    dates: list[str],
    model_name: str,
    aurora_levels: np.ndarray,
    output_csv: Path,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = True,
) -> pd.DataFrame:
    prediction_zarr = MODEL_ZARRS[model_name]

    work_items = [
        (date_str, ll, ltd)
        for date_str in dates
        for ll, ltd in LEAD_TIMES
    ]
    n = len(work_items)

    if verbose:
        print("\n" + "=" * 70)
        print("  LEVEL ABLATION — Conservation Metrics (3 levels)")
        print("=" * 70)
        print(f"  Model      : {model_name}")
        print(f"  Prediction : {prediction_zarr}")
        print(f"  ERA5       : {ERA5_ZARR}")
        print(f"  Levels     : {aurora_levels}")
        print(f"  Dates      : {len(dates)}")
        print(f"  Timesteps  : ≤{MAX_TIMESTEPS} per date")
        print(f"  Workers    : {workers}")
        print(f"  Output     : {output_csv}")
        print("=" * 70)

    TASK_TIMEOUT = 600

    all_rows: list[dict] = []
    all_ts_rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for idx, (date_str, ll, ltd) in enumerate(work_items, 1):
            fut = pool.submit(
                _evaluate_one,
                prediction_zarr, ERA5_ZARR,
                date_str, ll, ltd,
                idx, n,
                aurora_levels,
                verbose,
            )
            futures[fut] = (idx, date_str, ll)

        for fut in as_completed(futures):
            idx, date_str, ll = futures[fut]
            try:
                summary_rows, ts_rows = fut.result(timeout=TASK_TIMEOUT)
                all_rows.extend(summary_rows)
                all_ts_rows.extend(ts_rows)
            except TimeoutError:
                if verbose:
                    print(f"  ⚠ Task {idx} ({date_str} {ll}) timed out", flush=True)
            except Exception as exc:
                if verbose:
                    print(f"  ⚠ Worker exception (task {idx}): {exc}", flush=True)

    all_rows.sort(key=lambda r: (r["date"], r["lead_time_hours"], r["metric_name"]))

    df = pd.DataFrame(all_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"\n  ✓ Results → {output_csv}  ({len(df)} rows)")

    if all_ts_rows:
        ts_csv = output_csv.parent / f"physics_timeseries_{model_name}_2022_3levels.csv"
        df_ts = pd.DataFrame(all_ts_rows)
        df_ts.sort_values(["date", "forecast_hour"], inplace=True)
        df_ts.to_csv(ts_csv, index=False)
        if verbose:
            print(f"  ✓ Time-series → {ts_csv}  ({len(df_ts)} rows)")

    return df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Level ablation study: 3-level conservation metrics"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_ZARRS.keys()),
        help="Model to evaluate (aurora, pangu, hres)",
    )
    parser.add_argument(
        "--dates", nargs="+", default=None,
        help="Override dates (e.g. 2022-01-01 2022-01-02). Default: full year 2022.",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    # Resolve dates
    if args.dates:
        dates = args.dates
    else:
        dates = []
        for m in range(1, 13):
            n_days = calendar.monthrange(YEAR, m)[1]
            for d in range(1, n_days + 1):
                dates.append(f"{YEAR}-{m:02d}-{d:02d}")

    # Discover Aurora's levels (once, before forking workers)
    aurora_levels = discover_aurora_levels()

    output_csv = OUTPUT_DIR / f"physics_evaluation_{args.model}_2022_3levels.csv"

    run_evaluation(
        dates=dates,
        model_name=args.model,
        aurora_levels=aurora_levels,
        output_csv=output_csv,
        workers=args.workers,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
