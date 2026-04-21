#!/usr/bin/env python
"""
Physics Evaluation Runner — WeatherBench 2 Zarr Streaming

Streams Aurora predictions and ERA5 ground-truth data from public WB2
Zarr buckets, computes all physics metrics at multiple forecast horizons
(6 h, Day 5, Day 10), and saves a single long-format CSV.

Output Format (melted long)
---------------------------
  date | lead_time_hours | metric_name | model_value | era5_value | n_levels | sp_method

Data Sources
------------
  Aurora : gs://weatherbench2/datasets/aurora/2022-1440x721.zarr
  ERA5   : gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr

Usage
-----
  python run_physics_evaluation.py --year 2022
  python run_physics_evaluation.py --year 2022 --workers 16
  python run_physics_evaluation.py --dates 2022-01-01 2022-01-02
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

# Companion library (same directory)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from physics_metrics import (
    compute_conservation_scalars,
    compute_drift_percentages,
    compute_drift_slope,
    compute_pure_tcwv,
    _find_effective_resolution,
    compute_geostrophic_imbalance,
    compute_hydrostatic_imbalance,
    compute_ke_spectrum,
    compute_spectral_scores,
    derive_surface_pressure,
    get_grid_cell_area,
    _find_var,
    _detect_level_dim,
    _detect_pred_td_dim,
    SP_NAMES,
    MSL_NAMES,
    T_NAMES,
    T2M_NAMES,
    ZSFC_NAMES,
    Q_NAMES,
    U_NAMES,
    V_NAMES,
    PHI_NAMES,
)

# Suppress xarray FutureWarning about timedelta decoding
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*prediction_timedelta.*")


# ============================================================================
# Configuration
# ============================================================================

AURORA_ZARR = "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr"
ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
ERA5_DAILY_ZARR = "gs://weatherbench2/datasets/era5_daily/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr"

# IFS HRES t=0 (analysis) as alternative reference
IFS_T0_ZARR = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
IFS_T0_LOWRES_ZARR = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-512x256_equiangular_conservative.zarr"

OUTPUT_DIR = Path.home() / "aurora_thesis" / "thesis" / "benchmark" / "results"

# Target lead times: (label, timedelta)
# Using 12h as the first lead time to align with NeuralGCM (which lacks 6h step)
LEAD_TIMES: list[tuple[str, np.timedelta64]] = [
    ("12h",  np.timedelta64(12,  "h")),
    ("5d",   np.timedelta64(120, "h")),
    ("10d",  np.timedelta64(240, "h")),
]

# Map each target lead_td to the END of the drift-regression window.
# Start is always 12 h (to align with NeuralGCM).
DRIFT_WINDOW_END: dict[int, np.timedelta64] = {
    12:  np.timedelta64(24,  "h"),   # 12h target → window 12h–24h
    120: np.timedelta64(120, "h"),   # 5d  target → window 12h–120h
    240: np.timedelta64(240, "h"),   # 10d target → window 12h–240h
}

DEFAULT_WORKERS = 8


# ============================================================================
# Zarr I/O
# ============================================================================

def open_zarr_anonymous(url: str) -> xr.Dataset:
    """Open a public GCS Zarr store without authentication."""
    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    # Sanitise variable AND dimension names (some Zarrs have trailing whitespace)
    rename = {}
    for v in ds.data_vars:
        if v != v.strip():
            rename[v] = v.strip()
    for d in ds.dims:
        if d != d.strip():
            rename[d] = d.strip()
    # Normalise short lat/lon dim names (e.g. GraphCast uses "lat"/"lon")
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def load_static_fields(ds_era5: xr.Dataset) -> xr.Dataset:
    """
    Extract static fields (z_sfc, land-sea mask) from ERA5 at time=0.
    """
    static_vars = {}

    def _extract_static(ds, name):
        var = ds[name]
        if "time" in var.dims:
            var = var.isel(time=0, drop=True)
        return var

    for name in ("geopotential_at_surface", "z_sfc", "orography"):
        if name in ds_era5.data_vars:
            static_vars[name] = _extract_static(ds_era5, name)
            break

    for name in ("land_sea_mask", "lsm"):
        if name in ds_era5.data_vars:
            static_vars[name] = _extract_static(ds_era5, name)
            break

    if not static_vars:
        raise ValueError(
            f"No static fields found. "
            f"Available: {list(ds_era5.data_vars)[:20]}"
        )

    return xr.Dataset(static_vars)


def _get_ps(
    ds: xr.Dataset,
    ds_static: xr.Dataset,
    level_dim: str = "level",
    t2m_mean: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Return surface pressure, deriving from MSL if needed,
    with fallback to bottom-level barometric estimation."""
    # 1. Direct surface pressure variable
    sp_name = _find_var(ds, SP_NAMES)
    if sp_name is not None:
        sp = ds[sp_name]
        sp.attrs["derivation_method"] = "direct_sp"
        return sp

    # 2. Hypsometric from MSL + T2m
    try:
        sp = derive_surface_pressure(ds, ds_static, t2m_mean=t2m_mean)
        if t2m_mean is not None:
            sp.attrs["derivation_method"] = "hypsometric_msl_daily_t2m"
        else:
            sp.attrs["derivation_method"] = "hypsometric_msl"
        return sp
    except (ValueError, KeyError):
        pass

    raise ValueError(
        f"Cannot derive surface pressure: no SP variable and "
        f"hypsometric derivation from MSL failed. "
        f"Available vars: {list(ds.data_vars)}"
    )


# ============================================================================
# Grid Alignment
# ============================================================================


def _grids_match(
    ds_a: xr.Dataset,
    ds_b: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    atol: float = 1e-3,
) -> bool:
    """Return True if two datasets share the same lat/lon grid (order-insensitive)."""
    if ds_a.sizes.get(lat_name, 0) != ds_b.sizes.get(lat_name, 0):
        return False
    if ds_a.sizes.get(lon_name, 0) != ds_b.sizes.get(lon_name, 0):
        return False
    lat_a = np.sort(ds_a[lat_name].values)
    lat_b = np.sort(ds_b[lat_name].values)
    if not np.allclose(lat_a, lat_b, atol=atol):
        return False
    lon_a = np.sort(ds_a[lon_name].values)
    lon_b = np.sort(ds_b[lon_name].values)
    if not np.allclose(lon_a, lon_b, atol=atol):
        return False
    return True


def _align_era5_to_aurora(
    ds_era5: xr.Dataset,
    ds_aurora: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """
    Align ERA5 grid to match Aurora for spectral comparison.

    Only two cases are allowed:
    1. Exact shape match — grids are identical (or close enough).
    2. ERA5 has exactly 1 extra latitude row (e.g. 721 vs 720) — drop the
       pole row so the grids match.

    No interpolation is performed.  Any other mismatch raises an error.
    """
    n_era5 = ds_era5.sizes.get(lat_name, 0)
    n_aurora = ds_aurora.sizes.get(lat_name, 0)
    n_lon_era5 = ds_era5.sizes.get(lon_name, 0)
    n_lon_aurora = ds_aurora.sizes.get(lon_name, 0)

    if n_lon_era5 != n_lon_aurora:
        raise ValueError(
            f"Longitude grid mismatch: ERA5 has {n_lon_era5}, "
            f"prediction has {n_lon_aurora}. Cannot align without interpolation."
        )

    if n_era5 == n_aurora:
        result = ds_era5
    elif n_era5 == n_aurora + 1:
        # ERA5 has 1 extra lat row (pole) — drop it
        lats = ds_era5[lat_name].values
        if lats[0] > lats[-1]:  # Descending (N→S): drop last row (south pole)
            result = ds_era5.isel({lat_name: slice(0, -1)})
        else:                   # Ascending (S→N): drop first row (south pole)
            result = ds_era5.isel({lat_name: slice(1, None)})
    else:
        raise ValueError(
            f"Latitude grid mismatch: ERA5 has {n_era5} rows, "
            f"prediction has {n_aurora}. Only exact match or 1-row "
            f"pole difference is supported (no interpolation)."
        )

    # Reassign latitude coordinates from prediction to ensure exact alignment
    result = result.assign_coords({lat_name: ds_aurora[lat_name].values})

    # Ensure (latitude, longitude) dimension order for spectral analysis
    if lat_name in result.dims and lon_name in result.dims:
        dims_list = list(result.dims)
        idx_lon = dims_list.index(lon_name)
        idx_lat = dims_list.index(lat_name)
        if idx_lon < idx_lat:
            dims_list[idx_lon], dims_list[idx_lat] = (
                dims_list[idx_lat], dims_list[idx_lon]
            )
            result = result.transpose(*dims_list)

    return result


# ============================================================================
# Date Resolution
# ============================================================================

def _resolve_dates(args) -> list[str]:
    """Build a list of ISO date strings from CLI arguments."""
    if args.dates:
        return args.dates
    if args.month:
        year, month = args.month.split("-")
        n_days = calendar.monthrange(int(year), int(month))[1]
        return [f"{year}-{month}-{d:02d}" for d in range(1, n_days + 1)]
    year = args.year
    dates = []
    for m in range(1, 13):
        n_days = calendar.monthrange(year, m)[1]
        for d in range(1, n_days + 1):
            dates.append(f"{year}-{m:02d}-{d:02d}")
    return dates


# ============================================================================
# Single-Slice Evaluation → melted rows
# ============================================================================

def _evaluate_one(
    aurora_zarr_path: str,
    era5_zarr_path: str,
    era5_daily_zarr_path: str,
    date_str: str,
    lead_label: str,
    lead_td: np.timedelta64,
    counter: int,
    total: int,
    mode: str,
    verbose: bool,
    static_zarr_path: str | None = None,
    model_name: str = "aurora",
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Fetch, load, and evaluate one (date × lead_time) combination.
    Runs in a separate process, so it opens its own Zarr connections.

    Returns (summary_rows, ts_rows, spectrum_rows)
    """
    # Open datasets inside the worker process to avoid pickling issues
    ds_aurora_full = None
    if mode in ("joint", "prediction", "aurora"):
        ds_aurora_full = open_zarr_anonymous(aurora_zarr_path)
    
    ds_era5_full = open_zarr_anonymous(era5_zarr_path)
    ds_era5_daily = open_zarr_anonymous(era5_daily_zarr_path)
    
    # Load static fields — from a dedicated static Zarr if provided
    # (needed when era5_zarr is e.g. HRES-T0 which lacks static fields)
    if static_zarr_path:
        ds_static_src = open_zarr_anonymous(static_zarr_path)
    else:
        ds_static_src = ds_era5_full
    ds_static = load_static_fields(ds_static_src)

    # Ensure static fields are in (latitude, longitude) order
    for var_name in list(ds_static.data_vars):
        v = ds_static[var_name]
        if "latitude" in v.dims and "longitude" in v.dims:
            if list(v.dims).index("longitude") < list(v.dims).index("latitude"):
                ds_static[var_name] = v.transpose("latitude", "longitude")

    z_sfc_name = _find_var(ds_static, ZSFC_NAMES)
    z_sfc = ds_static[z_sfc_name]
    
    # Grid cell area
    area = get_grid_cell_area(ds_era5_full.isel(time=0, drop=True))

    lead_hours = int(lead_td / np.timedelta64(1, "h"))
    # Cast to ns resolution so xarray .sel() matches the ERA5 Zarr time coord
    init_time = np.datetime64(date_str, "ns")
    valid_time = init_time + lead_td

    _t2m_daily_cache = {}
    def _get_daily_t2m(vt: np.datetime64):
        try:
            day_time = vt.astype('datetime64[D]')
            if day_time not in _t2m_daily_cache:
                tname = _find_var(ds_era5_daily, T2M_NAMES)
                if tname:
                    t_day = ds_era5_daily[tname].sel(time=day_time, method="nearest").load()
                    t_day = t_day.interp(
                        latitude=ds_era5_full.latitude,
                        longitude=ds_era5_full.longitude,
                        method="linear"
                    )
                    _t2m_daily_cache[day_time] = t_day
                else:
                    _t2m_daily_cache[day_time] = None
            return _t2m_daily_cache[day_time]
        except Exception as exc:
            if verbose:
                 print(f"  [{counter}/{total}] ⚠ Could not fetch daily mean T2m: {exc}")
            return None

    # Get daily t2m for the valid_time (for the main snapshot)
    t2m_mean = _get_daily_t2m(valid_time)

    rows: list[dict] = []

    # Will be set after model data is loaded
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

    _log(
        f"  [{counter}/{total}] "
        f"init={init_time}  lead={lead_label} ({lead_hours}h)  "
        f"valid={valid_time}"
    )

    try:
        # ---- Fetch and load Aurora slice ----
        ds_aurora_t = None
        ps_aurora = None
        area_aurora = None
        _lead_td_mismatch = False

        if ds_aurora_full is not None:
            # Normalise dimension names (some zarrs have trailing whitespace)
            dim_rename = {d: d.strip() for d in ds_aurora_full.dims if d != d.strip()}
            if dim_rename:
                ds_aurora_full = ds_aurora_full.rename(dim_rename)

            ds_aurora_t = ds_aurora_full.sel(time=init_time)

            # Auto-detect prediction_timedelta dimension name
            pred_td_dim = _detect_pred_td_dim(ds_aurora_t)
            _lead_td_mismatch = False  # track if nearest != requested
            if pred_td_dim is not None and pred_td_dim in ds_aurora_t.dims:
                ds_aurora_t = ds_aurora_t.sel(
                    {pred_td_dim: lead_td}, method="nearest"
                )
                # Check if the actual selected timedelta matches requested
                actual_td = ds_aurora_t.coords.get(pred_td_dim)
                if actual_td is not None:
                    actual_td_val = actual_td.values
                    if isinstance(actual_td_val, np.timedelta64) and actual_td_val != lead_td:
                        _log(f"    [{counter}] ⚠ Requested lead={lead_td}, "
                             f"nearest available={actual_td_val} — skipping metrics for this lead time")
                        _lead_td_mismatch = True

            # Drop variables not needed for physics metrics to reduce I/O.
            # Critical for datasets with large chunks (e.g. FuXi chunks
            # span all 60 prediction timedeltas, so each var is ~120 MB).
            _NEEDED_VARS = set()
            for names in (T_NAMES, PHI_NAMES, U_NAMES, V_NAMES,
                          Q_NAMES, MSL_NAMES, SP_NAMES, T2M_NAMES, ZSFC_NAMES):
                _NEEDED_VARS.update(names)
            drop_vars = [v for v in ds_aurora_t.data_vars
                         if v.strip() not in _NEEDED_VARS]
            if drop_vars:
                _log(f"    [{counter}] Dropping {len(drop_vars)} unneeded vars: {drop_vars}")
                ds_aurora_t = ds_aurora_t.drop_vars(drop_vars)

            if "time" in ds_aurora_t.dims:
                 ds_aurora_t = ds_aurora_t.isel(time=0)

            # Ensure spatial dimensions are in standard (latitude, longitude) order
            if "latitude" in ds_aurora_t.dims and "longitude" in ds_aurora_t.dims:
                dims_list = list(ds_aurora_t.dims)
                idx_lon = dims_list.index("longitude")
                idx_lat = dims_list.index("latitude")
                if idx_lon < idx_lat:
                    _log(f"    [{counter}] Transposing spatial dims from (lon, lat) to (lat, lon)")
                    dims_list[idx_lon], dims_list[idx_lat] = dims_list[idx_lat], dims_list[idx_lon]
                    ds_aurora_t = ds_aurora_t.transpose(*dims_list)

            ds_aurora_t = ds_aurora_t.load()

            # ---- Static fields must match model grid shape ----
            # For static fields (orography, land-sea mask) that vary slowly,
            # interpolation to the model grid is acceptable when grids differ.
            lat_name = "latitude"
            lon_name = "longitude"
            model_grid_matches_era5 = _grids_match(ds_aurora_t, ds_era5_full)

            ds_static_model = ds_static
            z_sfc_model = z_sfc
            if not model_grid_matches_era5:
                # Interpolate static fields to the model grid (nearest neighbour)
                interp_coords = {
                    lat_name: ds_aurora_t[lat_name].values,
                    lon_name: ds_aurora_t[lon_name].values,
                }
                ds_static_model = ds_static.interp(interp_coords, method="nearest")
                z_sfc_name = _find_var(ds_static_model, ZSFC_NAMES)
                if z_sfc_name is not None:
                    z_sfc_model = ds_static_model[z_sfc_name]
                _log(f"    [{counter}] Model grid differs from ERA5 — interpolating static fields")

            # Only derive surface pressure if humidity is available,
            # since ps is only used for mass/water/energy metrics (which
            # require q).  Skipping saves an expensive hypsometric calc.
            has_q = _find_var(ds_aurora_t, Q_NAMES) is not None
            # Auto-detect the level dimension name for this dataset
            aurora_level_dim = _detect_level_dim(ds_aurora_t)
            
            # Check if model has SP or MSL (needed to derive surface pressure independently)
            # Note: 3D geopotential alone is NOT sufficient - geopotential_interpolation
            # still requires ERA5's surface geopotential (z_sfc), so models with only
            # 3D geopotential but no SP/MSL should use ERA5 sp directly.
            _has_sp = _find_var(ds_aurora_t, SP_NAMES) is not None
            _has_msl = _find_var(ds_aurora_t, MSL_NAMES) is not None
            _model_can_derive_sp = _has_sp or _has_msl
            
            # Check if model has P-E for water budget calculation
            _has_pe = _find_var(ds_aurora_t, ("P_minus_E_cumulative",)) is not None
            
            # If model lacks SP/MSL, we'll use ERA5 sp instead
            _use_era5_sp = not _model_can_derive_sp
            
            if _use_era5_sp:
                _log(f"    [{counter}] Model lacks SP/MSL — will use ERA5 surface pressure")
            
            if has_q and not _use_era5_sp:
                try:
                    ps_aurora = _get_ps(ds_aurora_t, ds_static_model, level_dim=aurora_level_dim, t2m_mean=t2m_mean)
                except Exception as exc:
                    _log(f"    [{counter}] ⚠ Could not derive surface pressure: {exc}")
                    _log(f"    [{counter}]   → mass/water/energy metrics will be NaN")
                    ps_aurora = None
            else:
                ps_aurora = None

            # Track metadata for output CSV
            if aurora_level_dim in ds_aurora_t.dims:
                _n_levels = ds_aurora_t.sizes[aurora_level_dim]
            _sp_method = (
                ps_aurora.attrs.get("derivation_method", "unknown")
                if ps_aurora is not None
                else "none"
            )

             # Compute area matching the model's own grid.
            # If model and ERA5 share the same lat grid (±1 point AND
            # matching coordinate values), reuse ERA5 area.  Otherwise
            # compute area from the model grid directly.
            area_aurora = area
            if lat_name in ds_aurora_t.dims:
                n_aurora = ds_aurora_t.sizes[lat_name]
                n_area = area.sizes[lat_name]
                if n_aurora == n_area and model_grid_matches_era5:
                    pass  # grids truly match
                elif abs(n_aurora - n_area) <= 1 and model_grid_matches_era5:
                    # Off by 1 (e.g. 720 vs 721) — simple trim
                    n = min(n_aurora, n_area)
                    area_aurora = area.isel({lat_name: slice(0, n)})
                    area_aurora = area_aurora.assign_coords(
                        {lat_name: ds_aurora_t[lat_name].values[:n]}
                    )
                else:
                    # Different grid — compute area from model grid
                    _log(f"    [{counter}] Computing area from model grid ({n_aurora} lats vs ERA5 {n_area})")
                    area_aurora = get_grid_cell_area(ds_aurora_t)

        # ---- Fetch and load ERA5 slice at valid time ----
        ds_era5_t = None
        ps_era5 = None
        
        if mode in ("prediction", "aurora"):
             
             ds_era5_t = ds_era5_full.sel(time=valid_time)
             if "time" in ds_era5_t.dims:
                 ds_era5_t = ds_era5_t.isel(time=0)
             
             ds_era5_t = ds_era5_t.load()
             
             # If model lacks SP/MSL/geopotential: get ERA5 sp for conservation metrics
             if _use_era5_sp:
                 era5_level_dim = _detect_level_dim(ds_era5_t)
                 ps_era5 = _get_ps(ds_era5_t, ds_static, level_dim=era5_level_dim, t2m_mean=t2m_mean)
             else:
                 # ps_era5 is not needed in prediction-only mode for models with SP/MSL
                 ps_era5 = None 
        
        else:
            # joint / era5 mode: Load full 3D ERA5 for intrinsic metrics
            ds_era5_t = ds_era5_full.sel(time=valid_time)

            # Squeeze time dimension if present
            if "time" in ds_era5_t.dims:
                ds_era5_t = ds_era5_t.isel(time=0)

            ds_era5_t = ds_era5_t.load()
            
            # Surface pressure (needed for intrinsic metrics)
            era5_level_dim = _detect_level_dim(ds_era5_t)
            ps_era5 = _get_ps(ds_era5_t, ds_static, level_dim=era5_level_dim, t2m_mean=t2m_mean)
        
        # If model lacks SP/MSL/geopotential: use ERA5 sp for model conservation metrics
        if _use_era5_sp and ps_era5 is not None and has_q:
            # Interpolate ERA5 sp to model grid if needed
            if not model_grid_matches_era5 and ds_aurora_t is not None:
                ps_aurora = ps_era5.interp(
                    latitude=ds_aurora_t.latitude,
                    longitude=ds_aurora_t.longitude,
                    method="linear"
                )
            else:
                ps_aurora = ps_era5
            ps_aurora.attrs["derivation_method"] = "era5_sp"
            _sp_method = "era5_sp"
            _log(f"    [{counter}] Using ERA5 surface pressure for conservation metrics")

    except Exception as exc:
        # Data loading failed entirely — nothing we can do
        _log(f"    [{counter}] ⚠ Data loading failed: {exc}")
        rows.append({
            "date": date_str,
            "lead_time_hours": lead_hours,
            "metric_name": "ERROR",
            "model_value": None,
            "era5_value": None,
            "n_levels": _n_levels,
            "sp_method": _sp_method,
        })
        return rows, [], []

    # ================================================================
    # Conservation / Stability drift metrics  +  time-series output
    # ================================================================
    # Compute the LINEAR TREND (slope per day) of mass/energy over a
    # time window [6 h … end_time].  end_time is looked up from
    # DRIFT_WINDOW_END for the current lead_td.  While iterating we
    # also collect hydrostatic_rmse and geostrophic_rmse at each step
    # for the secondary time-series CSV.
    #
    # PERFORMANCE RULES:
    #  - Evaluate every available timestep (no subsampling) for
    #    high-resolution diurnal-cycle time-series plots
    #  - Process in mini-batches of 5 to align with Zarr chunks
    #  - Never .load() the full 4-D window
    # ----------------------------------------------------------------

    ts_rows: list[dict] = []       # time-series rows for secondary CSV
    spectrum_rows: list[dict] = []  # KE spectrum rows for spectrum CSV

    if mode in ("joint", "prediction", "aurora") and ds_aurora_full is not None and not _lead_td_mismatch:
        try:
            td_start = np.timedelta64(12, "h")  # Start at 12h to align with NeuralGCM
            td_end = DRIFT_WINDOW_END.get(lead_hours, lead_td)

            # --- Aurora time-series ---
            ds_pred_init = ds_aurora_full.sel(time=init_time)
            pred_td_dim = _detect_pred_td_dim(ds_pred_init) or "prediction_timedelta"

            # Lazy selection of timedelta window [12 h … end_time]
            ds_pred_window = ds_pred_init.sel(
                {pred_td_dim: slice(td_start, td_end)}
            )

            # Drop unneeded variables lazily (before any .load())
            _CONS_VARS = {
                "temperature", "geopotential",
                "u_component_of_wind", "v_component_of_wind",
                "specific_humidity", "q",
                "mean_sea_level_pressure", "msl",
                "surface_pressure", "sp",
                "2m_temperature", "t2m",
                "P_minus_E_cumulative",
            }
            drop_vars = [v for v in ds_pred_window.data_vars
                         if v.strip() not in _CONS_VARS]
            if drop_vars:
                ds_pred_window = ds_pred_window.drop_vars(drop_vars)

            # Squeeze time if still present as a dimension
            if "time" in ds_pred_window.dims:
                ds_pred_window = ds_pred_window.isel(time=0)

            # Handle ensemble — same logic as for the single snapshot
            for ens_dim in ("number", "realization", "sample"):
                if ens_dim in ds_pred_window.dims:
                    n_ens = ds_pred_window.sizes[ens_dim]
                    if n_ens > 10:
                        ds_pred_window = ds_pred_window.mean(dim=ens_dim)
                    else:
                        ds_pred_window = ds_pred_window.isel({ens_dim: 0})

            # Keep every available timestep (no subsampling) so that
            # diurnal cycles are fully resolved in the time-series output.

            avail_tds = ds_pred_window[pred_td_dim].values
            if len(avail_tds) < 2:
                _log(f"    [{counter}] ⚠ Drift: <2 time steps in window — skipping")
                raise ValueError("Need ≥2 time steps for drift regression")

            hours_aurora = []
            dry_vals, water_vals, energy_vals = [], [], []
            pe_vals = []
            hydro_vals, geo_vals = [], []
            aurora_level_dim_d = _detect_level_dim(ds_pred_window)

            for td_val in avail_tds:
                snap = ds_pred_window.sel({pred_td_dim: td_val}).load()

                # Ensure (lat, lon) ordering
                if "latitude" in snap.dims and "longitude" in snap.dims:
                    sdims = list(snap.dims)
                    si_lon = sdims.index("longitude")
                    si_lat = sdims.index("latitude")
                    if si_lon < si_lat:
                        sdims[si_lon], sdims[si_lat] = sdims[si_lat], sdims[si_lon]
                        snap = snap.transpose(*sdims)

                # Compute the valid time for this timestep: init_time + td_val
                # Note: snap.time.values returns init_time, not valid_time
                snap_valid_time = init_time + td_val

                # Compute t2m_mean for this snapshot date (for PS derivation)
                snap_t2m_mean = None
                try:
                    if isinstance(snap_valid_time, np.datetime64):
                        snap_t2m_mean = _get_daily_t2m(snap_valid_time)
                except Exception:
                    pass

                # For NeuralGCM: use ERA5 sp, compute both standard metrics AND P-E budget
                if _use_era5_sp:
                    # Model lacks SP/MSL/geopotential: get ERA5 sp for this timestep
                    try:
                        era5_snap_t = ds_era5_full.sel(time=snap_valid_time, method="nearest")
                        if "time" in era5_snap_t.dims:
                            era5_snap_t = era5_snap_t.isel(time=0)
                        era5_snap_t = era5_snap_t.load()
                        era5_ld_snap = _detect_level_dim(era5_snap_t)
                        ps_snap_era5 = _get_ps(era5_snap_t, ds_static, level_dim=era5_ld_snap, t2m_mean=snap_t2m_mean)
                        
                        # Interpolate to model grid if needed
                        if not model_grid_matches_era5:
                            ps_snap = ps_snap_era5.interp(
                                latitude=snap.latitude,
                                longitude=snap.longitude,
                                method="linear"
                            )
                        else:
                            ps_snap = ps_snap_era5
                        
                        # Ensure ps_snap has same spatial dimension order as snap (lat, lon)
                        if "latitude" in ps_snap.dims and "longitude" in ps_snap.dims:
                            ps_dims = list(ps_snap.dims)
                            if ps_dims.index("longitude") < ps_dims.index("latitude"):
                                ps_snap = ps_snap.transpose("latitude", "longitude")
                        
                        ps_snap.attrs["derivation_method"] = "era5_sp"
                        
                        # Compute conservation metrics using ERA5 sp
                        dry, water, energy = compute_conservation_scalars(
                            snap, ps_snap, area_aurora, z_sfc=z_sfc_model,
                            level_dim=aurora_level_dim_d,
                        )
                        step_sp_method = "era5_sp"
                        
                        # Also compute P-E budget if available
                        pe_var = _find_var(snap, ("P_minus_E_cumulative",))
                        if pe_var:
                            pe_step = float((area_aurora * snap[pe_var]).sum())
                        else:
                            pe_step = float("nan")
                            
                    except Exception as exc:
                        # Fallback if ERA5 sp fetch fails
                        ps_snap = None
                        dry = float("nan")
                        energy = float("nan")
                        water = float("nan")
                        pe_step = float("nan")
                        step_sp_method = "failed"
                        if verbose:
                            print(f"    [{counter}] ⚠ ERA5 sp fetch failed: {exc}")
                else:
                    # Standard path for models with SP/MSL/geopotential
                    try:
                        ps_snap = _get_ps(snap, ds_static_model, level_dim=aurora_level_dim_d, t2m_mean=snap_t2m_mean)
                        dry, water, energy = compute_conservation_scalars(
                            snap, ps_snap, area_aurora, z_sfc=z_sfc_model,
                            level_dim=aurora_level_dim_d,
                        )
                        step_sp_method = ps_snap.attrs.get("derivation_method", "unknown")
                        pe_step = float("nan")  # Track A doesn't use P-E
                    except Exception:
                        # Fallback to pure pressure levels and P-E budget
                        ps_snap = None
                        dry = float("nan")    # Cannot compute dry mass without ps
                        energy = float("nan") # Cannot compute energy without ps

                        q_var = _find_var(snap, Q_NAMES)
                        pe_var = _find_var(snap, ("P_minus_E_cumulative",))

                        if q_var and pe_var:
                            pure_tcwv = compute_pure_tcwv(snap, q_name=q_var, level_dim=aurora_level_dim_d)
                            water = float((area_aurora * pure_tcwv).sum())
                            pe_step = float((area_aurora * snap[pe_var]).sum())
                            step_sp_method = "fixed_1000hPa_pure_levels"
                        else:
                            water = float("nan")
                            pe_step = float("nan")
                            step_sp_method = "failed"

                # Balance RMSEs (hydrostatic + geostrophic)
                try:
                    hydro = compute_hydrostatic_imbalance(
                        snap, area_aurora, level_dim=aurora_level_dim_d,
                    )
                except Exception:
                    hydro = float("nan")
                try:
                    geo = compute_geostrophic_imbalance(
                        snap, area_aurora, level_dim=aurora_level_dim_d,
                    )
                except Exception:
                    geo = float("nan")

                h = float(td_val / np.timedelta64(1, "h"))
                hours_aurora.append(h)
                dry_vals.append(dry)
                water_vals.append(water)
                energy_vals.append(energy)
                pe_vals.append(pe_step)
                hydro_vals.append(hydro)
                geo_vals.append(geo)

                # Append to time-series rows
                ts_rows.append({
                    "date": date_str,
                    "forecast_hour": h,
                    "dry_mass_Eg": dry,
                    "water_mass_kg": water,
                    "total_energy_J": energy,
                    "pe_cumulative_kg": pe_step,
                    "hydrostatic_rmse": hydro,
                    "geostrophic_rmse": geo,
                    "sp_method": step_sp_method,
                })

                del snap, ps_snap  # free memory immediately

            hours_aurora = np.array(hours_aurora)
            dry_vals = np.array(dry_vals)
            water_vals = np.array(water_vals)
            energy_vals = np.array(energy_vals)
            pe_vals = np.array(pe_vals)

            _log(f"    [{counter}] Drift: {len(avail_tds)} steps "
                 f"[{hours_aurora[0]:.0f}h–{hours_aurora[-1]:.0f}h]"
                 f"  sp_method={step_sp_method}")

            # Also log balance RMSEs from the target lead time snapshot
            # Include ERA5 RMSE at the same valid time for comparison
            era5_hydro, era5_geo = None, None
            if ds_era5_t is not None:
                era5_ld = _detect_level_dim(ds_era5_t)
                try:
                    era5_hydro = compute_hydrostatic_imbalance(
                        ds_era5_t, area, level_dim=era5_ld,
                    )
                except Exception:
                    pass
                try:
                    era5_geo = compute_geostrophic_imbalance(
                        ds_era5_t, area, level_dim=era5_ld,
                    )
                except Exception:
                    pass
            _append("hydrostatic_rmse", hydro_vals[-1], era5_hydro)
            _append("geostrophic_rmse", geo_vals[-1], era5_geo)

            # --- ERA5 time-series (water & energy anomalous drift) ---
            hours_era5_arr = np.array([], dtype=np.float64)
            water_vals_e = np.array([], dtype=np.float64)
            energy_vals_e = np.array([], dtype=np.float64)

            try:
                hours_era5 = []
                water_e_list, energy_e_list = [], []

                for h in hours_aurora:
                    vt = init_time + np.timedelta64(int(h), "h")

                    era5_snap = ds_era5_full.sel(time=vt)
                    drop_e_vars = [v for v in era5_snap.data_vars if v.strip() not in _CONS_VARS]
                    if drop_e_vars:
                        era5_snap = era5_snap.drop_vars(drop_e_vars)
                    if "time" in era5_snap.dims:
                        era5_snap = era5_snap.isel(time=0)
                    era5_snap = era5_snap.load()

                    # Ensure (latitude, longitude) dim order to match area
                    if "latitude" in era5_snap.dims and "longitude" in era5_snap.dims:
                        _edims = list(era5_snap.dims)
                        _i_lon = _edims.index("longitude")
                        _i_lat = _edims.index("latitude")
                        if _i_lon < _i_lat:
                            _edims[_i_lon], _edims[_i_lat] = _edims[_i_lat], _edims[_i_lon]
                            era5_snap = era5_snap.transpose(*_edims)

                    # Compute t2m_mean for this ERA5 snapshot
                    snap_t2m_mean_e = None
                    try:
                        vt_e = era5_snap.time.values
                        if isinstance(vt_e, np.datetime64):
                            snap_t2m_mean_e = _get_daily_t2m(vt_e)
                    except Exception:
                        pass

                    era5_ld = _detect_level_dim(era5_snap)
                    try:
                        ps_e = _get_ps(era5_snap, ds_static, level_dim=era5_ld, t2m_mean=snap_t2m_mean_e)
                    except Exception:
                        ps_e = None

                    if ps_e is not None:
                        _, w_e, e_e = compute_conservation_scalars(
                            era5_snap, ps_e, area, z_sfc=z_sfc,
                            level_dim=era5_ld,
                        )
                    else:
                        w_e, e_e = float("nan"), float("nan")

                    hours_era5.append(h)
                    water_e_list.append(w_e)
                    energy_e_list.append(e_e)
                    del era5_snap, ps_e

                hours_era5_arr = np.array(hours_era5)
                water_vals_e = np.array(water_e_list)
                energy_vals_e = np.array(energy_e_list)
            except Exception as exc:
                _log(f"    [{counter}] ⚠ ERA5 drift series failed: {exc}")

            # --- Compute final drift percentages via helper ---
            if len(hours_era5_arr) >= 2 and len(water_vals_e) >= 2:
                drift = compute_drift_percentages(
                    hours_aurora, dry_vals, water_vals, energy_vals,
                    hours_era5_arr, water_vals_e, energy_vals_e,
                )
            else:
                # ERA5 series unavailable — compute dry-only drift
                slope_dry = compute_drift_slope(hours_aurora, dry_vals)
                dry_ref = float(dry_vals[0])
                drift = {
                    "dry_mass_drift_pct_per_day": (
                        (slope_dry / dry_ref) * 100.0
                        if dry_ref != 0 and np.isfinite(slope_dry)
                        else float("nan")
                    ),
                    "water_mass_drift_pct_per_day":   float("nan"),
                    "total_energy_drift_pct_per_day": float("nan"),
                }

            for metric_name, val in drift.items():
                _append(metric_name, val)

            # --- Water budget residual drift (P-E based) ---
            # For models using ERA5 sp + P-E: always compute water budget drift
            # For other models: only compute when using fixed_1000hPa_pure_levels fallback
            _compute_pe_budget = (
                (_use_era5_sp and _has_pe and len(pe_vals) >= 2 and not all(np.isnan(pe_vals))) or
                (step_sp_method == "fixed_1000hPa_pure_levels" and len(pe_vals) >= 2)
            )
            if _compute_pe_budget:
                # 1. Calculate the cumulative discrepancy D(t)
                w_0 = water_vals[0]
                pe_0 = pe_vals[0]
                
                # Because the model variable is P-E, an increase means water left the atmosphere.
                # Therefore, D(t) = (W(t) - W(0)) - (-(PE(t) - PE(0)))
                # Which simplifies to: D(t) = (W(t) - W(0)) + (PE(t) - PE(0))
                discrepancy = (water_vals - w_0) + (pe_vals - pe_0)

                # 2. Calculate the linear drift slope of the discrepancy
                slope_D = compute_drift_slope(hours_aurora, discrepancy)
                
                # 3. Convert to %/day relative to initial column water
                if w_0 != 0 and np.isfinite(slope_D):
                    water_budget_drift_pct = (slope_D / w_0) * 100.0
                else:
                    water_budget_drift_pct = float("nan")

                _append("water_budget_drift_pct_per_day", water_budget_drift_pct)
                _log(f"    [{counter}] Water budget drift (P-E): "
                     f"{water_budget_drift_pct:.6g} %/day")

            _log(f"    [{counter}] ✓ Drift: "
                 f"dry={drift['dry_mass_drift_pct_per_day']:+.6g}%/day  "
                 f"water={drift['water_mass_drift_pct_per_day']:+.6g}%/day  "
                 f"energy={drift['total_energy_drift_pct_per_day']:+.6g}%/day")

        except Exception as exc:
            _log(f"    [{counter}] ⚠ Drift metrics failed: {exc}")

    elif mode == "era5" and ds_era5_t is not None:
        # In era5 mode, we only care about intrinsic metrics computed directly on ERA5
        try:
            era5_ld = _detect_level_dim(ds_era5_t)
            
            hydro = float("nan")
            try:
                hydro = compute_hydrostatic_imbalance(ds_era5_t, area, level_dim=era5_ld)
            except Exception as exc:
                _log(f"    [{counter}] ⚠ ERA5 hydrostatic failed: {exc}")
                
            geo = float("nan")
            try:
                geo = compute_geostrophic_imbalance(ds_era5_t, area, level_dim=era5_ld)
            except Exception as exc:
                _log(f"    [{counter}] ⚠ ERA5 geostrophic failed: {exc}")
            
            _append("hydrostatic_rmse", None, hydro)
            _append("geostrophic_rmse", None, geo)
            
            if ps_era5 is not None:
                dry, water, energy = compute_conservation_scalars(
                    ds_era5_t, ps_era5, area, z_sfc=z_sfc, level_dim=era5_ld
                )
                _append("dry_mass_Eg", None, dry)
                _append("water_mass_kg", None, water)
                _append("total_energy_J", None, energy)
                
            _log(f"    [{counter}] ✓ ERA5 Intrinsic: hydro={hydro:.4g}, geo={geo:.4g}")
            
        except Exception as exc:
            _log(f"    [{counter}] ⚠ ERA5 intrinsic metrics failed: {exc}")


    # ---- Spectral metrics (comparative) ----
    # Only meaningful if we have Aurora predictions to compare against ERA5
    # Skip if we had a lead-time mismatch (e.g. NeuralGCM has no 6h step)
    if mode in ("joint", "prediction", "aurora") and ds_aurora_t is not None and ds_era5_t is not None and not _lead_td_mismatch:
        try:
            # Align ERA5 grid to Aurora (720 vs 721 lat) before spectral analysis
            ds_era5_aligned = _align_era5_to_aurora(ds_era5_t, ds_aurora_t)
        except ValueError as exc:
            _log(f"    [{counter}] ⚠ Grid alignment failed (spectral skipped): {exc}")
            ds_era5_aligned = None

        if ds_era5_aligned is not None:
            try:
                k_pred, e_pred = compute_ke_spectrum(ds_aurora_t)
                k_era5, e_era5 = compute_ke_spectrum(ds_era5_aligned)

                # Align lengths in case l_max differs slightly between grids
                n_min = min(len(e_pred), len(e_era5))
                k_common = k_pred[:n_min]
                e_pred = e_pred[:n_min]
                e_era5 = e_era5[:n_min]

                # Effective resolution & small-scale ratio
                try:
                    L_eff, ratio = _find_effective_resolution(k_common, e_pred, e_era5)
                    _append("effective_resolution_km", L_eff)
                    _append("small_scale_ratio",       ratio)
                except Exception as exc:
                    _log(f"    [{counter}] ⚠ Eff.Res error: {exc}")

                # Spectral divergence & residual
                try:
                    s_div, s_res = compute_spectral_scores(e_pred, e_era5)
                    _append("spectral_divergence", s_div)
                    _append("spectral_residual",   s_res)
                except Exception as exc:
                    _log(f"    [{counter}] ⚠ Spectral error: {exc}")

                # Raw KE spectrum rows
                for wi in range(n_min):
                    spectrum_rows.append({
                        "date": date_str,
                        "lead_hours": lead_hours,
                        "wavenumber": int(k_common[wi]),
                        "energy_pred": float(e_pred[wi]),
                        "energy_era5": float(e_era5[wi]),
                    })
                _log(f"    [{counter}] ✓ KE spectrum saved (l_max={n_min-1})")

            except ImportError:
                pass
            except Exception as exc:
                _log(f"    [{counter}] ⚠ KE spectrum error: {exc}")

    # If no rows were produced at all (all metrics failed), write a
    # placeholder so the CSV retains the (date, lead_time) structure.
    if not rows:
        _log(f"    [{counter}] ⚠ All metrics failed — writing NaN placeholder row")
        _append("ALL_METRICS_FAILED", None, None)

    return rows, ts_rows, spectrum_rows


# ============================================================================
# Sanity Check
# ============================================================================

def _sanity_check_era5(
    ds_era5_full: xr.Dataset,
    verbose: bool = True,
) -> None:
    """
    Verify that ERA5 .sel(time=…) returns *different* data for different
    valid times.  If all three sample slices are identical, the datetime
    resolution is wrong and results will be meaningless.
    """
    # Pick 3 sample times from the ERA5 dataset's own time range
    # (at ~25%, 50%, 75%) to avoid assuming a particular year range.
    times = ds_era5_full.time.values
    n = len(times)
    sample_times = [
        np.datetime64(times[n // 4], "ns"),
        np.datetime64(times[n // 2], "ns"),
        np.datetime64(times[3 * n // 4], "ns"),
    ]

    # Pick a variable that should change over time
    test_var = _find_var(ds_era5_full, T_NAMES) or _find_var(ds_era5_full, MSL_NAMES)
    if test_var is None:
        if verbose:
            print("  ⚠ Sanity check skipped: no temperature or MSL variable found.")
        return

    if verbose:
        print(f"\n  Sanity check — verifying ERA5 time selection on '{test_var}':")

    means = []
    for t in sample_times:
        slc = ds_era5_full[test_var].sel(time=t, method="nearest")
        actual_time = slc.time.values
        m = float(slc.mean())
        means.append(m)
        if verbose:
            print(f"    requested={t}  actual={actual_time}  mean={m:.6g}")

    if means[0] == means[1] == means[2]:
        raise RuntimeError(
            "ERA5 sanity check FAILED: all three sample times returned "
            "identical data.  This indicates a datetime resolution bug.  "
            f"Means: {means}"
        )

    if verbose:
        print("  ✓ ERA5 time selection looks correct (values differ).\n")


# ============================================================================
# Batch Analysis
# ============================================================================

def _parse_lead_times(spec: str) -> list[tuple[str, np.timedelta64]]:
    """Parse a comma-separated lead-time specification.

    Accepted formats per item: ``6h``, ``12h``, ``5d``, ``120h``, ``10d``, ``240h``.
    Returns a list of ``(label, timedelta)`` tuples.
    """
    result = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token.endswith("d"):
            days = int(token[:-1])
            td = np.timedelta64(days * 24, "h")
            result.append((token, td))
        elif token.endswith("h"):
            hours = int(token[:-1])
            td = np.timedelta64(hours, "h")
            # Use a friendly label for common values
            if hours % 24 == 0 and hours >= 48:
                label = f"{hours // 24}d"
            else:
                label = f"{hours}h"
            result.append((label, td))
        else:
            raise ValueError(f"Cannot parse lead-time token: {token!r}")
    return result


def run_evaluation(
    dates: list[str],
    output_csv: Path,
    mode: str = "joint",
    workers: int = DEFAULT_WORKERS,
    verbose: bool = True,
    prediction_zarr: str = AURORA_ZARR,
    era5_zarr: str = ERA5_ZARR,
    model_name: str = "aurora",
    lead_times: list[tuple[str, np.timedelta64]] | None = None,
    static_zarr: str | None = None,
) -> pd.DataFrame:
    """
    Compute all physics metrics for each (date × lead time), parallelised.

    Parameters
    ----------
    dates : list[str]
        ISO date strings, e.g. ["2022-01-01", "2022-01-15"].
    output_csv : Path
        Destination CSV file.
    mode : str
        'joint', 'era5', or 'aurora'.
    workers : int
        Number of parallel threads.
    verbose : bool
        Print progress.
    model_name : str
        Model name for secondary CSV filename.
    lead_times : list of (label, timedelta), optional
        Override the default LEAD_TIMES (6h, 5d, 10d).

    Returns
    -------
    pd.DataFrame
        Long-format with columns:
        date, lead_time_hours, metric_name, model_value, era5_value,
        n_levels, sp_method.
    """
    # Build work items: list of (date, lead_label, lead_td)
    _lead_times = lead_times if lead_times is not None else LEAD_TIMES
    work_items = [
        (date_str, lead_label, lead_td)
        for date_str in dates
        for lead_label, lead_td in _lead_times
    ]
    n_combos = len(work_items)

    if verbose:
        print("\n" + "=" * 70)
        print("  PHYSICS EVALUATION — WeatherBench 2 Zarr Streaming")
        print("=" * 70)
        print(f"  Prediction : {prediction_zarr}")
        print(f"  ERA5       : {era5_zarr}")
        print(f"  Dates  : {len(dates)}")
        print(f"  Lead times : {[label for label, _ in _lead_times]}")
        print(f"  Total evals: {n_combos}")
        print(f"  Workers    : {workers}")
        print(f"  Mode       : {mode}")
        print(f"  Output : {output_csv}")
        print("=" * 70)

    # ---- Open Zarr lazily (shared across threads) ----
    # For ProcessPoolExecutor, we pass paths to workers and they open datasets themselves.
    # But checking if we can open them first is good practice.
    if verbose:
        print("\n  Checking Zarr stores (anonymous) …")
    
    # Just open to check presence and print metadata
    ds_aurora_check = None
    if mode in ("joint", "prediction", "aurora"):
        ds_aurora_check = open_zarr_anonymous(prediction_zarr)
    
    ds_era5_check = open_zarr_anonymous(era5_zarr)

    if verbose:
        if ds_aurora_check is not None:
            print(f"  Aurora vars: {list(ds_aurora_check.data_vars)[:12]}")
        print(f"  ERA5 vars  : {list(ds_era5_check.data_vars)[:12]}")
        print(f"  ERA5 time dtype: {ds_era5_check.time.dtype}")
        print(f"  ERA5 time range: {ds_era5_check.time.values[0]} → "
              f"{ds_era5_check.time.values[-1]}")

    # ---- Sanity check: verify ERA5 .sel(time=…) returns different data ----
    _sanity_check_era5(ds_era5_check, verbose=verbose)

    # ---- Parallel evaluation ----
    if verbose:
        print(f"\n  Launching {workers} processes …\n")

    all_rows: list[dict] = []
    all_ts_rows: list[dict] = []
    all_spectrum_rows: list[dict] = []
    
    # Per-task timeout (seconds).  Prevents a single stale GCS connection
    # from blocking the entire run indefinitely.
    TASK_TIMEOUT = 600  # 10 minutes

    # Use ProcessPoolExecutor for true parallelism (avoids GIL for spectral)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for idx, (date_str, lead_label, lead_td) in enumerate(work_items, 1):
            fut = pool.submit(
                _evaluate_one,
                prediction_zarr, era5_zarr, ERA5_DAILY_ZARR, # Pass URL paths, not dataset objects
                date_str, lead_label, lead_td,
                idx, n_combos, mode, verbose,
                static_zarr,
                model_name,
            )
            futures[fut] = (idx, date_str, lead_label)

        for fut in as_completed(futures):
            idx, date_str, lead_label = futures[fut]
            try:
                summary_rows, ts_rows, spectrum_rows = fut.result(timeout=TASK_TIMEOUT)
                all_rows.extend(summary_rows)
                all_ts_rows.extend(ts_rows)
                all_spectrum_rows.extend(spectrum_rows)
            except TimeoutError:
                if verbose:
                    print(f"  ⚠ Task {idx} ({date_str} {lead_label}) timed out "
                          f"after {TASK_TIMEOUT}s — skipping", flush=True)
            except Exception as exc:
                if verbose:
                    print(f"  ⚠ Worker exception (task {idx}): {exc}", flush=True)

    # ---- Sort by date + lead time for clean output ----
    all_rows.sort(key=lambda r: (r["date"], r["lead_time_hours"], r["metric_name"]))

    # ---- Save summary CSV ----
    df = pd.DataFrame(all_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if verbose:
        _print_summary(df)
        print(f"\n  ✓ Results saved → {output_csv}")

    # ---- Save time-series CSV ----
    if all_ts_rows:
        # Derive year from first date in the list
        year_str = dates[0][:4] if dates else "unknown"
        # Extract reference suffix from main output filename (e.g., "_ifs")
        main_stem = output_csv.stem  # e.g., "physics_evaluation_pangu_2020_ifs"
        ref_suffix = ""
        if main_stem.endswith("_ifs"):
            ref_suffix = "_ifs"
        ts_csv = output_csv.parent / f"physics_timeseries_{model_name}_{year_str}{ref_suffix}.csv"
        df_ts = pd.DataFrame(all_ts_rows)
        df_ts.sort_values(["date", "forecast_hour"], inplace=True)
        df_ts.to_csv(ts_csv, index=False)
        if verbose:
            print(f"  ✓ Time-series saved → {ts_csv}  ({len(df_ts)} rows)")

    # ---- Save KE spectrum CSV ----
    if all_spectrum_rows:
        year_str = dates[0][:4] if dates else "unknown"
        # Extract reference suffix from main output filename (e.g., "_ifs")
        main_stem = output_csv.stem
        ref_suffix = ""
        if main_stem.endswith("_ifs"):
            ref_suffix = "_ifs"
        spec_csv = output_csv.parent / f"ke_spectrum_{model_name}_{year_str}{ref_suffix}.csv"
        df_spec = pd.DataFrame(all_spectrum_rows)
        df_spec.sort_values(["date", "lead_hours", "wavenumber"], inplace=True)
        df_spec.to_csv(spec_csv, index=False)
        if verbose:
            print(f"  ✓ KE spectrum saved → {spec_csv}  ({len(df_spec)} rows)")

    return df


# ============================================================================
# Summary
# ============================================================================

def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    ok = df[df["metric_name"] != "ERROR"]
    n_dates = ok["date"].nunique()
    n_errors = len(df[df["metric_name"] == "ERROR"])
    print(f"  Total metric rows : {len(ok)}")
    print(f"  Dates evaluated   : {n_dates}")
    print(f"  Error entries     : {n_errors}")

    if len(ok) == 0:
        print("  ⚠  No successful calculations.")
        return

    for lead_hours, lead_grp in ok.groupby("lead_time_hours"):
        print(f"\n  ── Lead time = {lead_hours} h ──")

        for metric, mgrp in lead_grp.groupby("metric_name"):
            model = mgrp["model_value"].dropna()
            era5 = mgrp["era5_value"].dropna()

            if len(era5) > 0 and len(model) > 0:
                print(
                    f"    {metric:30s}  "
                    f"Model={model.mean():.6g}  "
                    f"ERA5={era5.mean():.6g}  "
                    f"Δ={model.mean() - era5.mean():+.4g}"
                )
            elif len(model) > 0:
                print(
                    f"    {metric:30s}  "
                    f"Model={model.mean():.6g}"
                )
            elif len(era5) > 0:
                print(
                    f"    {metric:30s}  "
                    f"ERA5={era5.mean():.6g}"
                )

    print("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Physics evaluation for Aurora vs ERA5 (WB2 Zarr streaming)"
    )
    parser.add_argument(
        "--year", type=int, default=2022,
        help="Year to evaluate (default: 2022). "
             "Ignored if --dates or --month is provided.",
    )
    parser.add_argument(
        "--dates", nargs="+", default=None,
        help="Dates to evaluate, e.g. 2022-01-01 2022-01-15",
    )
    parser.add_argument(
        "--month", type=str, default=None,
        help="Evaluate all days of a month, e.g. 2022-01",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--mode", type=str, choices=["joint", "era5", "prediction", "aurora"], default="joint",
        help="Evaluation mode (default: joint). 'prediction' mode is optimized for model evaluation.",
    )
    parser.add_argument(
        "--model", type=str, default="aurora",
        help="Name of the model (default: aurora). Used for output filename.",
    )
    parser.add_argument(
        "--prediction-zarr", type=str, default=AURORA_ZARR,
        help=f"Path to prediction Zarr (default: {AURORA_ZARR})",
    )
    parser.add_argument(
        "--era5-zarr", type=str, default=ERA5_ZARR,
        help=f"Path to ERA5 Zarr (default: {ERA5_ZARR})",
    )
    parser.add_argument(
        "--reference", type=str, choices=["era5", "ifs"], default="era5",
        help="Reference dataset: 'era5' (default) or 'ifs' (IFS HRES t=0 analysis). "
             "When 'ifs' is selected, output files get '_ifs' suffix.",
    )
    parser.add_argument(
        "--lead-times", type=str, default=None,
        help="Comma-separated lead times, e.g. '12h,5d,10d'. "
             "Default: 6h,5d,10d.",
    )
    parser.add_argument(
        "--static-zarr", type=str, default=None,
        help="Path to Zarr with static fields (geopotential_at_surface, land_sea_mask). "
             "Defaults to --era5-zarr.  Set this when --era5-zarr lacks static fields "
             "(e.g. HRES-T0).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    dates = _resolve_dates(args)

    # Handle --reference option
    ref_zarr = args.era5_zarr
    static_zarr = args.static_zarr
    ref_suffix = ""
    
    if args.reference == "ifs":
        # Use IFS HRES t=0 as reference instead of ERA5
        # Check if model uses low-res grid (NeuralGCM)
        if "512x256" in args.prediction_zarr or "neuralgcm" in args.model.lower() or "512x256" in args.model:
            ref_zarr = IFS_T0_LOWRES_ZARR
        else:
            ref_zarr = IFS_T0_ZARR
        # IFS HRES t=0 lacks static fields, so we need ERA5 for those
        if static_zarr is None:
            static_zarr = ERA5_ZARR
        ref_suffix = "_ifs"

    if args.output:
        output = Path(args.output)
    else:
        output = OUTPUT_DIR / f"physics_evaluation_{args.model}_{args.year}{ref_suffix}.csv"

    lt = _parse_lead_times(args.lead_times) if args.lead_times else None

    run_evaluation(
        dates=dates,
        output_csv=output,
        mode=args.mode,
        workers=args.workers,
        verbose=not args.quiet,
        prediction_zarr=args.prediction_zarr,
        era5_zarr=ref_zarr,
        model_name=args.model,
        lead_times=lt,
        static_zarr=static_zarr,
    )


if __name__ == "__main__":
    main()
