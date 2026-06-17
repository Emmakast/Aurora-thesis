#!/usr/bin/env python
"""
Physics Evaluation Runner — Steered Aurora Predictions

Evaluates physical consistency of base vs steered predictions
across multiple alpha values for the neutral dates.
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

# Import physics metrics from companion library
sys.path.insert(0, str(Path(__file__).parent))
from physics_metrics import (
    compute_conservation_scalars,
    compute_geostrophic_imbalance,
    compute_hydrostatic_imbalance,
    compute_ke_spectrum,
    compute_spectral_scores,
    derive_surface_pressure,
    get_grid_cell_area,
    _find_var,
    _detect_level_dim,
    _find_effective_resolution,
    compute_lapse_rate_wasserstein,
    compute_drift_percentages,
    SP_NAMES,
    MSL_NAMES,
    ZSFC_NAMES,
    Q_NAMES,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# Configuration
# ============================================================================

ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

OUTPUT_DIR = Path.home() / "aurora_thesis" / "thesis" / "results"
LOCAL_DIR = Path("/scratch-shared/ekasteleyn/ao_neutral_steered")

LEAD_HOURS = 72

VAR_NAME_MAP = {
    "t": "temperature",
    "z": "geopotential", 
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}

DEFAULT_WORKERS = 8
DEFAULT_ALPHAS = [0.0, 1.0, 2.0, 5.0, 10.0]

def get_file_path(date_str: str, alpha: float) -> Optional[Path]:
    date_tag = date_str.replace("-", "")
    if alpha == 0.0:
        filename = f"base_ao_ao81_polar_{date_tag}_1200_alpha_0.0.nc"
    else:
        filename = f"steered_ao_ao81_polar_{date_tag}_1200_polar_north_lat60p0_alpha_{alpha}.nc"
    
    local_path = LOCAL_DIR / filename
    if local_path.exists():
        return local_path
    return None

def open_zarr_anonymous(url: str) -> xr.Dataset:
    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    rename = {}
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

    for name in ("geopotential_at_surface", "z_sfc", "orography"):
        if name in ds_era5.data_vars:
            static_vars[name] = _extract_static(ds_era5, name)
            break

    for name in ("land_sea_mask", "lsm"):
        if name in ds_era5.data_vars:
            static_vars[name] = _extract_static(ds_era5, name)
            break
    return xr.Dataset(static_vars)

def _get_ps(ds: xr.Dataset, ds_static: xr.Dataset, level_dim: str = "level") -> xr.DataArray:
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

    raise ValueError("Cannot derive surface pressure")

def _align_era5_to_aurora(
    ds_era5: xr.Dataset,
    ds_aurora: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    n_era5 = ds_era5.sizes.get(lat_name, 0)
    n_aurora = ds_aurora.sizes.get(lat_name, 0)
    n_lon_era5 = ds_era5.sizes.get(lon_name, 0)
    n_lon_aurora = ds_aurora.sizes.get(lon_name, 0)

    if n_lon_era5 != n_lon_aurora:
        raise ValueError(f"Longitude mismatch: {n_lon_era5} vs {n_lon_aurora}")

    if n_era5 == n_aurora:
        result = ds_era5
    elif n_era5 == n_aurora + 1:
        lats = ds_era5[lat_name].values
        if lats[0] > lats[-1]:  # Descending
            result = ds_era5.isel({lat_name: slice(0, -1)})
        else:                   # Ascending
            result = ds_era5.isel({lat_name: slice(1, None)})
    else:
        raise ValueError(f"Latitude mismatch: ERA5 has {n_era5}, aurora has {n_aurora}")

    result = result.assign_coords({lat_name: ds_aurora[lat_name].values})

    if lat_name in result.dims and lon_name in result.dims:
        dims_list = list(result.dims)
        idx_lon = dims_list.index(lon_name)
        idx_lat = dims_list.index(lat_name)
        if idx_lon < idx_lat:
            dims_list[idx_lon], dims_list[idx_lat] = dims_list[idx_lat], dims_list[idx_lon]
            result = result.transpose(*dims_list)
    return result

def _evaluate_one(
    date_str: str,
    alpha: float,
    counter: int,
    total: int,
    verbose: bool,
    no_spectrum: bool = False,
) -> list[dict]:
    rows = []
    
    def _log(msg):
        if verbose:
            print(msg, flush=True)
    
    _log(f"  [{counter}/{total}] {date_str} alpha={alpha}")
    
    try:
        local_path = get_file_path(date_str, alpha)
        if local_path is None:
            _log(f"    ⚠ File not found for {date_str} alpha {alpha}")
            return rows
            
        ds_aurora = xr.open_dataset(local_path)
        rename_map = {k: v for k, v in VAR_NAME_MAP.items() if k in ds_aurora.data_vars}
        if rename_map:
            ds_aurora = ds_aurora.rename(rename_map)
        
        # Load ERA5
        ds_era5 = open_zarr_anonymous(ERA5_ZARR)
        ds_static = load_static_fields(ds_era5)
        
        # Load Aurora data
        ds_aurora = ds_aurora.load()
        
        # Squeeze dimensions if needed
        if "time" in ds_aurora.dims:
            ds_aurora = ds_aurora.isel(time=0)
            
        init_time = np.datetime64(f"{date_str}T12:00", "ns")
        lead_td = np.timedelta64(LEAD_HOURS, "h")
        valid_time = init_time + lead_td

        # Align ERA5 to Aurora Grid
        ds_era5_t = ds_era5.sel(time=valid_time, method="nearest").load()
        ds_era5_aligned = _align_era5_to_aurora(ds_era5_t, ds_aurora)
        
        # Grid cell area
        area_era5 = get_grid_cell_area(ds_era5.isel(time=0, drop=True))
        area_aurora = _align_era5_to_aurora(area_era5.to_dataset(name="area"), ds_aurora)["area"]
        
        # Get static fields
        z_sfc_name = _find_var(ds_static, ZSFC_NAMES)
        z_sfc_era5 = ds_static[z_sfc_name] if z_sfc_name else None
        z_sfc_aurora = None
        if z_sfc_era5 is not None:
            z_sfc_aurora = _align_era5_to_aurora(z_sfc_era5.to_dataset(name="z_sfc"), ds_aurora)["z_sfc"]
        
        # Level dimension
        level_dim = _detect_level_dim(ds_aurora)
        n_levels = len(ds_aurora[level_dim]) if level_dim else None
        
        def _append(metric_name, model_val, era5_val=None):
            rows.append({
                "date": date_str,
                "alpha": alpha,
                "lead_time_hours": LEAD_HOURS,
                "metric_name": metric_name,
                "model_value": model_val,
                "era5_value": era5_val,
                "n_levels": n_levels,
            })
        
        # --- Compute Metrics ---
        
        # 1. Hydrostatic Imbalance
        try:
            hydro_aurora = compute_hydrostatic_imbalance(ds_aurora, area_aurora, level_dim=level_dim)
            hydro_era5 = compute_hydrostatic_imbalance(ds_era5_t, area_era5)
            _append("hydrostatic_rmse", float(hydro_aurora), float(hydro_era5))
            _append("hydrostatic_rmse_diff", float(hydro_aurora - hydro_era5))
            _log(f"    Hydrostatic RMSE: {hydro_aurora:.4f} (ERA5: {hydro_era5:.4f})")
        except Exception as e:
            _log(f"    ⚠ Hydrostatic error: {e}")
        
        # 2. Geostrophic Imbalance
        try:
            geo_aurora = compute_geostrophic_imbalance(ds_aurora, area_aurora, level_dim=level_dim)
            geo_era5 = compute_geostrophic_imbalance(ds_era5_t, area_era5)
            _append("geostrophic_rmse", float(geo_aurora), float(geo_era5))
            _append("geostrophic_rmse_diff", float(geo_aurora - geo_era5))
            _log(f"    Geostrophic RMSE: {geo_aurora:.4f} (ERA5: {geo_era5:.4f})")
        except Exception as e:
            _log(f"    ⚠ Geostrophic error: {e}")
            
        # 3. Lapse Rate
        try:
            lr_w1 = compute_lapse_rate_wasserstein(ds_aurora, ds_era5_t, area_aurora, level_dim_pred=level_dim)
            if "lapse_rate_w1_global" in lr_w1:
                _append("lapse_rate_wasserstein", lr_w1["lapse_rate_w1_global"])
        except Exception as e:
            _log(f"    ⚠ Lapse rate error: {e}")
        
        # 4. Conservation metrics & Drift
        try:
            ds_static_aurora = ds_static.interp(
                latitude=ds_aurora.latitude,
                longitude=ds_aurora.longitude,
                method="nearest"
            )
            ps_aurora = _get_ps(ds_aurora, ds_static_aurora, level_dim=level_dim)
            ps_era5 = _get_ps(ds_era5_t, ds_static)
            
            dry_aurora, water_aurora, energy_aurora = compute_conservation_scalars(
                ds_aurora, ps_aurora, area_aurora, z_sfc=z_sfc_aurora, level_dim=level_dim
            )
            dry_era5, water_era5, energy_era5 = compute_conservation_scalars(
                ds_era5_t, ps_era5, area_era5, z_sfc=z_sfc_era5
            )
            
            _append("dry_mass_Eg", float(dry_aurora), float(dry_era5))
            _append("water_mass_kg", float(water_aurora), float(water_era5))
            _append("total_energy_J", float(energy_aurora), float(energy_era5))
            
            try:
                ds_era5_0h = ds_era5.sel(time=init_time, method="nearest").load()
                ps_era5_0h = _get_ps(ds_era5_0h, ds_static)
                dry_era5_0h, water_era5_0h, energy_era5_0h = compute_conservation_scalars(
                    ds_era5_0h, ps_era5_0h, area_era5, z_sfc=z_sfc_era5
                )
                
                drift_metrics = compute_drift_percentages(
                    hours_aurora=np.array([0, 72]),
                    dry_aurora=np.array([dry_era5_0h, dry_aurora]),
                    water_aurora=np.array([water_era5_0h, water_aurora]),
                    energy_aurora=np.array([energy_era5_0h, energy_aurora]),
                    hours_era5=np.array([0, 72]),
                    water_era5=np.array([water_era5_0h, water_era5]),
                    energy_era5=np.array([energy_era5_0h, energy_era5])
                )
                
                _append("dry_mass_drift_pct_per_day", float(drift_metrics["dry_mass_drift_pct_per_day"]))
                _append("water_mass_drift_pct_per_day", float(drift_metrics["water_mass_drift_pct_per_day"]))
                _append("total_energy_drift_pct_per_day", float(drift_metrics["total_energy_drift_pct_per_day"]))
            except Exception as e:
                _log(f"    ⚠ Conservation/Drift error: {e}")
                
        except Exception as e:
            _log(f"    ⚠ Conservation/Drift error: {e}")
        
        # 5. KE Spectrum & Effective Resolution
        if not no_spectrum:
            try:
                wn_aurora, e_aurora = compute_ke_spectrum(ds_aurora, level_dim=level_dim)
                wn_era5, e_era5 = compute_ke_spectrum(ds_era5_aligned)
                
                if e_aurora is not None and e_era5 is not None:
                    spec_div, spec_res = compute_spectral_scores(e_aurora, e_era5)
                    _append("spectrum_divergence", float(spec_div))
                    _append("spectrum_residual", float(spec_res))
                    res = _find_effective_resolution(wn_aurora, e_aurora, e_era5)
                    L_eff = res[0] if isinstance(res, tuple) else res
                    _append("effective_resolution_km", float(L_eff))
            except Exception as e:
                _log(f"    ⚠ Spectrum error: {e}")
        
        ds_aurora.close()
    
    except Exception as e:
        _log(f"    ⚠ Error: {e}")
    
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-spectrum", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    args = parser.parse_args()
    
    output_csv = Path(args.output) if args.output else OUTPUT_DIR / "physics_aurora_ao81_steered.csv"
    
    dates_df = pd.read_csv("/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_ao_81.csv")
    dates_df_neutral = dates_df[dates_df["Type"] == "Neutral"].copy()
    dates = [f"{row.Year}-{row.Month:02d}-{row.Day:02d}" for _, row in dates_df_neutral.iterrows()]
    
    work_items = [(date_str, alpha) for date_str in dates for alpha in args.alphas]
    n_total = len(work_items)
    
    if args.verbose:
        print(f"Starting evaluation of {n_total} items with {args.workers} workers...")
    
    all_rows = []
    
    if args.workers == 1:
        for idx, (date_str, alpha) in enumerate(work_items, 1):
            rows = _evaluate_one(date_str, alpha, idx, n_total, args.verbose, args.no_spectrum)
            all_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for idx, (date_str, alpha) in enumerate(work_items, 1):
                fut = pool.submit(_evaluate_one, date_str, alpha, idx, n_total, args.verbose, args.no_spectrum)
                futures[fut] = (date_str, alpha)
            
            for fut in as_completed(futures):
                try:
                    rows = fut.result(timeout=600)
                    all_rows.extend(rows)
                except Exception as e:
                    date_str, alpha = futures[fut]
                    print(f"  ⚠ Failed {date_str} alpha={alpha}: {e}")
    
    df = pd.DataFrame(all_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    if args.verbose:
        print(f"\n✓ Saved {len(df)} rows to {output_csv}")

if __name__ == "__main__":
    main()
