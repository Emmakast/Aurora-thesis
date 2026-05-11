from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import sys
sys.path.insert(0, str(Path(__file__).parent))
from physics_metrics import (
    compute_hydrostatic_imbalance,
    get_grid_cell_area,
    _find_var,
    _detect_level_dim,
    _detect_pred_td_dim,
    T_NAMES,
    Q_NAMES,
)

# Module-level cache: opened lazily on first use within each worker
_WORKER_MODEL_DS = None
_WORKER_REF_DS = None
_WORKER_INIT_ARGS = None  # (model_url, ref_url)


def open_zarr_anonymous(url: str) -> xr.Dataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ds = xr.open_zarr(url, storage_options={"token": "anon"}, decode_timedelta=True)
    rename = {v: v.strip() for v in ds.data_vars if v != v.strip()}
    rename.update({d: d.strip() for d in ds.dims if d != d.strip()})
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def _get_worker_datasets():
    """Lazily open zarr datasets on first call within a worker process."""
    global _WORKER_MODEL_DS, _WORKER_REF_DS, _WORKER_INIT_ARGS
    if _WORKER_MODEL_DS is None:
        model_url, ref_url = _WORKER_INIT_ARGS
        print(f"Worker opening zarr connections...", flush=True)
        _WORKER_MODEL_DS = open_zarr_anonymous(model_url)
        _WORKER_REF_DS = open_zarr_anonymous(ref_url)
    return _WORKER_MODEL_DS, _WORKER_REF_DS


def _init_worker(model_url: str, ref_url: str):
    """Store URLs only — actual zarr open is deferred to first task."""
    global _WORKER_INIT_ARGS
    _WORKER_INIT_ARGS = (model_url, ref_url)

def _align_area(area: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    """Ensure area dimensions match the spatial dim order of ds."""
    # Find the lat and lon dim names as they appear in ds
    lat_dim = next((d for d in ds.dims if d in area.dims and "lat" in d.lower()), None)
    lon_dim = next((d for d in ds.dims if d in area.dims and "lon" in d.lower()), None)

    if lat_dim is None or lon_dim is None:
        # Fallback: just match by position using whatever dims area has
        return area

    # Only transpose if the order is actually wrong
    if area.dims != (lat_dim, lon_dim):
        area = area.transpose(lat_dim, lon_dim)
    return area


def _evaluate_hydrostatic_only(date_str: str, lead_hours: int):
    ds_model_full, ds_ref_full = _get_worker_datasets()

    init_time = np.datetime64(date_str, "ns")
    lead_td = np.timedelta64(lead_hours, "h")
    valid_time = init_time + lead_td

    area = get_grid_cell_area(ds_ref_full.isel(time=0, drop=True))
    # Force canonical (latitude, longitude) order immediately after computation
    lat_dim = next((d for d in area.dims if "lat" in d.lower()), None)
    lon_dim = next((d for d in area.dims if "lon" in d.lower()), None)
    if lat_dim and lon_dim and area.dims != (lat_dim, lon_dim):
        area = area.transpose(lat_dim, lon_dim)
    hydro_model, hydro_ref = np.nan, np.nan

    # Model
    try:
        ds_model_t = ds_model_full.sel(time=init_time)
        pred_td_dim = _detect_pred_td_dim(ds_model_t)
        if pred_td_dim in ds_model_t.dims:
            ds_model_t = ds_model_t.sel({pred_td_dim: lead_td}, method="nearest")
        if "time" in ds_model_t.dims:
            ds_model_t = ds_model_t.isel(time=0)
        ds_model_t = ds_model_t.load()
        ld_model = _detect_level_dim(ds_model_t)

        area_model = get_grid_cell_area(ds_model_t)
        area_model = _align_area(area_model, ds_model_t)  # Fix #2

        hydro_model = compute_hydrostatic_imbalance(ds_model_t, area_model, level_dim=ld_model)
    except Exception as e:
        print(f"  [!] Model failed for {date_str} + {lead_hours}h: {e}", flush=True)

    # Reference
    try:
        ds_ref_t = ds_ref_full.sel(time=valid_time)
        if "time" in ds_ref_t.dims:
            ds_ref_t = ds_ref_t.isel(time=0)
        ds_ref_t = ds_ref_t.load()
        ld_ref = _detect_level_dim(ds_ref_t)

        area_ref = _align_area(area, ds_ref_t)  # Fix #2
        hydro_ref = compute_hydrostatic_imbalance(ds_ref_t, area_ref, level_dim=ld_ref)
    except Exception as e:
        print(f"  [!] Ref failed for {date_str} + {lead_hours}h: {e}", flush=True)

    return date_str, lead_hours, hydro_model, hydro_ref


def main():
    parser = argparse.ArgumentParser("Update just hydrostatic values in an existing CSV.")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--prediction-zarr", type=str, required=True)
    parser.add_argument("--ref-zarr", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)  # Fix #1: safer default
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    mask = df["metric_name"] == "hydrostatic_rmse"
    tasks = df[mask][["date", "lead_time_hours"]].drop_duplicates()
    print(f"Found {len(tasks)} hydrostatic entries to update.")

    updates = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.prediction_zarr, args.ref_zarr),
    ) as pool:
        futures = {
            pool.submit(_evaluate_hydrostatic_only, r.date, r.lead_time_hours): (r.date, r.lead_time_hours)
            for _, r in tasks.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            date_str, lead_h = futures[fut]
            try:
                _, _, h_mod, h_ref = fut.result()
                updates[(date_str, lead_h)] = (h_mod, h_ref)
                print(f"[{i}/{len(tasks)}] {date_str} + {lead_h}h -> Model: {h_mod:.4f}, Ref: {h_ref:.4f}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(tasks)}] Task failed for {date_str} + {lead_h}h: {e}", flush=True)

    def patch_model(row):
        if row["metric_name"] == "hydrostatic_rmse" and (row["date"], row["lead_time_hours"]) in updates:
            return updates[(row["date"], row["lead_time_hours"])][0]
        return row["model_value"]

    def patch_ref(row):
        if row["metric_name"] == "hydrostatic_rmse" and (row["date"], row["lead_time_hours"]) in updates:
            return updates[(row["date"], row["lead_time_hours"])][1]
        return row["ref_value"]

    df["model_value"] = df.apply(patch_model, axis=1)
    df["ref_value"] = df.apply(patch_ref, axis=1)

    backup_path = csv_path.with_suffix(".csv.bak")
    if not backup_path.exists():
        import shutil
        shutil.copy(csv_path, backup_path)
        print(f"Created backup at {backup_path}")

    df.to_csv(csv_path, index=False)
    print(f"Saved updated CSV to {csv_path}")


if __name__ == "__main__":
    main()