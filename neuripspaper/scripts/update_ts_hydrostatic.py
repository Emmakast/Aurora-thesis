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
    _detect_level_dim,
    _detect_pred_td_dim,
)

_WORKER_MODEL_DS = None
_WORKER_INIT_ARGS = None 

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
    global _WORKER_MODEL_DS, _WORKER_INIT_ARGS
    if _WORKER_MODEL_DS is None:
        model_url = _WORKER_INIT_ARGS
        print(f"Worker opening zarr connection...", flush=True)
        _WORKER_MODEL_DS = open_zarr_anonymous(model_url)
    return _WORKER_MODEL_DS

def _init_worker(model_url: str):
    global _WORKER_INIT_ARGS
    _WORKER_INIT_ARGS = model_url

def _align_area(area: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    lat_dim = next((d for d in ds.dims if d in area.dims and "lat" in d.lower()), None)
    lon_dim = next((d for d in ds.dims if d in area.dims and "lon" in d.lower()), None)
    if lat_dim is None or lon_dim is None:
        return area
    if area.dims != (lat_dim, lon_dim):
        area = area.transpose(lat_dim, lon_dim)
    return area

def _evaluate_ts_hydrostatic(date_str: str, lead_hours: int):
    ds_model_full = _get_worker_datasets()
    init_time = np.datetime64(date_str, "ns")
    lead_td = np.timedelta64(int(lead_hours), "h")

    hydro_model = np.nan
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
        area_model = _align_area(area_model, ds_model_t)

        hydro_model = compute_hydrostatic_imbalance(ds_model_t, area_model, level_dim=ld_model)
    except Exception as e:
        print(f"  [!] Model failed for {date_str} + {lead_hours}h: {e}", flush=True)

    return date_str, lead_hours, hydro_model

def main():
    parser = argparse.ArgumentParser("Update hydrostatic values in a timeseries CSV.")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--prediction-zarr", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if "hydrostatic_rmse" not in df.columns:
        df["hydrostatic_rmse"] = np.nan

    tasks = df[["date", "forecast_hour"]].drop_duplicates()
    print(f"Found {len(tasks)} time series entries to update.")

    updates = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.prediction_zarr,),
    ) as pool:
        futures = {
            pool.submit(_evaluate_ts_hydrostatic, r.date, r.forecast_hour): (r.date, r.forecast_hour)
            for _, r in tasks.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            date_str, lead_h = futures[fut]
            try:
                _, _, h_mod = fut.result()
                updates[(date_str, lead_h)] = h_mod
                print(f"[{i}/{len(tasks)}] {date_str} + {lead_h}h -> Model: {h_mod:.4f}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(tasks)}] Task failed for {date_str} + {lead_h}h: {e}", flush=True)

    def patch_model(row):
        val = updates.get((row["date"], row["forecast_hour"]))
        return val if val is not None else row["hydrostatic_rmse"]

    df["hydrostatic_rmse"] = df.apply(patch_model, axis=1)

    backup_path = csv_path.with_suffix(".csv.bak")
    if not backup_path.exists():
        import shutil
        shutil.copy(csv_path, backup_path)
        print(f"Created backup at {backup_path}")

    df.to_csv(csv_path, index=False)
    print(f"Saved updated CSV to {csv_path}")

if __name__ == "__main__":
    main()
