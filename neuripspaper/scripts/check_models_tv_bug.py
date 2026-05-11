import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import sys
import argparse

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

def open_zarr_anonymous(url: str) -> xr.Dataset:
    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    rename = {v: v.strip() for v in ds.data_vars if v != v.strip()}
    rename.update({d: d.strip() for d in ds.dims if d != d.strip()})
    if "lat" in ds.dims and "latitude" not in ds.dims: rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims: rename["lon"] = "longitude"
    if rename: ds = ds.rename(rename)
    return ds

def _apply_virtual_temperature(ds: xr.Dataset) -> xr.Dataset:
    t_var = _find_var(ds, T_NAMES)
    q_var = _find_var(ds, Q_NAMES)
    if t_var and q_var:
        ds = ds.copy()
        ds[t_var] = ds[t_var] * (1.0 + 0.608 * ds[q_var])
    return ds

def check_model(model_name: str, zarr_path: str, csv_path: Path, area: xr.DataArray):
    if not csv_path.exists():
        print(f"[{model_name}] CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    hydro_df = df[df["metric_name"] == "hydrostatic_rmse"]
    if hydro_df.empty: return
    
    # Take the very first row to test
    row = hydro_df.iloc[0]
    date_str = row["date"]
    lead_h = int(row["lead_time_hours"])
    old_val = row["model_value"]

    print(f"\n--- Testing {model_name} for {date_str} + {lead_h}h ---")
    print(f"Old CSV value: {old_val}")

    ds_model = open_zarr_anonymous(zarr_path)
    init_time = np.datetime64(date_str, "ns")
    lead_td = np.timedelta64(lead_h, "h")

    snap = ds_model.sel(time=init_time)
    pred_dim = _detect_pred_td_dim(snap)
    if pred_dim: snap = snap.sel({pred_dim: lead_td}, method="nearest")
    if "time" in snap.dims: snap = snap.isel(time=0)
    snap = snap.load()

    area_model = area
    if "latitude" in snap.dims and snap.sizes["latitude"] != area.sizes.get("latitude"):
        area_model = get_grid_cell_area(snap)

    ld = _detect_level_dim(snap)
    has_q = _find_var(snap, Q_NAMES) is not None
    print(f"Model outputs specific humidity (Q)? {'Yes' if has_q else 'No'}")

    snap_tv = _apply_virtual_temperature(snap)
    new_val = compute_hydrostatic_imbalance(snap_tv, area_model, level_dim=ld)

    print(f"New Tv value : {new_val}")
    diff = abs(new_val - old_val)
    print(f"Difference   : {diff:.6f} -> {'BUG PRESENT (Needs update)' if diff > 1e-4 else 'ALL GOOD (Already used Tv)'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--zarr", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    base = Path("/home/ekasteleyn/aurora_thesis/neuripspaper")
    era5_zarr = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
    ds_ref = open_zarr_anonymous(era5_zarr)
    area = get_grid_cell_area(ds_ref.isel(time=0, drop=True))

    csv_path = Path(args.csv)
    try:
        check_model(args.model, args.zarr, csv_path, area)
    except Exception as e:
        print(f"Error checking {args.model}: {e}")

if __name__ == "__main__":
    main()
