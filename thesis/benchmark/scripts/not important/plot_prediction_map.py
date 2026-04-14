#!/usr/bin/env python3
"""
Plot a global/regional map of a single prediction vs ERA5 at a given lead time.

Usage:
    python plot_prediction_map.py --model aurora --date 2022-01-01 --lead-hours 240 --variable wind500
    python plot_prediction_map.py --model aurora --date 2022-01-01 --lead-hours 240 --variable z500

For wind500, three panels are shown: ERA5 | Aurora | Pangu (all at the same valid time).
For other variables, three panels are: ERA5 | primary model | difference.

Reads from WeatherBench2 zarr stores anonymously.
"""

import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

MODEL_ZARRS = {
    "aurora":     "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr",
    "pangu":      "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
    "graphcast":  "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr",
    "hres":       "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    "neuralgcm":  "gs://weatherbench2/datasets/neuralgcm_deterministic/2020-512x256.zarr",
    "fuxi":       "gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr",
}

# Map shorthand variable names to WB2 variable names and optional level.
# wind500 is a derived variable: sqrt(u² + v²) at 500 hPa.
VAR_CONFIG = {
    "z500":    {"name": "geopotential",          "level": 500,  "label": "Z500 (m²/s²)", "cmap": "RdBu_r"},
    "t850":    {"name": "temperature",            "level": 850,  "label": "T850 (K)",      "cmap": "RdBu_r"},
    "u500":    {"name": "u_component_of_wind",    "level": 500,  "label": "U500 (m/s)",    "cmap": "RdBu_r"},
    "msl":     {"name": "mean_sea_level_pressure","level": None, "label": "MSLP (Pa)",     "cmap": "RdBu_r"},
    "t2m":     {"name": "2m_temperature",         "level": None, "label": "T2m (K)",       "cmap": "RdBu_r"},
    "wind500": {"name": "wind500",                "level": 500,  "label": "Wind speed 500 hPa (m/s)", "cmap": "viridis"},
}


def open_zarr(url: str) -> xr.Dataset:
    ds = xr.open_zarr(url, storage_options={"token": "anon"}, consolidated=True)
    # Strip whitespace/tabs from variable names (some zarr stores have trailing tabs)
    rename_map = {v: v.strip() for v in ds.data_vars if v != v.strip()}
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def extract_field(ds: xr.Dataset, var_cfg: dict, init_time, lead_hours: int | None = None) -> np.ndarray:
    """Extract a 2D field from a zarr dataset at a given init_time + lead.

    For the synthetic 'wind500' variable, computes sqrt(u² + v²) at 500 hPa.
    """
    level = var_cfg["level"]

    def _sel_da(vname):
        if lead_hours is not None:
            lead_td = np.timedelta64(lead_hours, "h")
            for dim in ("prediction_timedelta", "lead_time", "step"):
                if dim in ds.dims:
                    da = ds[vname].sel({dim: lead_td}, method="nearest")
                    for tdim in ("time", "init_time", "initial_time"):
                        if tdim in da.dims:
                            da = da.sel({tdim: init_time}, method="nearest")
                            break
                    break
            else:
                valid_time = pd.Timestamp(init_time) + pd.Timedelta(hours=lead_hours)
                da = ds[vname].sel(time=valid_time, method="nearest")
        else:
            da = ds[vname].sel(time=init_time, method="nearest")

        if level is not None and "level" in da.dims:
            da = da.sel(level=level, method="nearest")
        return da

    if var_cfg["name"] == "wind500":
        u = _sel_da("u_component_of_wind").values.squeeze().astype(float)
        v = _sel_da("v_component_of_wind").values.squeeze().astype(float)
        field = np.sqrt(u ** 2 + v ** 2)
    else:
        field = _sel_da(var_cfg["name"]).values.squeeze()

    return field, ds.latitude.values, ds.longitude.values


def _regrid_to(field: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray,
               tgt_lat: np.ndarray, tgt_lon: np.ndarray) -> np.ndarray:
    """Bilinearly regrid *field* from src grid to tgt grid (both regular lat/lon)."""
    from scipy.interpolate import RegularGridInterpolator
    lat = src_lat.copy()
    fld = field.copy()
    if lat[0] > lat[-1]:          # descending → ascending
        lat = lat[::-1]
        fld = fld[::-1]
    interp = RegularGridInterpolator(
        (lat, src_lon % 360), fld, method="linear", bounds_error=False, fill_value=None,
    )
    lat_g, lon_g = np.meshgrid(tgt_lat, tgt_lon % 360, indexing="ij")
    return interp((lat_g, lon_g))


def plot_three_models(
    fields: list[np.ndarray],
    titles: list[str],
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    label: str,
    cmap: str,
    suptitle: str,
    output: Path,
    dpi: int = 200,
):
    """Plot three fields side-by-side with a shared colorbar range."""
    vmin = np.nanpercentile(fields[0], 2)   # anchored to ERA5
    vmax = np.nanpercentile(fields[0], 98)

    proj = ccrs.PlateCarree()
    lons2d, lats2d = np.meshgrid(lon, lat)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), subplot_kw={"projection": proj})

    for ax, field, title in zip(axes, fields, titles):
        im = ax.pcolormesh(lons2d, lats2d, field, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto", transform=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_global()
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046, aspect=40,
                     label=label)

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def plot_comparison(
    pred: np.ndarray,
    era5: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    label: str,
    cmap: str,
    model: str,
    date_str: str,
    lead_hours: int,
    output: Path,
    dpi: int = 200,
):
    diff = pred - era5
    vmax_diff = np.nanpercentile(np.abs(diff), 98)
    vmin = np.nanpercentile(era5, 2)
    vmax = np.nanpercentile(era5, 98)

    proj = ccrs.PlateCarree()
    lons2d, lats2d = np.meshgrid(lon, lat)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), subplot_kw={"projection": proj})

    titles = [f"ERA5 (valid: +{lead_hours}h)", f"{model.upper()} (valid: +{lead_hours}h)", f"Difference ({model} − ERA5)"]
    fields = [era5, pred, diff]
    vmins  = [vmin, vmin, -vmax_diff]
    vmaxs  = [vmax, vmax,  vmax_diff]
    cmaps  = [cmap, cmap, "RdBu_r"]

    for ax, field, title, vm, vM, cm in zip(axes, fields, titles, vmins, vmaxs, cmaps):
        im = ax.pcolormesh(lons2d, lats2d, field, cmap=cm, vmin=vm, vmax=vM,
                           shading="auto", transform=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_global()
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046, aspect=40,
                     label=label)

    fig.suptitle(f"{label}  |  Init: {date_str}  |  Lead: {lead_hours}h ({lead_hours//24}d)", fontsize=13)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot prediction vs ERA5 at a given lead time")
    parser.add_argument("--model",      type=str, default="aurora", choices=list(MODEL_ZARRS))
    parser.add_argument("--model-zarr", type=str, default=None, help="Override model zarr URL")
    parser.add_argument("--era5-zarr",  type=str, default=ERA5_ZARR)
    parser.add_argument("--date",       type=str, default="2022-01-01", help="Init date YYYY-MM-DD")
    parser.add_argument("--lead-hours", type=int, default=240)
    parser.add_argument("--variable",   type=str, default="wind500", choices=list(VAR_CONFIG))
    parser.add_argument("--output",     type=str, default=None)
    parser.add_argument("--dpi",        type=int, default=200)
    args = parser.parse_args()

    var_cfg = VAR_CONFIG[args.variable]
    model_zarr = args.model_zarr or MODEL_ZARRS[args.model]

    output = Path(args.output) if args.output else (
        RESULTS_DIR / "plots_maps" /
        f"map_{args.model}_{args.variable}_lead{args.lead_hours}h_{args.date}.png"
    )

    init_time  = np.datetime64(args.date)
    valid_time = pd.Timestamp(args.date) + pd.Timedelta(hours=args.lead_hours)

    print(f"Loading ERA5 zarr...")
    ds_era5 = open_zarr(args.era5_zarr)
    era5_field, era5_lat, era5_lon = extract_field(ds_era5, var_cfg, valid_time, lead_hours=None)

    print(f"Loading {args.model} zarr...")
    ds_model = open_zarr(model_zarr)
    pred, lat, lon = extract_field(ds_model, var_cfg, init_time, args.lead_hours)

    if era5_field.shape != pred.shape:
        era5_field = _regrid_to(era5_field, era5_lat, era5_lon, lat, lon)

    suptitle = f"{var_cfg['label']}  |  Init: {args.date}  |  Lead: {args.lead_hours}h ({args.lead_hours//24}d)"

    if args.variable == "wind500":
        # Third panel: Pangu prediction at the same lead time
        pangu_zarr = MODEL_ZARRS["pangu"]
        print(f"Loading pangu zarr for comparison panel...")
        ds_pangu = open_zarr(pangu_zarr)
        pangu_field, pangu_lat, pangu_lon = extract_field(ds_pangu, var_cfg, init_time, args.lead_hours)
        if pangu_field.shape != pred.shape:
            pangu_field = _regrid_to(pangu_field, pangu_lat, pangu_lon, lat, lon)

        lead_str = f"+{args.lead_hours}h"
        plot_three_models(
            [era5_field, pred, pangu_field],
            [f"ERA5 (valid: {lead_str})", f"Aurora (valid: {lead_str})", f"Pangu (valid: {lead_str})"],
            lat, lon,
            label=var_cfg["label"],
            cmap=var_cfg["cmap"],
            suptitle=suptitle,
            output=output,
            dpi=args.dpi,
        )
    else:
        plot_comparison(
            pred, era5_field, lat, lon,
            label=var_cfg["label"],
            cmap=var_cfg["cmap"],
            model=args.model,
            date_str=args.date,
            lead_hours=args.lead_hours,
            output=output,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
