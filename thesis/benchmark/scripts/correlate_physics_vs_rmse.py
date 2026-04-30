#!/usr/bin/env python3
"""
Correlate Physical Consistency Metrics vs Standard Forecast RMSE.

This script compares daily physical metrics (computed locally) against
daily Z500/T850 RMSE computed from WeatherBench2 forecast and ERA5 datasets.

Since WB2 benchmark_results are pre-aggregated (no daily dimension), we compute
daily RMSE directly from the forecast Zarr stores vs ERA5.

Usage:
    python correlate_physics_vs_rmse.py
    python correlate_physics_vs_rmse.py --models pangu fuxi --lead-times 12 120
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# Force unbuffered output for logging
def _print(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ============================================================================
# Configuration
# ============================================================================

PHYSICS_RESULTS_DIR = Path("/home/ekasteleyn/aurora_thesis/neuripspaper/results")
RMSE_RESULTS_DIR = Path("/home/ekasteleyn/aurora_thesis/thesis/benchmark/results")
PLOTS_DIR = RMSE_RESULTS_DIR / "plots_correlation"

# Model configurations: prediction Zarr, ERA5 Zarr (for grid matching)
MODEL_CONFIG = {
    "pangu": {
        "pred_zarr": "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
        "era5_zarr": "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    },
    "fuxi": {
        "pred_zarr": "gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr",
        "era5_zarr": "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    },
    "graphcast": {
        "pred_zarr": "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr",
        "era5_zarr": "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    },
    "hres": {
        "pred_zarr": "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        "era5_zarr": "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    },
    "neuralgcm": {
        "pred_zarr": "gs://weatherbench2/datasets/neuralgcm_deterministic/2020-512x256.zarr",
        "era5_zarr": "gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr",
    },
}

# Lead times to analyze (hours)
TARGET_LEAD_TIMES = [12, 120, 240]

# Physical metrics to correlate
PHYSICAL_METRICS = [
    "geostrophic_rmse",
    "hydrostatic_rmse",
    "effective_resolution_km",
    "spectral_divergence",
    "spectral_residual",
    "dry_mass_drift_pct_per_day",
    "water_mass_drift_pct_per_day",
    "total_energy_drift_pct_per_day",
]

METRIC_LABELS = {
    "geostrophic_rmse": "Geostrophic RMSE",
    "hydrostatic_rmse": "Hydrostatic RMSE",
    "effective_resolution_km": "Effective Resolution",
    "spectral_divergence": "Spectral Divergence",
    "spectral_residual": "Spectral Residual",
    "dry_mass_drift_pct_per_day": "Dry Air Mass",
    "water_mass_drift_pct_per_day": "Water Mass",
    "total_energy_drift_pct_per_day": "Total Energy",
}


# ============================================================================
# Data Loading
# ============================================================================

def open_zarr_anonymous(url: str) -> xr.Dataset:
    """Open a public GCS Zarr store without authentication."""
    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    # Normalise dimension names
    rename = {}
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def load_physics_metrics(model: str, year: int = 2020) -> pd.DataFrame:
    """Load local physics evaluation CSV."""
    csv_path = PHYSICS_RESULTS_DIR / f"physics_evaluation_{model}_{year}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Physics CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Filter to target lead times
    df = df[df["lead_time_hours"].isin(TARGET_LEAD_TIMES)]
    
    # Pivot to wide format: one row per (date, lead_time), columns = metrics
    df_wide = df.pivot_table(
        index=["date", "lead_time_hours"],
        columns="metric_name",
        values="model_value",
        aggfunc="first"
    ).reset_index()
    
    return df_wide


def compute_daily_rmse(
    model: str,
    dates: list[str],
    lead_hours: list[int],
    variables: dict[str, int],  # {var_name: level}
) -> pd.DataFrame:
    """
    Compute daily RMSE for specified variables/levels from WB2 Zarr.
    
    Parameters
    ----------
    model : str
        Model name (key in MODEL_CONFIG)
    dates : list[str]
        ISO date strings to evaluate
    lead_hours : list[int]
        Lead times in hours
    variables : dict
        Mapping of variable name to pressure level (e.g., {"geopotential": 500})
    
    Returns
    -------
    pd.DataFrame
        Columns: date, lead_time_hours, z500_rmse, t850_rmse, ...
    """
    config = MODEL_CONFIG[model]
    
    _print(f"  Opening prediction Zarr: {config['pred_zarr']}")
    ds_pred = open_zarr_anonymous(config["pred_zarr"])
    
    _print(f"  Opening ERA5 Zarr: {config['era5_zarr']}")
    ds_era5 = open_zarr_anonymous(config["era5_zarr"])
    
    # Detect prediction_timedelta dimension
    pred_td_dim = None
    for dim in ["prediction_timedelta", "lead_time", "step"]:
        if dim in ds_pred.dims:
            pred_td_dim = dim
            break
    
    if pred_td_dim is None:
        raise ValueError(f"Cannot find prediction timedelta dimension in {config['pred_zarr']}")
    
    # Detect level dimension
    level_dim = None
    for dim in ["level", "pressure_level", "isobaricInhPa"]:
        if dim in ds_pred.dims or dim in ds_pred.coords:
            level_dim = dim
            break
    
    results = []
    
    for date_str in dates:
        init_time = np.datetime64(date_str, "ns")
        
        for lead_h in lead_hours:
            lead_td = np.timedelta64(lead_h, "h")
            valid_time = init_time + lead_td
            
            row = {"date": date_str, "lead_time_hours": lead_h}
            
            try:
                # Select prediction slice
                ds_p = ds_pred.sel(time=init_time)
                if pred_td_dim in ds_p.dims:
                    ds_p = ds_p.sel({pred_td_dim: lead_td}, method="nearest")
                
                # Select ERA5 at valid time
                ds_e = ds_era5.sel(time=valid_time)
                
                # Compute RMSE for each variable
                for var_name, level in variables.items():
                    rmse_key = f"{var_name}_{level}_rmse"
                    
                    # Get prediction variable
                    pred_var = None
                    for name in [var_name, var_name.replace("_", " ")]:
                        if name in ds_p.data_vars:
                            pred_var = ds_p[name]
                            break
                    
                    if pred_var is None:
                        row[rmse_key] = np.nan
                        continue
                    
                    # Select level if applicable
                    if level_dim and level_dim in pred_var.dims:
                        pred_var = pred_var.sel({level_dim: level})
                    
                    # Get ERA5 variable
                    era5_var = None
                    for name in [var_name, var_name.replace("_", " ")]:
                        if name in ds_e.data_vars:
                            era5_var = ds_e[name]
                            break
                    
                    if era5_var is None:
                        row[rmse_key] = np.nan
                        continue
                    
                    # Select level
                    era5_level_dim = None
                    for dim in ["level", "pressure_level", "isobaricInhPa"]:
                        if dim in era5_var.dims:
                            era5_level_dim = dim
                            break
                    if era5_level_dim:
                        era5_var = era5_var.sel({era5_level_dim: level})
                    
                    # Load and compute RMSE
                    pred_vals = pred_var.load().values
                    era5_vals = era5_var.load().values
                    
                    # Handle grid mismatch (interpolate if needed)
                    if pred_vals.shape != era5_vals.shape:
                        # Simple: compute RMSE on overlapping region
                        min_shape = tuple(min(p, e) for p, e in zip(pred_vals.shape, era5_vals.shape))
                        pred_vals = pred_vals[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else pred_vals
                        era5_vals = era5_vals[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else era5_vals
                    
                    rmse = np.sqrt(np.nanmean((pred_vals - era5_vals) ** 2))
                    row[rmse_key] = rmse
                
                results.append(row)
                
            except Exception as e:
                pass # suppress printing errors for every date
    
    return pd.DataFrame(results)


def load_cached_rmse(model: str, year: int = 2020) -> Optional[pd.DataFrame]:
    """Load cached daily RMSE from CSV if available."""
    cache_path = RMSE_RESULTS_DIR / f"daily_rmse_{model}_{year}.csv"
    if cache_path.exists():
        _print(f"  Loading cached RMSE from {cache_path}")
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df
    return None


def save_cached_rmse(df: pd.DataFrame, model: str, year: int = 2020) -> None:
    """Save daily RMSE to CSV for caching."""
    cache_path = RMSE_RESULTS_DIR / f"daily_rmse_{model}_{year}.csv"
    df.to_csv(cache_path, index=False)
    _print(f"  Saved RMSE cache to {cache_path}")


def compute_daily_rmse_fast(
    model: str,
    dates: list[str],
    lead_hours: list[int],
    use_cache: bool = True,
    year: int = 2020,
) -> pd.DataFrame:
    """
    Compute daily Z500 and T850 RMSE using vectorized operations.
    
    This is a faster implementation that loads data in batches.
    Caches results to CSV for faster subsequent runs.
    """
    # Check cache first
    if use_cache:
        cached = load_cached_rmse(model, year)
        if cached is not None:
            # Filter to requested dates and lead_hours
            filtered = cached[
                (cached["date"].isin(dates)) &
                (cached["lead_time_hours"].isin(lead_hours))
            ]
            if len(filtered) > 0:
                _print(f"  Using {len(filtered)} cached RMSE entries")
                return filtered
    
    config = MODEL_CONFIG[model]
    
    _print(f"  Loading datasets for {model}...")
    ds_pred = open_zarr_anonymous(config["pred_zarr"])
    ds_era5 = open_zarr_anonymous(config["era5_zarr"])
    
    # Detect dimensions
    pred_td_dim = next((d for d in ["prediction_timedelta", "lead_time", "step"] if d in ds_pred.dims), None)
    level_dim_pred = next((d for d in ["level", "pressure_level"] if d in ds_pred.dims or d in ds_pred.coords), None)
    level_dim_era5 = next((d for d in ["level", "pressure_level"] if d in ds_era5.dims or d in ds_era5.coords), None)
    
    # Find variable names
    z_name = next((v for v in ds_pred.data_vars if "geopotential" in v.lower()), None)
    t_name = next((v for v in ds_pred.data_vars if "temperature" in v.lower() and "2m" not in v.lower()), None)
    
    _print(f"    Variables: Z={z_name}, T={t_name}")
    _print(f"    Pred TD dim: {pred_td_dim}, Level dim: {level_dim_pred}")
    
    results = []
    
    for i, date_str in enumerate(dates):
        if i % 10 == 0:
            _print(f"    Processing date {i+1}/{len(dates)}: {date_str}")
        init_time = np.datetime64(date_str, "ns")
        
        for lead_h in lead_hours:
            lead_td = np.timedelta64(lead_h, "h")
            valid_time = init_time + lead_td
            
            row = {"date": date_str, "lead_time_hours": lead_h}
            
            try:
                # Select slices
                ds_p = ds_pred.sel(time=init_time)
                if pred_td_dim and pred_td_dim in ds_p.dims:
                    ds_p = ds_p.sel({pred_td_dim: lead_td}, method="nearest")
                    # Check if nearest match is actually the requested lead time
                    actual_td = ds_p.coords.get(pred_td_dim)
                    if actual_td is not None:
                        actual_h = int(actual_td.values / np.timedelta64(1, "h"))
                        if abs(actual_h - lead_h) > 6:  # Allow 6h tolerance
                            row["z500_rmse"] = np.nan
                            row["t850_rmse"] = np.nan
                            results.append(row)
                            continue
                
                ds_e = ds_era5.sel(time=valid_time)
                
                # Z500 RMSE
                if z_name:
                    z_pred = ds_p[z_name]
                    z_era5 = ds_e["geopotential"] if "geopotential" in ds_e else None
                    
                    if z_era5 is not None:
                        if level_dim_pred and level_dim_pred in z_pred.dims:
                            z_pred = z_pred.sel({level_dim_pred: 500})
                        if level_dim_era5 and level_dim_era5 in z_era5.dims:
                            z_era5 = z_era5.sel({level_dim_era5: 500})
                        
                        z_pred_vals = z_pred.load().values.flatten()
                        z_era5_vals = z_era5.load().values.flatten()
                        
                        # Align sizes
                        min_len = min(len(z_pred_vals), len(z_era5_vals))
                        row["z500_rmse"] = np.sqrt(np.nanmean(
                            (z_pred_vals[:min_len] - z_era5_vals[:min_len]) ** 2
                        ))
                    else:
                        row["z500_rmse"] = np.nan
                else:
                    row["z500_rmse"] = np.nan
                
                # T850 RMSE
                if t_name:
                    t_pred = ds_p[t_name]
                    t_era5 = ds_e["temperature"] if "temperature" in ds_e else None
                    
                    if t_era5 is not None:
                        if level_dim_pred and level_dim_pred in t_pred.dims:
                            t_pred = t_pred.sel({level_dim_pred: 850})
                        if level_dim_era5 and level_dim_era5 in t_era5.dims:
                            t_era5 = t_era5.sel({level_dim_era5: 850})
                        
                        t_pred_vals = t_pred.load().values.flatten()
                        t_era5_vals = t_era5.load().values.flatten()
                        
                        min_len = min(len(t_pred_vals), len(t_era5_vals))
                        row["t850_rmse"] = np.sqrt(np.nanmean(
                            (t_pred_vals[:min_len] - t_era5_vals[:min_len]) ** 2
                        ))
                    else:
                        row["t850_rmse"] = np.nan
                else:
                    row["t850_rmse"] = np.nan
                
            except Exception as e:
                _print(f"    ⚠ {date_str} lead={lead_h}h: {e}")
                row["z500_rmse"] = np.nan
                row["t850_rmse"] = np.nan
            
            results.append(row)
    
    df_results = pd.DataFrame(results)
    
    # Cache the results
    if use_cache and len(df_results) > 0:
        save_cached_rmse(df_results, model, year)
    
    return df_results


# ============================================================================
# Correlation Analysis
# ============================================================================

def compute_correlations(
    df: pd.DataFrame,
    physics_metrics: list[str],
    rmse_cols: list[str] = ["z500_rmse", "t850_rmse"],
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations between physics metrics and RMSE.
    
    Returns a DataFrame with columns:
        lead_time_hours, metric, rmse_type, pearson_r, pearson_p, spearman_r, spearman_p
    """
    results = []
    
    for lead_h in df["lead_time_hours"].unique():
        df_lead = df[df["lead_time_hours"] == lead_h].dropna()
        
        for metric in physics_metrics:
            if metric not in df_lead.columns:
                continue
            
            for rmse_col in rmse_cols:
                if rmse_col not in df_lead.columns:
                    continue
                
                # Drop rows with NaN in either column
                valid = df_lead[[metric, rmse_col]].dropna()
                if len(valid) < 3:  # Need at least 3 points for correlation
                    continue
                
                x = valid[metric].values
                y = valid[rmse_col].values
                
                # Pearson
                pearson_r, pearson_p = stats.pearsonr(x, y)
                
                # Spearman
                spearman_r, spearman_p = stats.spearmanr(x, y)
                
                results.append({
                    "lead_time_hours": lead_h,
                    "metric": metric,
                    "rmse_type": rmse_col,
                    "pearson_r": pearson_r,
                    "pearson_r2": pearson_r ** 2,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "n_samples": len(valid),
                })
    
    return pd.DataFrame(results)


# ============================================================================
# Visualization
# ============================================================================

def plot_correlation_table(corr_df: pd.DataFrame, model: str, outdir: Path, lead_times: list[int]):
    """Draw a color-coded Matplotlib table and print the LaTeX version for Correlative summaries."""
    if corr_df.empty: return
    
    header_color = np.array([0.9, 0.9, 0.9])
    red = np.array([1.0, 0.75, 0.75])
    white = np.array([1.0, 1.0, 1.0])

    cols = ["Metric", "Statistic"]
    for lh in lead_times:
        cols.extend([f"{lh}h Z500", f"{lh}h T850"])

    cell_texts = [cols]
    cell_colors = [[header_color] * len(cols)]

    latex_lines = [
        "\\begin{table}[H]",
        "    \\centering",
        f"    \\caption{{Coefficient of determination ($R^2$) and Spearman's rank correlation ($\\rho$) between standard predictive error (RMSE) and physical consistency metrics for {model.upper()}.}}",
        "    \\label{tab:rmse_correlation_" + model + "}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{@{}ll" + "rr" * len(lead_times) + "@{}}",
        "        \\toprule",
        "        \\multirow{2}{*}{\\textbf{Metric}} & \\multirow{2}{*}{\\textbf{Statistic}} & " + 
        " & ".join([f"\\multicolumn{{2}}{{c}}{{\\textbf{{{lh}h}}}}" for lh in lead_times]) + " \\\\",
        "        " + " ".join([f"\\cmidrule(lr){{{3+2*i}-{4+2*i}}}" for i in range(len(lead_times))]),
        "        & & " + " & ".join(["\\textbf{Z500} & \\textbf{T850}" for _ in lead_times]) + " \\\\",
        "        \\midrule"
    ]

    for m_idx, metric in enumerate(PHYSICAL_METRICS):
        label = METRIC_LABELS.get(metric, metric)
        row_r2_t = [label, "$R^2$"]
        row_rho_t = [label, "$\\rho$"]
        row_r2_c = [white, white.copy()]
        row_rho_c = [white, white.copy()]
        
        latex_r2 = f"        \\multirow{{2}}{{*}}{{{label}}} & $R^2$"
        latex_rho = f"         & $\\rho$"

        for lh in lead_times:
            for rmse_t in ["z500_rmse", "t850_rmse"]:
                sub = corr_df[(corr_df["metric"] == metric) & (corr_df["lead_time_hours"] == lh) & (corr_df["rmse_type"] == rmse_t)]
                if not sub.empty:
                    val_r2 = sub["pearson_r2"].values[0]
                    val_rho = sub["spearman_r"].values[0]
                    
                    row_r2_t.append(f"{val_r2:.3f}")
                    row_rho_t.append(f"{val_rho:.3f}")
                    latex_r2 += f" & {val_r2:.3f}"
                    latex_rho += f" & {val_rho:.3f}"
                    
                    intensity_r2 = min(abs(val_r2), 1.0) * 0.9
                    intensity_rho = min(abs(val_rho), 1.0) * 0.7
                    row_r2_c.append(white * (1 - intensity_r2) + red * intensity_r2)
                    row_rho_c.append(white * (1 - intensity_rho) + red * intensity_rho)
                else:
                    for r, l_ref in zip([row_r2_t, row_rho_t], [latex_r2, latex_rho]):
                        r.append("—")
                    latex_r2 += " & —"
                    latex_rho += " & —"
                    row_r2_c.append(white.copy())
                    row_rho_c.append(white.copy())

        latex_lines.append(latex_r2 + " \\\\")
        latex_lines.append(latex_rho + " \\\\ \\addlinespace")

        cell_texts.extend([row_r2_t, row_rho_t])
        cell_colors.extend([row_r2_c, row_rho_c])
        
    latex_lines[-1] = latex_lines[-1].replace(" \\addlinespace", "")
    latex_lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}%",
        "    }",
        "\\end{table}"
    ])
    
    print("\n" + "\n".join(latex_lines) + "\n")

    fig, ax = plt.subplots(figsize=(max(12, 1.4 * len(cols)), len(cell_texts) * 0.45))
    ax.axis("off")
    
    # ── Allocate more space to the first metric column ──
    col_widths = [0.35, 0.15] + [0.12] * (len(cols) - 2)
    
    table = ax.table(
        cellText=cell_texts,
        cellColours=[[tuple(c) for c in row] for row in cell_colors],
        colWidths=col_widths,
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 1.6)

    for j in range(len(cols)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor(tuple(header_color))

    # Cell borders logic to mimic multi-row
    for r in range(1, len(cell_texts), 2):
        table[r, 0].visible_edges = 'LRT'
        table[r+1, 0].visible_edges = 'LRB'
        table[r+1, 0].get_text().set_text("")

    out_file = outdir / f"correlation_table_{model}.png"
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _print(f"  Saved image table to: {out_file}")

def plot_scatter_grid(
    df: pd.DataFrame,
    model: str,
    physics_metrics: list[str],
    lead_times: list[int],
    rmse_col: str = "z500_rmse",
    corr_df: Optional[pd.DataFrame] = None,
):
    """
    Create a grid of scatter plots: rows = metrics, columns = lead times.
    """
    n_metrics = len(physics_metrics)
    n_leads = len(lead_times)
    
    fig, axes = plt.subplots(
        n_metrics, n_leads,
        figsize=(4 * n_leads, 3 * n_metrics),
        squeeze=False,
    )
    
    for i, metric in enumerate(physics_metrics):
        for j, lead_h in enumerate(lead_times):
            ax = axes[i, j]
            
            # Filter data
            df_sub = df[df["lead_time_hours"] == lead_h].dropna(subset=[metric, rmse_col])
            
            if len(df_sub) < 5:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel("")
                ax.set_ylabel("")
                continue
            
            # Scatter plot
            if "model" in df_sub.columns and df_sub["model"].nunique() > 1:
                sns.scatterplot(
                    data=df_sub,
                    x=rmse_col,
                    y=metric,
                    hue="model",
                    alpha=0.6,
                    s=20,
                    ax=ax,
                    legend=False # Disable legend inside subplots to avoid clutter
                )
            else:
                sns.scatterplot(
                    data=df_sub,
                    x=rmse_col,
                    y=metric,
                    alpha=0.5,
                    s=20,
                    ax=ax,
                )
            
            # Add correlation annotation
            if corr_df is not None:
                corr_row = corr_df[
                    (corr_df["lead_time_hours"] == lead_h) &
                    (corr_df["metric"] == metric) &
                    (corr_df["rmse_type"] == rmse_col)
                ]
                if len(corr_row) > 0:
                    r2 = corr_row["pearson_r2"].values[0]
                    rho = corr_row["spearman_r"].values[0]
                    ax.text(
                        0.05, 0.95,
                        f"R²={r2:.3f}\nρ={rho:.3f}",
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )
            
            # Labels
            if i == 0:
                ax.set_title(f"Lead = {lead_h}h", fontsize=11)
            if j == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
            else:
                ax.set_ylabel("")
            if i == n_metrics - 1:
                ax.set_xlabel("Z500 RMSE (m²/s²)" if "z500" in rmse_col else "T850 RMSE (K)", fontsize=9)
            else:
                ax.set_xlabel("")
    
    fig.suptitle(f"{model.upper()}: Physical Metrics vs Forecast RMSE", fontsize=14, y=1.01)
    fig.tight_layout()
    
    return fig


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Correlate physics metrics with forecast RMSE")
    parser.add_argument(
        "--models", nargs="+", default=["pangu", "fuxi", "graphcast", "hres", "neuralgcm"],
        help="Models to analyze",
    )
    parser.add_argument(
        "--lead-times", nargs="+", type=int, default=TARGET_LEAD_TIMES,
        help="Lead times in hours",
    )
    parser.add_argument(
        "--year", type=int, default=2020,
        help="Year to analyze",
    )
    parser.add_argument(
        "--max-dates", type=int, default=None,
        help="Limit number of dates for testing",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force recompute RMSE (ignore cached values)",
    )
    args = parser.parse_args()
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    all_correlations = []
    all_merged_dfs = []  # List to store merged dataframes from all models
    
    for model in args.models:
        _print(f"\n{'='*60}")
        _print(f"  Processing: {model}")
        _print(f"{'='*60}")
        
        # Check if model is supported
        if model not in MODEL_CONFIG:
            _print(f"  ⚠ Model '{model}' not in MODEL_CONFIG, skipping.")
            continue
        
        # Load physics metrics
        try:
            df_physics = load_physics_metrics(model, args.year)
            _print(f"  Loaded {len(df_physics)} physics metric rows")
        except FileNotFoundError as e:
            _print(f"  ⚠ {e}")
            continue
        
        # Get unique dates from physics data
        dates = df_physics["date"].dt.strftime("%Y-%m-%d").unique().tolist()
        if args.max_dates:
            dates = dates[:args.max_dates]
        _print(f"  Computing RMSE for {len(dates)} dates...")
        
        # Compute daily RMSE
        try:
            df_rmse = compute_daily_rmse_fast(model, dates, args.lead_times, use_cache=not args.no_cache, year=args.year)
            _print(f"  Computed RMSE for {len(df_rmse)} (date, lead_time) pairs")
        except Exception as e:
            _print(f"  ⚠ Error computing RMSE: {e}")
            continue
        
        # Merge physics and RMSE
        df_rmse["date"] = pd.to_datetime(df_rmse["date"])
        df_merged = pd.merge(
            df_physics,
            df_rmse,
            on=["date", "lead_time_hours"],
            how="inner",
        )
        _print(f"  Merged: {len(df_merged)} rows")
        
        if len(df_merged) < 10:
            _print(f"  ⚠ Insufficient merged data for {model}, skipping individual analysis.")
        else:
            # Compute correlations
            avail_metrics = [m for m in PHYSICAL_METRICS if m in df_merged.columns]
            corr_df = compute_correlations(df_merged, avail_metrics)
            corr_df["model"] = model
            all_correlations.append(corr_df)
            
            # Print correlation summary
            _print(f"\n  Correlation Summary (Z500 RMSE):")
            _print(f"  {'Metric':<35} {'12h R²':>8} {'120h R²':>8} {'240h R²':>8}")
            _print(f"  {'-'*60}")
            for metric in avail_metrics:
                row_str = f"  {metric:<35}"
                for lead_h in args.lead_times:
                    if "metric" not in corr_df.columns or corr_df.empty:
                        row_str += f" {'N/A':>8}"
                        continue
                    sub = corr_df[
                        (corr_df["metric"] == metric) &
                        (corr_df["lead_time_hours"] == lead_h) &
                        (corr_df["rmse_type"] == "z500_rmse")
                    ]
                    if len(sub) > 0:
                        r2 = sub["pearson_r2"].values[0]
                        row_str += f" {r2:>8.3f}"
                    else:
                        row_str += f" {'N/A':>8}"
                _print(row_str)
            
            plot_correlation_table(corr_df, model, PLOTS_DIR, args.lead_times)
            
            # Create scatter plots
            fig = plot_scatter_grid(
                df_merged,
                model,
                avail_metrics,
                args.lead_times,
                rmse_col="z500_rmse",
                corr_df=corr_df,
            )
            fig.savefig(PLOTS_DIR / f"scatter_{model}_z500.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            _print(f"\n  Saved: {PLOTS_DIR / f'scatter_{model}_z500.png'}")
            
            # Also for T850
            fig = plot_scatter_grid(
                df_merged,
                model,
                avail_metrics,
                args.lead_times,
                rmse_col="t850_rmse",
                corr_df=corr_df,
            )
            fig.savefig(PLOTS_DIR / f"scatter_{model}_t850.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            _print(f"  Saved: {PLOTS_DIR / f'scatter_{model}_t850.png'}")

        # Add to aggregate list regardless of size (to contribute to total)
        if not df_merged.empty:
            df_merged["model"] = model
            all_merged_dfs.append(df_merged)

    # Save all correlations
    if all_correlations:
        df_all_corr = pd.concat(all_correlations, ignore_index=True)
        corr_csv = PLOTS_DIR / "correlation_summary.csv"
        df_all_corr.to_csv(corr_csv, index=False)
        _print(f"\n✓ Correlation summary saved: {corr_csv}")
    
    _print("\n✓ Done!")


if __name__ == "__main__":
    main()
