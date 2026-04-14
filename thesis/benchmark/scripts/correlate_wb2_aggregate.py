#!/usr/bin/env python3
"""
Quick correlation analysis: Physics Metrics vs WB2 Benchmark Results.

Compares yearly-averaged physics metrics against WB2 pre-computed RMSE/ACC
across 5 models (pangu, fuxi, graphcast, hres, neuralgcm) × 3 lead times.

This uses the aggregated WB2 benchmark .nc files (not daily data).

Usage:
    python correlate_wb2_aggregate.py
"""

import xarray as xr
import pandas as pd
import numpy as np
import gcsfs
import tempfile
import os
from pathlib import Path
from scipy import stats

# WB2 benchmark files for each model
WB2_FILES = {
    "pangu": "weatherbench2/benchmark_results/pangu_hres_init_vs_era5_1440x721_2020.nc",
    "fuxi": "weatherbench2/benchmark_results/fuxi_vs_era5_1440x721_2020.nc",
    "graphcast": "weatherbench2/benchmark_results/graphcast_hres_init_vs_era5_1440x721_2020.nc",
    "hres": "weatherbench2/benchmark_results/hres_vs_era5_1440x721_2020.nc",
    "neuralgcm": "weatherbench2/benchmark_results/neuralgcm_ens_mean_vs_era5_240x121_2020.nc",
}

RESULTS_DIR = Path.home() / "aurora_thesis" / "thesis" / "results"
OUTPUT_DIR = RESULTS_DIR / "plots_correlation"
TARGET_LEADS = [12, 120, 240]

# Physics metrics to correlate
PHYSICS_METRICS = [
    "geostrophic_rmse", "hydrostatic_rmse", "effective_resolution_km",
    "spectral_divergence", "spectral_residual", "small_scale_ratio",
    "dry_mass_drift_pct_per_day", "water_mass_drift_pct_per_day", 
    "total_energy_drift_pct_per_day"
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fs = gcsfs.GCSFileSystem(token='anon')

    # Load WB2 metrics
    print("Loading WB2 benchmark results...")
    wb2_rows = []
    for model, path in WB2_FILES.items():
        try:
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
                tmp = f.name
            fs.get(path, tmp)
            ds = xr.open_dataset(tmp)
            
            for lead_h in TARGET_LEADS:
                lead_td = np.timedelta64(lead_h, 'h')
                ds_lead = ds.sel(lead_time=lead_td, region='global')
                
                row = {"model": model, "lead_time_hours": lead_h}
                
                # Z500 RMSE
                if 'rmse.geopotential' in ds:
                    row["wb2_z500_rmse"] = float(ds_lead['rmse.geopotential'].sel(level=500).values)
                
                # T850 RMSE
                if 'rmse.temperature' in ds:
                    row["wb2_t850_rmse"] = float(ds_lead['rmse.temperature'].sel(level=850).values)
                
                # Z500 ACC
                if 'acc.geopotential' in ds:
                    row["wb2_z500_acc"] = float(ds_lead['acc.geopotential'].sel(level=500).values)

                # T850 ACC
                if 'acc.temperature' in ds:
                    row["wb2_t850_acc"] = float(ds_lead['acc.temperature'].sel(level=850).values)

                # MSLP RMSE
                if 'rmse.mean_sea_level_pressure' in ds:
                    row["wb2_mslp_rmse"] = float(ds_lead['rmse.mean_sea_level_pressure'].values)

                wb2_rows.append(row)
            
            os.unlink(tmp)
            print(f"  ✓ {model}")
        except Exception as e:
            print(f"  ⚠ {model}: {e}")

    df_wb2 = pd.DataFrame(wb2_rows)

    # Load physics summary metrics (compute from raw daily evaluation CSVs)
    print("\nLoading physics metrics from raw evaluation files...")
    physics_rows = []
    
    # Define metric columns explicitly to avoid confusion
    metric_cols = [
        "geostrophic_rmse", "hydrostatic_rmse", "effective_resolution_km", 
        "spectral_divergence", "spectral_residual", "small_scale_ratio",
        "dry_mass_drift_pct_per_day", "water_mass_drift_pct_per_day",
        "total_energy_drift_pct_per_day"
    ]
    
    for model in WB2_FILES.keys():
        csv_path = RESULTS_DIR / f"physics_evaluation_{model}_2020.csv"
        try:
            if not csv_path.exists():
                print(f"  ⚠ {model}: File not found ({csv_path})")
                continue
                
            df = pd.read_csv(csv_path)
            
            # Pivot table to wide format: (date, lead_time) -> metrics
            df_wide = df.pivot_table(
                index=["date", "lead_time_hours"],
                columns="metric_name",
                values="model_value",
                aggfunc="first"
            ).reset_index()
            
            # Group by lead_time and compute mean
            summary = df_wide.groupby("lead_time_hours").mean(numeric_only=True).reset_index()
            
            for _, r in summary.iterrows():
                lead_h = r["lead_time_hours"]
                if lead_h not in TARGET_LEADS:
                    continue
                
                row = {"model": model, "lead_time_hours": lead_h}
                for metric in metric_cols:
                    if metric in r and pd.notna(r[metric]):
                        row[metric] = r[metric]
                physics_rows.append(row)
                
            print(f"  ✓ {model} (computed summary from {len(df)} raw rows)")
        except Exception as e:
            print(f"  ⚠ {model}: {e}")

    df_physics = pd.DataFrame(physics_rows)

    # Merge
    df = pd.merge(df_wb2, df_physics, on=["model", "lead_time_hours"])
    print(f"\nMerged data: {len(df)} rows (5 models × 3 lead times)")

    # Compute correlations
    print("\n" + "="*80)
    print("CORRELATION: Physics Metrics vs WB2 Standard Metrics")
    print("(Spearman ρ across 5 models × 3 lead times = 15 data points)")
    print("="*80)
    print(f"\n{'Physics Metric':<35} {'vs Z500 RMSE':>14} {'vs T850 RMSE':>14} {'vs Z500 ACC':>14} {'vs T850 ACC':>14} {'vs MSLP RMSE':>14}")
    print("-"*110)

    corr_results = []
    for metric in PHYSICS_METRICS:
        if metric not in df.columns:
            print(f"{metric:<35} {'missing':>14} {'':>14} {'':>14} {'':>14} {'':>14}")
            continue
        
        valid = df[[metric, "wb2_z500_rmse", "wb2_t850_rmse", "wb2_z500_acc", "wb2_t850_acc", "wb2_mslp_rmse"]].dropna()
        if len(valid) < 4:
            print(f"{metric:<35} {'<4 pts':>14} {'':>14} {'':>14} {'':>14} {'':>14}")
            continue
        
        x = valid[metric].values
        
        r_z500, p_z500 = stats.spearmanr(x, valid["wb2_z500_rmse"].values)
        r_t850, p_t850 = stats.spearmanr(x, valid["wb2_t850_rmse"].values)
        r_zacc, p_zacc = stats.spearmanr(x, valid["wb2_z500_acc"].values)
        r_tacc, p_tacc = stats.spearmanr(x, valid["wb2_t850_acc"].values)
        r_mslp, p_mslp = stats.spearmanr(x, valid["wb2_mslp_rmse"].values)
        
        corr_results.append({
            "metric": metric,
            "spearman_z500_rmse": r_z500,
            "p_z500_rmse": p_z500,
            "spearman_t850_rmse": r_t850,
            "p_t850_rmse": p_t850,
            "spearman_z500_acc": r_zacc,
            "p_z500_acc": p_zacc,
            "spearman_t850_acc": r_tacc,
            "p_t850_acc": p_tacc,
            "spearman_mslp_rmse": r_mslp,
            "p_mslp_rmse": p_mslp,
            "n_samples": len(valid),
        })
        
        sig_z = "*" if p_z500 < 0.05 else " "
        sig_t = "*" if p_t850 < 0.05 else " "
        sig_za = "*" if p_zacc < 0.05 else " "
        sig_ta = "*" if p_tacc < 0.05 else " "
        sig_m = "*" if p_mslp < 0.05 else " "
        
        print(f"{metric:<35} {r_z500:>+.3f}{sig_z}(n={len(valid)}) {r_t850:>+.3f}{sig_t}      {r_zacc:>+.3f}{sig_za}      {r_tacc:>+.3f}{sig_ta}      {r_mslp:>+.3f}{sig_m}")

    print("\n* = p < 0.05 (statistically significant)")

    # Save results
    df_corr = pd.DataFrame(corr_results)
    corr_csv = OUTPUT_DIR / "wb2_aggregate_correlation.csv"
    df_corr.to_csv(corr_csv, index=False)
    print(f"\n✓ Correlation results saved: {corr_csv}")

    # Save merged data
    data_csv = OUTPUT_DIR / "wb2_aggregate_merged_data.csv"
    df.to_csv(data_csv, index=False)
    print(f"✓ Merged data saved: {data_csv}")

    # Show data table
    print("\n" + "="*80)
    print("DATA: Model averages at each lead time")
    print("="*80)
    print(df[["model", "lead_time_hours", "wb2_z500_rmse", "effective_resolution_km", 
             "spectral_divergence", "geostrophic_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
