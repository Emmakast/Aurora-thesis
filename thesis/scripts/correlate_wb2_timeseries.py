#!/usr/bin/env python3
"""
Correlation analysis: Physics Time-Series Profiles vs WB2 Lead-Time Profiles.

Uses the full forecast evolution (averaged over the year) to correlate
physics metrics at each lead time against standard error metrics.
This provides a much larger sample size (~40 lead times x 5 models).
"""

import xarray as xr
import pandas as pd
import numpy as np
import gcsfs
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fs = gcsfs.GCSFileSystem(token='anon')

    # ========================================================================
    # 1. Load WB2 Benchmark Profiles (averaged over 2020)
    # ========================================================================
    print("Loading WB2 benchmark profiles...")
    wb2_rows = []
    
    for model, path in WB2_FILES.items():
        try:
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
                tmp = f.name
            fs.get(path, tmp)
            
            # Open without decoding timedeltas initially to inspect
            ds = xr.open_dataset(tmp, decode_timedelta=False)
            
            # Decode lead_time manually if needed or let xarray handle it
            lead_hours = []
            if "lead_time" in ds.coords:
                lead_times = ds.lead_time.values
                # Check attributes for units
                units = ds.lead_time.attrs.get("units", "")
                
                # Heuristic based on magnitude if not timedelta64
                first_val = lead_times[0] if len(lead_times) > 0 else 0
                if isinstance(first_val, np.timedelta64):
                     for val in lead_times:
                        lead_hours.append(float(val / np.timedelta64(1, 'h')))
                else:
                    # If values are small (< 10000), assume hours/days, not nanoseconds
                    # If "hours" in units, just take value.
                    scale_factor = 1.0
                    if np.max(lead_times) > 1e10: # likely nanoseconds
                        scale_factor = 3600000000000.0
                    
                    for val in lead_times:
                        lead_hours.append(float(val) / scale_factor)
            else:
                print(f"  ⚠ {model}: No lead_time coordinate")
                continue

            for i, h in enumerate(lead_hours):
                row = {"model": model, "lead_time_hours": float(h)}
                
                # Z500 RMSE
                try:
                    if 'rmse.geopotential' in ds:
                        val = ds['rmse.geopotential'].isel(lead_time=i).sel(level=500).values
                        row["wb2_z500_rmse"] = float(val) if val.size == 1 else float(val[0])
                    # T850 RMSE
                    if 'rmse.temperature' in ds:
                        val = ds['rmse.temperature'].isel(lead_time=i).sel(level=850).values
                        row["wb2_t850_rmse"] = float(val) if val.size == 1 else float(val[0])
                    # Z500 ACC
                    if 'acc.geopotential' in ds:
                        val = ds['acc.geopotential'].isel(lead_time=i).sel(level=500).values
                        row["wb2_z500_acc"] = float(val) if val.size == 1 else float(val[0])
                except Exception as e:
                    # Skip if specific level missing
                    pass
                
                wb2_rows.append(row)
            
            os.unlink(tmp)
            print(f"  ✓ {model} ({len(lead_hours)} steps)")
            
        except Exception as e:
            print(f"  ⚠ {model}: {e}")

    df_wb2 = pd.DataFrame(wb2_rows)

    # ========================================================================
    # 2. Load Physics Time-Series (averaged over 2020)
    # ========================================================================
    print("\nLoading physics time-series...")
    physics_rows = []
    
    for model in WB2_FILES.keys():
        csv_path = RESULTS_DIR / f"physics_timeseries_{model}_2020.csv"
        if not csv_path.exists():
            print(f"  ⚠ {model}: File not found ({csv_path})")
            continue
            
        try:
            df = pd.read_csv(csv_path)
            
            # Compute Drift Anomaly: abs(Value(t) - Value(t=0))
            # First, find t=min for each date
            df["min_hour"] = df.groupby("date")["forecast_hour"].transform("min")
            
            # Get initial values (at 12h or 0h)
            df_init = df[df["forecast_hour"] == df["min_hour"]].set_index("date")
            
            # Map initial values back to main df
            for col in ["dry_mass_Eg", "water_mass_kg", "total_energy_J"]:
                if col in df.columns:
                    init_map = df_init[col].to_dict()
                    df[f"{col}_init"] = df["date"].map(init_map)
                    df[f"{col}_drift"] = (df[col] - df[f"{col}_init"]).abs() # Absolute drift
                    
                    # Normalize drift by initial value to get %
                    # Avoid div by zero
                    df[f"{col}_drift_pct"] = (df[f"{col}_drift"] / df[f"{col}_init"].replace(0, np.nan)) * 100.0

            # Group by forecast_hour to get yearly profile
            # We average the 'hydrostatic_rmse', 'geostrophic_rmse' directly
            # And we average the 'drift' columns
            
            cols_to_mean = [
                "hydrostatic_rmse", "geostrophic_rmse",
                "dry_mass_Eg_drift_pct", "water_mass_kg_drift_pct", "total_energy_J_drift_pct"
            ]
            cols_present = [c for c in cols_to_mean if c in df.columns]
            
            profile = df.groupby("forecast_hour")[cols_present].mean(numeric_only=True).reset_index()
            
            for _, r in profile.iterrows():
                row = {
                    "model": model, 
                    "lead_time_hours": float(r["forecast_hour"])
                }
                for c in cols_present:
                    row[c] = r[c]
                physics_rows.append(row)
                
            print(f"  ✓ {model} ({len(profile)} steps)")
            
        except Exception as e:
            print(f"  ⚠ {model}: {e}")

    df_physics = pd.DataFrame(physics_rows)
    
    # Rename drift columns for clarity
    df_physics = df_physics.rename(columns={
        "dry_mass_Eg_drift_pct": "dry_mass_drift_cum_pct",
        "water_mass_kg_drift_pct": "water_mass_drift_cum_pct",
        "total_energy_J_drift_pct": "energy_drift_cum_pct"
    })

    # ========================================================================
    # 2.5 Load Spectral Metrics (ke_spectrum)
    # ========================================================================
    print("\nLoading spectral metrics...")
    
    spectral_types = {
        "int": "ke_spectrum_{model}_2020.csv",
        "850": "ke_spectrum_850hpa_{model}_2020.csv"
    }
    
    spectral_rows = []

    # Initialize structure to store per model/lead_time metrics
    # Key: (model, lead_time_hours) -> dict of metrics
    spectral_data = {}

    for s_name, s_pattern in spectral_types.items():
        print(f"  Processing {s_name} spectrum ({s_pattern})...")
        
        for model in WB2_FILES.keys():
            csv_path = RESULTS_DIR / s_pattern.format(model=model)
            if not csv_path.exists():
                # print(f"    ⚠ {model}: File not found")
                continue
                
            try:
                # Read minimal columns
                if s_name == "int":
                     df_spec = pd.read_csv(csv_path, usecols=["lead_hours", "wavenumber", "energy_pred", "energy_era5"])
                     # Group by lead_hours and wavenumber
                     spectrum_mean = df_spec.groupby(["lead_hours", "wavenumber"]).mean().reset_index()
                else:
                     # 850hPa files are in long format: model,lead_hours,wavenumber,energy,source,date
                     df_spec = pd.read_csv(csv_path, usecols=["lead_hours", "wavenumber", "energy", "source"])
                     # Group by lead_hours, wavenumber, source -> mean energy
                     # Unstack source to get columns: energy_pred, energy_era5
                     spectrum_mean = df_spec.groupby(["lead_hours", "wavenumber", "source"])["energy"].mean().unstack(level="source").reset_index()
                     spectrum_mean.columns.name = None
                     # Rename columns if needed (assuming 'pred' and 'era5' are the source values)
                     spectrum_mean = spectrum_mean.rename(columns={"pred": "energy_pred", "era5": "energy_era5"})

                lead_times = spectrum_mean["lead_hours"].unique()
                
                for lt in lead_times:
                    grp = spectrum_mean[spectrum_mean["lead_hours"] == lt]
                    
                    # Metrics
                    l1 = (grp["energy_pred"] - grp["energy_era5"]).abs().sum()
                    total_pred = grp["energy_pred"].sum()
                    total_era5 = grp["energy_era5"].sum()
                    ratio = total_pred / total_era5 if total_era5 > 0 else 1.0
                    
                    # High-k (k > 20)
                    grp_high = grp[grp["wavenumber"] > 20]
                    high_pred = grp_high["energy_pred"].sum()
                    high_era5 = grp_high["energy_era5"].sum()
                    high_ratio = high_pred / high_era5 if high_era5 > 0 else 1.0
                    
                    key = (model, float(lt))
                    if key not in spectral_data:
                        spectral_data[key] = {"model": model, "lead_time_hours": float(lt)}
                    
                    spectral_data[key][f"spectral_{s_name}_l1"] = l1
                    spectral_data[key][f"spectral_{s_name}_ratio"] = ratio
                    spectral_data[key][f"spectral_{s_name}_high_k"] = high_ratio
                    
            except Exception as e:
                print(f"    ⚠ {model}: {e}")

    # Convert dictionary to list
    spectral_rows = list(spectral_data.values())
            
    if spectral_rows:
        df_spectral = pd.DataFrame(spectral_rows)
        # Merge into df_physics
        df_physics = pd.merge(df_physics, df_spectral, on=["model", "lead_time_hours"], how="outer")
        
    # ========================================================================
    # 3. Merge & Correlate
    # ========================================================================
    # Round lead times to nearest int to ensure matching (avoid float precision issues)
    df_wb2["lead_time_hours"] = df_wb2["lead_time_hours"].astype(float).round().astype(int)
    if not df_physics.empty:
        df_physics["lead_time_hours"] = df_physics["lead_time_hours"].astype(float).round().astype(int)
    
    df_merged = pd.merge(df_wb2, df_physics, on=["model", "lead_time_hours"], how="inner")
    
    print(f"\nMerged Time-Series Profile: {len(df_merged)} points")
    
    metrics = [
        "geostrophic_rmse", "hydrostatic_rmse",
        "dry_mass_drift_cum_pct", "water_mass_drift_cum_pct", "energy_drift_cum_pct",
        "spectral_int_l1", "spectral_int_ratio", "spectral_int_high_k",
        "spectral_850_l1", "spectral_850_ratio", "spectral_850_high_k"
    ]
    
    targets = ["wb2_z500_rmse", "wb2_t850_rmse", "wb2_z500_acc"]
    
    print("\n" + "="*80)
    print("CORRELATION: Physics Profile vs WB2 Error Profile")
    print("(Spearman ρ using full forecast evolution across all models)")
    print("="*80)
    print(f"{'Metric':<30} {'vs Z500 RMSE':>12} {'vs T850 RMSE':>12} {'vs Z500 ACC':>12}")
    print("-" * 75)
    
    results = []
    
    for m in metrics:
        if m not in df_merged.columns:
            continue
            
        valid = df_merged.dropna(subset=[m] + targets)
        if len(valid) < 5:
            continue
            
        row_str = f"{m:<30}"
        res_row = {"metric": m, "n": len(valid)}
        
        for t in targets:
            r, p = stats.spearmanr(valid[m], valid[t])
            sig = "*" if p < 0.05 else " "
            row_str += f" {r:>+6.3f}{sig}    "
            res_row[f"r_{t}"] = r
            res_row[f"p_{t}"] = p
            
        print(row_str)
        results.append(res_row)
        
    print("\n* = p < 0.05")
    
    # Save
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "wb2_timeseries_correlation.csv", index=False)
    print(f"\nSaved correlation results to {OUTPUT_DIR / 'wb2_timeseries_correlation.csv'}")
    
    # Plot
    for m in metrics:
        if m not in df_merged.columns: continue
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_merged, x="wb2_z500_rmse", y=m, hue="model", style="model", s=60, alpha=0.7)
        plt.title(f"{m} vs Z500 RMSE (Profile Correlation)")
        plt.xlabel("Z500 RMSE (m^2/s^2)")
        plt.ylabel(m)
        plt.savefig(OUTPUT_DIR / f"scatter_timeseries_{m}.png", bbox_inches="tight")
        plt.close()
        
    print("Saved scatter plots.")

if __name__ == "__main__":
    main()
