#!/usr/bin/env python3
"""
Adds drift calculation to Aurora S3 results and generates a summary table 
comparable to other models. 

This post-processing script:
1. Reads the physics evaluation output for S3.
2. Constructs the drift trajectory using available lead times (12h, 24h, 120h, 240h).
3. Computes the percentage drift per day metrics.
4. Uses `summarize_physics_metrics.py` to produce standard summaries.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Imort existing metrics & summarize script
sys.path.insert(0, str(Path(__file__).parent))
from physics_metrics import compute_drift_percentages
from summarize_physics_metrics import summarize

def compute_s3_drift_for_date(date_df: pd.DataFrame) -> list[dict]:
    """
    Given all metrics for a single start date across available lead times,
    compute the drift metrics and return them as new rows.
    """
    rows = []
    
    # We need to construct trajectories for dry_mass_Eg, water_mass_kg, total_energy_J
    metrics_needed = ["dry_mass_Eg", "water_mass_kg", "total_energy_J"]
    
    # Target windows analogous to run_physics_evaluation.py
    # target 12h -> window 12h-24h
    # target 24h -> window 12h-24h 
    # target 120h -> window 12h-120h
    # target 240h -> window 12h-240h
    
    windows = {
        12: [12, 24],
        24: [12, 24],
        120: [12, 24, 120],
        240: [12, 24, 120, 240]
    }
    
    init_hour = date_df["init_hour"].iloc[0]
    date_str = date_df["date"].iloc[0]
    n_levels = date_df["n_levels"].iloc[0] if "n_levels" in date_df.columns else None

    for target_lead, window_leads in windows.items():
        # filter dataframe for window subset
        sub_df = date_df[date_df["lead_time_hours"].isin(window_leads)]
        
        hours_aurora = []
        dry_vals, water_vals, energy_vals = [], [], []
        hours_era5 = []
        water_vals_e, energy_vals_e = [], []
        
        for lh in window_leads:
            lh_df = sub_df[sub_df["lead_time_hours"] == lh]
            if lh_df.empty:
                continue
                
            hours_aurora.append(lh)
            hours_era5.append(lh)
            
            dry_vals.append(lh_df[lh_df["metric_name"] == "dry_mass_Eg"]["model_value"].values[0])
            w_pred = lh_df[lh_df["metric_name"] == "water_mass_kg"]["model_value"].values[0]
            e_pred = lh_df[lh_df["metric_name"] == "total_energy_J"]["model_value"].values[0]
            water_vals.append(w_pred)
            energy_vals.append(e_pred)
            
            w_era5 = lh_df[lh_df["metric_name"] == "water_mass_kg"]["era5_value"].values[0]
            e_era5 = lh_df[lh_df["metric_name"] == "total_energy_J"]["era5_value"].values[0]
            water_vals_e.append(w_era5)
            energy_vals_e.append(e_era5)
            
        if len(hours_aurora) < 2:
            continue
            
        drift = compute_drift_percentages(
            np.array(hours_aurora), np.array(dry_vals), np.array(water_vals), np.array(energy_vals),
            np.array(hours_era5), np.array(water_vals_e), np.array(energy_vals_e)
        )
        
        for metric_name, val in drift.items():
            rows.append({
                "date": date_str,
                "init_hour": init_hour,
                "lead_time_hours": target_lead,
                "metric_name": metric_name,
                "model_value": val,
                "era5_value": np.nan,
                "n_levels": n_levels,
            })
            
    return rows

def process_and_summarize(input_csv: Path, output_summary_csv: Path):
    print(f"Reading {input_csv} ...")
    df = pd.read_csv(input_csv)
    
    # Calculate drift metrics for all dates
    drift_rows = []
    for date, date_df in df.groupby("date"):
        drift_rows.extend(compute_s3_drift_for_date(date_df))
        
    drift_df = pd.DataFrame(drift_rows)
    df_combined = pd.concat([df, drift_df], ignore_index=True)
    
    combined_csv = input_csv.parent / input_csv.name.replace(".csv", "_with_drift.csv")
    df_combined.to_csv(combined_csv, index=False)
    print(f"Saved drift-enriched evaluation file to {combined_csv}")
    
    # Produce the official summary table matching graphcast
    print("Generating comprehensive summary csv...")
    summarize(df_combined, output_summary_csv, year="2022", model="aurora_s3")
    print(f"Summary correctly written to {output_summary_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/ekasteleyn/aurora_thesis/thesis/results/physics_aurora_s3_2022_init0.csv")
    parser.add_argument("--output", default="/home/ekasteleyn/aurora_thesis/thesis/results/physics_summary_aurora_s3_2022.csv")
    args = parser.parse_args()
    
    process_and_summarize(Path(args.input), Path(args.output))
