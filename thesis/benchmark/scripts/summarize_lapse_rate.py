#!/usr/bin/env python
"""
Summarize and compare Lapse Rate Wasserstein distances between HRES and Pangu.
"""

from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("/home/ekasteleyn/aurora_thesis/thesis/benchmark/results")

def main():
    models = ["hres", "pangu", "fuxi", "graphcast"]
    dfs = []

    for model in models:
        file_path = RESULTS_DIR / f"lapse_rate_w1_{model}_2020.csv"
        if file_path.exists():
            dfs.append(pd.read_csv(file_path))
        else:
            print(f"Warning: Missing {file_path}")

    if not dfs:
        print("Error: No CSV files found.")
        return

    # Combine
    df = pd.concat(dfs, ignore_index=True)

    # Calculate the mean across all dates for each model/region/lead_time
    summary = df.groupby(["metric_name", "lead_hours", "model"])["value"].mean().unstack("model")
    
    # Calculate difference against HRES if HRES is present
    if "hres" in summary.columns:
        for model in models:
            if model != "hres" and model in summary.columns:
                summary[f"Diff (hres - {model})"] = summary["hres"] - summary[model]

    print("=" * 80)
    print(" MEAN LAPSE RATE WASSERSTEIN DISTANCE (W1) ")
    print(" (Lower values are closer to ERA5)")
    print("=" * 80)
    print(summary.to_string(float_format=("%.4f")))
    print("=" * 80)

if __name__ == "__main__":
    main()
