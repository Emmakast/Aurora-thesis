"""
3_evaluate_patching.py

Goal: Automate the calculation of the Arctic Oscillation (AO) index on the generated NetCDF files 
to measure the Causal Effect. Wraps the evaluate_ao.py script to extract the AO index 
for each Neutral date and computes the Average Treatment Effect (ATE).
"""

import subprocess
import os
import sys
import pandas as pd
from pathlib import Path
import re

def load_dates(csv_path, phase):
    """Load target dates for Active or Neutral phases."""
    df = pd.read_csv(csv_path)
    df_phase = df[df['Type'] == phase]
    dates = []
    for _, row in df_phase.iterrows():
        dates.append(f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Day']):02d}")
    return dates

def main():
    dates_csv = "/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_ao_81.csv"
    output_dir = Path("/scratch-shared/ekasteleyn/aurora_thesis_output/patched_rollouts")
    
    # Paths for evaluation
    evaluator_script = "/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/evaluate_ao.py"
    eof_path = "/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/ao_loading_pattern.nc"
    climatology_path = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr"
    
    neutral_dates = load_dates(dates_csv, phase="Neutral")
    print(f"Evaluating {len(neutral_dates)} Neutral dates...")
    
    results = []
    
    for i, day in enumerate(neutral_dates):
        print(f"\n[{i+1}/{len(neutral_dates)}] Evaluating date: {day}")
        
        for step in [4, 8, 12]:
            base_nc = output_dir / f"base_{day}_step{step:02d}.nc"
            patched_nc = output_dir / f"patched_{day}_step{step:02d}.nc"
            
            if not base_nc.exists() or not patched_nc.exists():
                print(f"  Missing base or patched NetCDF for {day} step {step}. Skipping.")
                continue
                
            cmd = [
                sys.executable, evaluator_script,
                "--base", str(base_nc),
                "--steered", str(patched_nc),
                "--eof", eof_path,
                "--climatology", climatology_path
            ]
            
            # print(f"  Running subprocess: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = result.stdout
                
                base_ao_match = re.search(r"Base AO Index:\s+([-\d\.]+)", output)
                steered_ao_match = re.search(r"Steered AO Index:\s+([-\d\.]+)", output)
                
                if base_ao_match and steered_ao_match:
                    base_ao = float(base_ao_match.group(1))
                    steered_ao = float(steered_ao_match.group(1))
                    delta = steered_ao - base_ao
                    
                    print(f"  ✓ Step {step:02d} (Day {step//4}): Base AO: {base_ao:.4f}, Patched AO: {steered_ao:.4f}, Causal Effect: {delta:.4f}")
                    
                    results.append({
                        "Date": day,
                        "Step": step,
                        "Day_Lead": step // 4,
                        "Base_AO": base_ao,
                        "Patched_AO": steered_ao,
                        "Causal_Effect": delta
                    })
                else:
                    print(f"  Failed to parse AO indices from output for step {step}.")
                    
            except subprocess.CalledProcessError as e:
                print(f"  Error running evaluator for {day} step {step}: {e}")
                continue

    if len(results) == 0:
        print("\nNo results were generated. Check the error logs above.")
        return
        
    results_df = pd.DataFrame(results)
    
    print("\n=============================================")
    print("      SPATIO-TEMPORAL CAUSAL TRACING RESULTS  ")
    print("=============================================")
    print(f"Evaluated Dates: {len(results_df['Date'].unique())}")
    for lead_day in [1, 2, 3]:
        df_lead = results_df[results_df["Day_Lead"] == lead_day]
        if not df_lead.empty:
            ate = df_lead["Causal_Effect"].mean()
            print(f"Day {lead_day} Average Treatment Effect (ATE): {ate:.4f} (n={len(df_lead)})")
    
    summary_csv = output_dir / "causal_tracing_summary.csv"
    results_df.to_csv(summary_csv, index=False)
    print(f"\nDetailed results saved to {summary_csv}")

if __name__ == "__main__":
    main()
