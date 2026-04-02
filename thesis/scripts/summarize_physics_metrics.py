#!/usr/bin/env python3
"""
Summarize physics evaluation metrics from a full-year CSV.

Produces a table with one row per (lead_time, metric) showing the
yearly average for each metric, computed as follows:

  - geostrophic_rmse, hydrostatic_rmse:
        mean( model_value )                     per lead time 
  - dry_mass_drift_pct_per_day, water_mass_drift_pct_per_day,
    total_energy_drift_pct_per_day:
        mean( model_value )                     per lead time
  - effective_resolution:
        mean( model_value )                     per lead time
  - spectral_divergence, spectral_residual, small_scale_ratio:
        mean( model_value )  per lead time
"""

import re

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def _infer_year(path_str: str) -> str:
    """Extract a 4-digit year from the filename, default '????'."""
    m = re.search(r'(\d{4})', Path(path_str).stem)
    return m.group(1) if m else "????"


def _infer_model(path_str: str) -> str:
    """Extract model name from filename like physics_evaluation_pangu_2020.csv."""
    stem = Path(path_str).stem
    # Remove known prefixes and year suffix
    stem = re.sub(r'^physics_(evaluation|summary)_', '', stem)
    stem = re.sub(r'_?\d{4}$', '', stem)
    return stem if stem else "unknown"

def summarize(df: pd.DataFrame, output_path: Path, year: str = "????", model: str = "unknown"):
    # ---- Quick data-quality check ----
    print("=== Data Quality Check ===")
    # Balance metrics have both model and era5 values
    for m in ["geostrophic_rmse", "hydrostatic_rmse"]:
        sub = df[df["metric_name"] == m]
        if "era5_value" in sub.columns:
            n_unique = sub["era5_value"].nunique()
            status = "✓ varies" if n_unique > 1 else "⚠ CONSTANT (bug?)"
            print(f"  {m:<25s}  {n_unique:>4d} unique ERA5 values  {status}")
        else:
             print(f"  {m:<25s}  ERA5 value column missing")
    # Drift metrics are model-only
    for m in ["dry_mass_drift_pct_per_day", "water_mass_drift_pct_per_day",
              "total_energy_drift_pct_per_day"]:
        sub = df[df["metric_name"] == m]
        if "model_value" in sub.columns:
            n_finite = sub["model_value"].dropna().count()
            print(f"  {m:<40s}  {n_finite:>4d} finite values")
        else:
            print(f"  {m:<40s}  model_value column missing")
    print()

    # ---------- metrics where we report model RMSE averaged per lead time ----------
    diff_metrics = [
        "geostrophic_rmse",
        "hydrostatic_rmse",
    ]

    mask_diff = df["metric_name"].isin(diff_metrics)
    df_diff = df.loc[mask_diff].copy()

    agg_dict_diff = dict(
        mean_model=("model_value", "mean") if "model_value" in df_diff.columns else ("lead_time_hours", "count"),
        std_model=("model_value", "std") if "model_value" in df_diff.columns else ("lead_time_hours", "count"),
        mean_era5=("era5_value", "mean") if "era5_value" in df_diff.columns else ("lead_time_hours", "count"),
        n=("model_value", "count") if "model_value" in df_diff.columns else ("lead_time_hours", "count"),
    )
    if "n_levels" in df_diff.columns:
        agg_dict_diff["n_levels"] = ("n_levels", "first")
    if "sp_method" in df_diff.columns:
        agg_dict_diff["sp_method"] = ("sp_method", "first")

    # Compute per-row difference (RMSE_pred - RMSE_era5) before aggregation
    if "model_value" in df_diff.columns and "era5_value" in df_diff.columns:
        df_diff["diff_value"] = df_diff["model_value"] - df_diff["era5_value"]
        agg_dict_diff["mean_diff"] = ("diff_value", "mean")
        agg_dict_diff["std_diff"] = ("diff_value", "std")

    summary_diff = (
        df_diff.groupby(["lead_time_hours", "metric_name"])
        .agg(**agg_dict_diff)
        .reset_index()
    )

    # ---------- effective resolution: average model_value per lead time ----------
    mask_eff = df["metric_name"] == "effective_resolution_km"
    df_eff = df.loc[mask_eff].copy()
    
    if "model_value" in df_eff.columns:
        # Replace any remaining inf values with the grid-scale resolution
        # (2πR / l_max).  We infer l_max from the finite minimum value;
        # if none exist we leave as-is.
        inf_mask = np.isinf(df_eff["model_value"])
        if inf_mask.any():
            finite_vals = df_eff.loc[~inf_mask, "model_value"]
            if len(finite_vals) > 0:
                grid_res = finite_vals.min()
            else:
                # All inf → fall back to 0.25° grid assumption: lmax=359
                grid_res = (2.0 * np.pi * 6.371e6 / 359.0) / 1000.0
            n_replaced = inf_mask.sum()
            print(f"  effective_resolution_km: replacing {n_replaced} inf → {grid_res:.1f} km (grid scale)")
            df_eff.loc[inf_mask, "model_value"] = grid_res

        agg_dict_eff = dict(
            mean_value=("model_value", "mean"),
            std_value=("model_value", "std"),
            n=("model_value", "count"),
        )
        if "n_levels" in df_eff.columns:
            agg_dict_eff["n_levels"] = ("n_levels", "first")
        if "sp_method" in df_eff.columns:
            agg_dict_eff["sp_method"] = ("sp_method", "first")

        summary_eff = (
            df_eff.groupby(["lead_time_hours", "metric_name"])
            .agg(**agg_dict_eff)
            .reset_index()
        )
    else:
        summary_eff = pd.DataFrame(columns=["lead_time_hours", "metric_name", "mean_value", "std_value", "n"])

    # ---------- spectral / small-scale: overall average per lead time ----------
    overall_metrics = [
        "spectral_divergence",
        "spectral_residual",
        "small_scale_ratio",
        "dry_mass_drift_pct_per_day",
        "water_mass_drift_pct_per_day",
        "total_energy_drift_pct_per_day",
        "mean_q_drift_pct_per_day",
    ]

    mask_overall = df["metric_name"].isin(overall_metrics)
    df_overall = df.loc[mask_overall].copy()

    if "model_value" in df_overall.columns:
        agg_dict_overall = dict(
            mean_value=("model_value", "mean"),
            std_value=("model_value", "std"),
            n=("model_value", "count"),
        )
        if "n_levels" in df_overall.columns:
            agg_dict_overall["n_levels"] = ("n_levels", "first")
        if "sp_method" in df_overall.columns:
            agg_dict_overall["sp_method"] = ("sp_method", "first")

        summary_overall = (
            df_overall.groupby(["lead_time_hours", "metric_name"])
            .agg(**agg_dict_overall)
            .reset_index()
        )
    else:
        summary_overall = pd.DataFrame(columns=["lead_time_hours", "metric_name", "mean_value", "std_value", "n"])

    # ---------- Pretty-print ----------
    print("=" * 80)
    print(f"Physics Evaluation Summary – {model} – Year {year}")
    print("=" * 80)

    for lt in sorted(df["lead_time_hours"].unique()):
        print(f"\n--- Lead time: {int(lt)} h ---")

        # Balance metrics (show RMSE_pred - RMSE_era5 as primary, raw values as context)
        sub_diff = summary_diff[summary_diff["lead_time_hours"] == lt].sort_values("metric_name")
        for _, row in sub_diff.iterrows():
            out_str = f"  {row['metric_name']:<30s}  "

            # Primary: difference (RMSE_pred - RMSE_era5)
            if "mean_diff" in row.index and not np.isnan(row["mean_diff"]):
                out_str += f"Δ(pred−ERA5)={row['mean_diff']:.6g} (±{row['std_diff']:.6g})"
            elif "model_value" in df_diff.columns and not np.isnan(row["mean_model"]):
                out_str += f"Model={row['mean_model']:.6g} (±{row['std_model']:.6g})"
            else:
                out_str += "(n=0)"

            # Context: raw RMSE values
            if "model_value" in df_diff.columns and not np.isnan(row["mean_model"]):
                out_str += f"  [pred={row['mean_model']:.4g}"
                if "era5_value" in df_diff.columns and not np.isnan(row["mean_era5"]):
                    out_str += f", ERA5={row['mean_era5']:.4g}"
                out_str += f", n={int(row['n'])}]"

            print(out_str)

        # Effective resolution
        sub_eff = summary_eff[summary_eff["lead_time_hours"] == lt]
        for _, row in sub_eff.iterrows():
            print(
                f"  {row['metric_name']:<30s}  "
                f"mean = {row['mean_value']:.1f}  "
                f"(±{row['std_value']:.1f}, n={int(row['n'])})"
            )

        # Spectral / ratio
        sub_ov = summary_overall[summary_overall["lead_time_hours"] == lt].sort_values("metric_name")
        for _, row in sub_ov.iterrows():
            print(
                f"  {row['metric_name']:<30s}  "
                f"mean = {row['mean_value']:.6f}  "
                f"(±{row['std_value']:.4f}, n={int(row['n'])})"
            )

    # ---------- Save combined CSV ----------
    # Build a unified long-format output
    rows_out = []
    for _, row in summary_diff.iterrows():
        entry = {
            "lead_time_hours": row["lead_time_hours"],
            "metric_name": row["metric_name"],
            "aggregation": "mean(pred−era5)",
            "mean_value": row.get("mean_diff", np.nan),
            "std": row.get("std_diff", np.nan),
            "mean_model": row["mean_model"],
            "std_model": row["std_model"],
            "mean_era5": row["mean_era5"],
            "n": int(row["n"]),
        }
        if "n_levels" in row.index:
            entry["n_levels"] = row["n_levels"]
        if "sp_method" in row.index:
            entry["sp_method"] = row["sp_method"]
        rows_out.append(entry)
    for _, row in summary_eff.iterrows():
        entry = {
            "lead_time_hours": row["lead_time_hours"],
            "metric_name": row["metric_name"],
            "aggregation": "mean(model)",
            "mean_value": row["mean_value"],
            "std": row["std_value"],
            "n": int(row["n"]),
        }
        if "n_levels" in row.index:
            entry["n_levels"] = row["n_levels"]
        if "sp_method" in row.index:
            entry["sp_method"] = row["sp_method"]
        rows_out.append(entry)
    for _, row in summary_overall.iterrows():
        entry = {
            "lead_time_hours": row["lead_time_hours"],
            "metric_name": row["metric_name"],
            "aggregation": "mean(model)",
            "mean_value": row["mean_value"],
            "std": row["std_value"],
            "n": int(row["n"]),
        }
        if "n_levels" in row.index:
            entry["n_levels"] = row["n_levels"]
        if "sp_method" in row.index:
            entry["sp_method"] = row["sp_method"]
        rows_out.append(entry)

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Summarize physics evaluation results (CSV)"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to the main evaluation CSV (joint or model).",
    )
    parser.add_argument(
        "--era5", type=str, default=None,
        help="Path to separate ERA5 evaluation CSV (optional, for split mode).",
    )
    parser.add_argument(
        "--aurora", type=str, default=None,
        help="Path to separate model evaluation CSV (optional, same as --input).",
    )
    args = parser.parse_args()

    # Handle various input combinations
    input_path = args.input or args.aurora
    
    if input_path is None:
        # Default fallback — try to find any physics_evaluation CSV
        input_path = "thesis/results/physics_evaluation_2022.csv"
        
    if not Path(input_path).exists() and args.era5 and Path(args.era5).exists():
        # Only ERA5 provided?
         print(f"Loading only ERA5 metrics from {args.era5}")
         df = pd.read_csv(args.era5)
         df_aurora = pd.DataFrame() # Empty
         df_era5 = df
         year = _infer_year(args.era5)
         model = "era5"
         in_name = Path(args.era5).name
         if "evaluation" in in_name:
             out_name = in_name.replace("evaluation", "summary")
         else:
             out_name = in_name.replace("physics_", "physics_summary_")
             if out_name == in_name:
                 out_name = "summary_" + in_name
         output_path = Path(args.era5).with_name(out_name)

    elif Path(input_path).exists():
        print(f"Loading metrics from {input_path}")
        df = pd.read_csv(input_path)
        
        # If separate ERA5 file provided, merge it
        if args.era5:
             print(f"Merging with ERA5 metrics from {args.era5}")
             df_era5_in = pd.read_csv(args.era5)
             
             # Joint Logic:
             # Model file has (date, lead, metric, model_value)
             # ERA5 file has (date, lead, metric, era5_value)
             # We want to merge them on keys.
             
             # Best strategy: use combine_first or merge.
             # Let's try to act like a database join on (date, lead_time_hours, metric_name)
             keys = ["date", "lead_time_hours", "metric_name"]
             
             # Drop empty columns from each side to avoid conflicts
             df = df.dropna(axis=1, how='all') 
             df_era5_in = df_era5_in.dropna(axis=1, how='all')

             df = pd.merge(df, df_era5_in, on=keys, how="outer", suffixes=('', '_era5'))
             
             # Combine columns if needed (e.g. if era5_value exists in both but one is NaN)
             if "era5_value_era5" in df.columns:
                 if "era5_value" in df.columns:
                    df["era5_value"] = df["era5_value"].fillna(df["era5_value_era5"])
                 else:
                    df["era5_value"] = df["era5_value_era5"]
                 df = df.drop(columns=["era5_value_era5"])
             if "model_value_era5" in df.columns:
                 # Should not happen typically, but for symmetry
                 if "model_value" in df.columns:
                    df["model_value"] = df["model_value"].fillna(df["model_value_era5"])
                 else:
                    df["model_value"] = df["model_value_era5"]
                 df = df.drop(columns=["model_value_era5"])
             # Also handle legacy aurora_value columns from old CSVs
             if "aurora_value" in df.columns and "model_value" not in df.columns:
                 df = df.rename(columns={"aurora_value": "model_value"})
             elif "aurora_value_era5" in df.columns:
                 df = df.drop(columns=["aurora_value_era5"], errors="ignore")
             # Combine n_levels / sp_method if duplicated by merge
             for meta_col in ("n_levels", "sp_method"):
                 if f"{meta_col}_era5" in df.columns:
                     if meta_col in df.columns:
                         df[meta_col] = df[meta_col].fillna(df[f"{meta_col}_era5"])
                     else:
                         df[meta_col] = df[f"{meta_col}_era5"]
                     df = df.drop(columns=[f"{meta_col}_era5"])

        in_name = Path(input_path).name
        if "evaluation" in in_name:
            out_name = in_name.replace("evaluation", "summary")
        else:
            out_name = in_name.replace("physics_", "physics_summary_")
            if out_name == in_name:
                out_name = "summary_" + in_name
        output_path = Path(input_path).with_name(out_name)
        year = _infer_year(input_path)
        model = _infer_model(input_path)
    else:
        print(f"Error: Input file needed. tried {input_path}")
        return

    summarize(df, output_path, year=year, model=model)

if __name__ == "__main__":
    main()
