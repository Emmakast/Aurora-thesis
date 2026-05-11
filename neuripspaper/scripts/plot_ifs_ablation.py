#!/usr/bin/env python3
"""
Ablation script to compare IFS HRES against a modified version
using the hypsometric equation and reference surface pressure.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Config ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

# The exact base names used in the CSV files
MODELS = ["hres", "ifs_ablation_hypso", "ifs_ablation_refsp"]

# Map each model to the directory it is stored in
MODEL_DIRS = {
    "hres": BASE_DIR / "results_ifs",
    "ifs_ablation_hypso": BASE_DIR / "results_ifs_hypso",
    "ifs_ablation_refsp": BASE_DIR / "results_ifs_refsp",
}

NICE = {
    "hres": "HRES (Direct)",
    "ifs_ablation_hypso": "HRES (US Standard)",
    "ifs_ablation_refsp": "HRES (Ref SP)",
}
MODEL_STYLES = {
    "hres": {"color": "#56B4E9", "marker": "P"},
    "ifs_ablation_hypso": {"color": "#D55E00", "marker": "s"},
    "ifs_ablation_refsp": {"color": "#009E73", "marker": "^"}
}

def load_summaries() -> dict[str, pd.DataFrame]:
    out = {}
    for m in MODELS:
        rdir = MODEL_DIRS.get(m)
        if not rdir or not rdir.exists():
            continue
            
        paths = [
            rdir / f"physics_evaluation_{m}_2020_ifs.csv",
            rdir / f"physics_evaluation_{m}_2020.csv",
            rdir / f"physics_summary_{m}_2020_ifs.csv",
            rdir / f"physics_summary_{m}_2020.csv",
            rdir / f"physics_summary_{m}_2020_vs_ifs.csv",
            rdir / f"physics_summary_{m}_2020_vs_era5.csv",
            rdir / f"physics_summary_{m}_s3_2022.csv",
            rdir / f"physics_summary_{m}_2022.csv"
        ]
        for p in paths:
            if p.exists():
                out[m] = pd.read_csv(p)
                break
    return out

def get_value(df: pd.DataFrame, metric: str) -> float:
    metric_col = next((c for c in ["metric_name", "metric", "variable", "name"] if c in df.columns), None)
    if not metric_col: return np.nan
    sub = df[df[metric_col] == metric]
    if sub.empty: return np.nan
    val_col = next((c for c in ["model_value", "mean_value", "mean_model", "value", "mean", "score"] if c in sub.columns), None)
    if not val_col: return np.nan
    val = sub[val_col].astype(float).mean()
    return float(val) if pd.notna(val) else np.nan

def fmt(val: float, metric: str) -> str:
    if np.isnan(val): return "—"
    return f"{val:+.4f}" if "drift" in metric else f"{val:.4f}"

# ── Table Plotting ───────────────────────────────────────────────────────────

def plot_conservation_table(outdir: Path, leads=[12, 120, 240]):
    outdir.mkdir(exist_ok=True)
    summaries = load_summaries()
    
    metrics_list = [
        ("dry_mass_drift_pct_per_day", "Dry Mass Drift →0 [%/day]"),
        ("water_mass_drift_pct_per_day", "Water Mass Drift →0 [%/day]"),
        ("total_energy_drift_pct_per_day", "Total Energy Drift →0 [%/day]"),
    ]
    
    header_color = np.array([0.9, 0.9, 0.9])
    red = np.array([1.0, 0.75, 0.75])
    white = np.array([1.0, 1.0, 1.0])

    model_labels = [NICE.get(m, m) for m in MODELS]
    max_abs = {}
    
    for metric, _ in metrics_list:
        vals = []
        for m in MODELS:
            if m in summaries:
                df = summaries[m]
                for lead in leads:
                    lead_col = next((c for c in ["lead_hours", "lead_time_hours", "lead_time", "forecast_hour"] if c in df.columns), None)
                    if not lead_col: continue
                    df_lt = df[df[lead_col] == lead]
                    if not df_lt.empty:
                        val = get_value(df_lt, metric)
                        if not np.isnan(val): vals.append(val)
        max_abs[metric] = max([abs(v) for v in vals]) if vals else 1.0
        if max_abs[metric] == 0: max_abs[metric] = 1.0

    cell_texts = [["Metric", "Lead Time"] + model_labels]
    cell_colors = [[header_color] * len(cell_texts[0])]
        
    for metric, m_label in metrics_list:
        for l_idx, lead in enumerate(leads):
            text_label = m_label if l_idx == len(leads)//2 else ""
            row_t = [text_label, f"{lead}h"]
            row_c = [white.copy(), white.copy()]

            for m in MODELS:
                val = np.nan
                if m in summaries:
                    df = summaries[m]
                    lead_col = next((c for c in ["lead_hours", "lead_time_hours", "lead_time", "forecast_hour"] if c in df.columns), None)
                    if lead_col:
                        df_lt = df[df[lead_col] == lead]
                        if not df_lt.empty:
                            val = get_value(df_lt, metric)
                            
                if np.isnan(val):
                    row_t.append("—")
                    row_c.append(white)
                else:
                    row_t.append(fmt(val, metric))
                    intensity = min(abs(val) / max_abs[metric], 1.0) * 0.8
                    row_c.append(white * (1 - intensity) + red * intensity)
                
            cell_texts.append(row_t)
            cell_colors.append(row_c)

    n_cols = len(cell_texts[0])
    n_rows = len(cell_texts)
    fig, ax = plt.subplots(figsize=(max(1.8 * n_cols, 12), max(0.3 * n_rows, 3.0)))
    ax.axis("off")
    
    colWidths = [0.45, 0.12] + [0.3] * len(MODELS)
    table = ax.table(
        cellText=cell_texts,
        cellColours=[[tuple(c) for c in row] for row in cell_colors],
        colWidths=colWidths, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.0, 1.6)

    for j in range(n_cols):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor(tuple(header_color))

    row_idx = 1
    for _ in metrics_list:
        for r in range(row_idx, row_idx + len(leads)):
            if r == row_idx: table[r, 0].visible_edges = 'LRT'
            elif r == row_idx + len(leads) - 1: table[r, 0].visible_edges = 'LRB'
            else: table[r, 0].visible_edges = 'LR'
            table[r, 0].set_text_props(fontweight="bold")
        row_idx += len(leads)

    fig.savefig(outdir / "ablation_table_conservation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved conservation summary table")

# ── Combined Plots ───────────────────────────────────────────────────────────

def plot_combined_conservation(outdir: Path):
    frames = []
    for m in MODELS:
        rdir = MODEL_DIRS.get(m)
        if not rdir or not rdir.exists():
            continue
            
        for path in rdir.glob(f"time_series_{m}_*.csv"):
            df = pd.read_csv(path)
            df["model"] = m
            frames.append(df)
            
    if not frames: 
        print("No timeseries data found for the selected models.")
        return
    df_all = pd.concat(frames, ignore_index=True)

    metrics = [("dry_mass_Eg", "Dry Air Mass (Eg)"),
               ("water_mass_kg", "Water Mass (kg)"),
               ("total_energy_J", "Total Energy (J)")]
               
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    handles_dict = {}
    for ax, (col, title) in zip(axes, metrics):
        for model in MODELS:
            mdf = df_all[df_all["model"] == model]
            if mdf.empty or col not in mdf.columns: continue
            
            rel_df = mdf[["date", "forecast_hour", col]].dropna().copy()
            if rel_df.empty: continue
            
            base = rel_df.sort_values("forecast_hour").groupby("date", as_index=False).first()[["date", col]].rename(columns={col: "base_val"})
            rel_df = rel_df.merge(base, on="date", how="left")
            rel_df = rel_df[rel_df["base_val"].abs() > 0]
            if rel_df.empty: continue
            
            rel_df["rel_pct"] = (rel_df[col] - rel_df["base_val"]) / rel_df["base_val"] * 100.0
            agg = rel_df.groupby("forecast_hour")["rel_pct"].agg(["mean", "std"]).reset_index()
            
            style = MODEL_STYLES.get(model, {"color": "grey", "marker": "."})
            label = NICE.get(model, model)
            
            line, = ax.plot(agg["forecast_hour"], agg["mean"], label=label, color=style["color"], marker=style["marker"], markersize=4)
            ax.fill_between(agg["forecast_hour"], agg["mean"] - agg["std"].fillna(0), agg["mean"] + agg["std"].fillna(0), color=style["color"], alpha=0.18, linewidth=0)
            
            if label not in handles_dict:
                handles_dict[label] = line
                
        ax.set_title(title, fontsize=28)
        ax.set_xlabel("Forecast Hour", fontsize=24)
        if ax == axes[0]: ax.set_ylabel("Relative Change (%)", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)

    if handles_dict:
        fig.legend(handles_dict.values(), handles_dict.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(MODELS), fontsize=22)
        
    fig.tight_layout()
    fig.savefig(outdir / "ablation_combined_conservation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved combined conservation plot")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(BASE_DIR / "plots_ifs_ablation"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {outdir}")
    print("Plotting Ablation Conservation Table...")
    plot_conservation_table(outdir)
    
    print("Plotting Combined Ablation Conservation Timeseries...")
    plot_combined_conservation(outdir)
    
    print("Done!")

if __name__ == "__main__":
    main()
