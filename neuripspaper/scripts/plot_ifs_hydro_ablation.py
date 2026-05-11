#!/usr/bin/env python3
"""
Script to compare IFS HRES against a modified version
using temperature (T) instead of virtual temperature (Tv) for hydrostatic balance.
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

MODELS = ["hres", "pangu", "graphcast", "neuralgcm"]
VARIANTS = ["Tv", "T"]

VARIANT_DIRS = {
    "Tv": BASE_DIR / "results_ifs",
    "T": BASE_DIR / "results_ifs_hydro",
}

NICE = {
    "graphcast": "GraphCast",
    "pangu": "Pangu",
    "hres": "HRES",
    "neuralgcm": "NeuralGCM",
}
MODEL_STYLES = {
    "aurora":    {"color": "#0072B2", "marker": "o"},
    "pangu":     {"color": "#D55E00", "marker": "s"},
    "fuxi":      {"color": "#009E73", "marker": "^"},
    "graphcast": {"color": "#000000", "marker": "D"},
    "neuralgcm": {"color": "#E69F00", "marker": "v"},
    "hres":      {"color": "#56B4E9", "marker": "P"},
}

def load_summaries() -> dict[tuple[str, str], pd.DataFrame]:
    out = {}
    for v in VARIANTS:
        rdir = VARIANT_DIRS.get(v)
        if not rdir or not rdir.exists():
            continue

        for m in MODELS:
            paths = [
                rdir / f"physics_evaluation_{m}_2020.csv",
                rdir / f"physics_evaluation_{m}_2020_ifs.csv",
                rdir / f"physics_summary_{m}_2020.csv",
                rdir / f"physics_summary_{m}_2020_ifs.csv",
            ]
            for p in paths:
                if p.exists():
                    out[(m, v)] = pd.read_csv(p)
                    break
    return out

def _get_ref_metric_mean(df: pd.DataFrame, metric: str) -> float:
    metric_col = next((c for c in ["metric_name", "metric", "variable", "name"] if c in df.columns), None)
    if not metric_col:
        return np.nan
    sub = df[df[metric_col] == metric]
    if sub.empty:
        return np.nan
    ref_col = next((c for c in ["ref_value", "mean_ref"] if c in sub.columns), None)
    if not ref_col:
        return np.nan
    vals = pd.to_numeric(sub[ref_col], errors="coerce")
    return float(vals.mean()) if vals.notna().any() else np.nan

def load_baselines(summaries: dict) -> dict[tuple[str, str], float]:
    """Retrieve precise reference baselines per model/variant."""
    bases = {}
    for m in MODELS:
        for v in VARIANTS:
            bases[(m, v)] = 0.0
            
            # 1. Fallback for NeuralGCM (low-res grid)
            if m == "neuralgcm":
                rdir = VARIANT_DIRS.get(v)
                txt_path = rdir / "ifs_0_7_rmse.txt" if rdir else None
                if txt_path and txt_path.exists():
                    with open(txt_path) as f:
                        for line in f:
                            if "hydrostatic_rmse" in line:
                                bases[(m, v)] = float(line.split(",")[1])
                else:
                    bases[(m, v)] = 262.391750661925
            
            # 2. Prefer explicit ref_value from summaries if available
            if (m, v) in summaries:
                df = summaries[(m, v)]
                val = _get_ref_metric_mean(df, "hydrostatic_rmse")
                if not np.isnan(val):
                    bases[(m, v)] = val
    return bases

def _extract_hydro_timeseries(df: pd.DataFrame, baseline: float = 0.0) -> pd.DataFrame:
    lead_col = next((c for c in ["forecast_hour", "lead_time_hours", "lead_hours", "lead_time"] if c in df.columns), None)
    if not lead_col:
        return pd.DataFrame(columns=["forecast_hour", "hydrostatic_rmse"])

    metric_col = next((c for c in ["metric_name", "metric", "variable", "name"] if c in df.columns), None)
    val_col = next((c for c in ["model_value", "mean_value", "mean_model", "value", "mean", "score"] if c in df.columns), None)
    ref_col = next((c for c in ["ref_value", "mean_ref"] if c in df.columns), None)

    if metric_col and val_col:
        sub = df[df[metric_col] == "hydrostatic_rmse"][[lead_col, val_col]].copy()
        if not sub.empty:
            if ref_col and df[ref_col].notna().any():
                sub["hydrostatic_rmse"] = pd.to_numeric(df.loc[sub.index, val_col], errors="coerce") - pd.to_numeric(df.loc[sub.index, ref_col], errors="coerce")
            else:
                sub["hydrostatic_rmse"] = pd.to_numeric(sub[val_col], errors="coerce") - baseline
            sub = sub.rename(columns={lead_col: "forecast_hour"})
            return sub.dropna(subset=["forecast_hour", "hydrostatic_rmse"])

    if "hydrostatic_rmse" in df.columns:
        out = df[[lead_col, "hydrostatic_rmse"]].copy()
        out = out.rename(columns={lead_col: "forecast_hour"})
        if "ref_hydrostatic_rmse" in df.columns:
            out["hydrostatic_rmse"] = pd.to_numeric(out["hydrostatic_rmse"], errors="coerce") - pd.to_numeric(df["ref_hydrostatic_rmse"], errors="coerce")
        else:
            out["hydrostatic_rmse"] = pd.to_numeric(out["hydrostatic_rmse"], errors="coerce") - baseline
        return out.dropna(subset=["forecast_hour", "hydrostatic_rmse"])

    return pd.DataFrame(columns=["forecast_hour", "hydrostatic_rmse"])

def get_delta(df: pd.DataFrame, metric: str, baseline: float = 0.0) -> float:
    metric_col = next((c for c in ["metric_name", "metric", "variable", "name"] if c in df.columns), None)
    if not metric_col: return np.nan
    sub = df[df[metric_col] == metric]
    if sub.empty: return np.nan
    
    val_col = next((c for c in ["model_value", "mean_value", "mean_model", "value", "mean", "score"] if c in sub.columns), None)
    ref_col = next((c for c in ["ref_value", "mean_ref"] if c in sub.columns), None)
    
    if val_col and ref_col and sub[ref_col].notna().any():
        valid = sub.dropna(subset=[val_col, ref_col])
        if valid.empty: return np.nan
        return float((valid[val_col].astype(float) - valid[ref_col].astype(float)).mean())
    elif val_col:
        val = pd.to_numeric(sub[val_col], errors="coerce").mean()
        return float(val) - baseline if pd.notna(val) else np.nan
    return np.nan

def fmt(val: float) -> str:
    if np.isnan(val): return "—"
    return f"{val:+.2f}"

# ── Table Plotting ───────────────────────────────────────────────────────────

def plot_hydrostatic_table(outdir: Path, leads=[12, 120, 240]):
    outdir.mkdir(exist_ok=True)
    summaries = load_summaries()
    baselines = load_baselines(summaries)
    
    header_color = np.array([0.9, 0.9, 0.9])
    red = np.array([1.0, 0.75, 0.75])
    white = np.array([1.0, 1.0, 1.0])

    model_labels = [NICE.get(m, m) for m in MODELS]
    
    metric = "hydrostatic_rmse"
    metric_label = "Hydrostatic\nRMSE Δ →0\n[m²/s²]"

    vals = []
    for m in MODELS:
        for v in VARIANTS:
            key = (m, v)
            if key in summaries:
                df = summaries[key]
                for lead in leads:
                    lead_col = next((c for c in ["lead_hours", "lead_time_hours", "lead_time", "forecast_hour"] if c in df.columns), None)
                    if not lead_col: continue
                    df_lt = df[df[lead_col] == lead]
                    if not df_lt.empty:
                        val = get_delta(df_lt, metric, baseline=baselines[(m, v)])
                        if not np.isnan(val): vals.append(val)
                    
    max_val = max([abs(v) for v in vals]) if vals else 1.0
    if max_val == 0: max_val = 1.0

    cell_texts = [["Metric", "Lead Time", "Formulation"] + model_labels]
    cell_colors = [[header_color] * len(cell_texts[0])]
        
    total_data_rows = len(leads) * len(VARIANTS)
    middle_data_row = total_data_rows // 2

    for l_idx, lead in enumerate(leads):
        for v_idx, variant in enumerate(VARIANTS):
            row_idx = l_idx * len(VARIANTS) + v_idx
            
            text_label = metric_label if row_idx == middle_data_row else ""
            lead_label = f"{lead}h" if v_idx == 0 else ""
            row_t = [text_label, lead_label, variant]
            row_c = [white.copy(), white.copy(), white.copy()]

            for m in MODELS:
                val = np.nan
                key = (m, variant)
                if key in summaries:
                    df = summaries[key]
                    lead_col = next((c for c in ["lead_hours", "lead_time_hours", "lead_time", "forecast_hour"] if c in df.columns), None)
                    if lead_col:
                        df_lt = df[df[lead_col] == lead]
                        if not df_lt.empty:
                            val = get_delta(df_lt, metric, baseline=baselines[(m, variant)])
                            
                if np.isnan(val):
                    row_t.append("—")
                    row_c.append(white)
                else:
                    row_t.append(fmt(val))
                    intensity = min(abs(val) / max_val, 1.0) * 0.8
                    row_c.append(white * (1 - intensity) + red * intensity)
                
            cell_texts.append(row_t)
            cell_colors.append(row_c)

    n_cols = len(cell_texts[0])
    n_rows = len(cell_texts)
    fig, ax = plt.subplots(figsize=(9, max(0.5 * n_rows, 3.5)))
    ax.axis("off")
    
    colWidths = [0.20, 0.14, 0.15] + [0.15] * len(MODELS)
    table = ax.table(
        cellText=cell_texts,
        cellColours=[[tuple(c) for c in row] for row in cell_colors],
        colWidths=colWidths, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.8)

    for j in range(n_cols):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor(tuple(header_color))

    for r in range(1, n_rows):
        # Handle Col 0 (Metric): Encompassing 1 massive row
        if r == 1:
            table[r, 0].visible_edges = 'LRT'
        elif r == n_rows - 1:
            table[r, 0].visible_edges = 'LRB'
        else:
            table[r, 0].visible_edges = 'LR'
        table[r, 0].set_text_props(fontweight="bold", verticalalignment="center")

        # Handle Col 1 (Lead Time): Encompass variants inside lead block
        internal_idx = (r - 1) % len(VARIANTS)
        if len(VARIANTS) == 1:
            table[r, 1].visible_edges = 'closed'
        elif internal_idx == 0:
            table[r, 1].visible_edges = 'LRT'
        elif internal_idx == len(VARIANTS) - 1:
            table[r, 1].visible_edges = 'LRB'
        else:
            table[r, 1].visible_edges = 'LR'
        table[r, 1].set_text_props(fontweight="bold", verticalalignment="center")

    fig.savefig(outdir / "hydrostatic_models_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved hydrostatic summary table")

# ── Timeseries Plotting ──────────────────────────────────────────────────────

def plot_hydrostatic_timeseries(outdir: Path):
    summaries = load_summaries()
    baselines = load_baselines(summaries)
    
    frames = []
    for v in VARIANTS:
        rdir = VARIANT_DIRS.get(v)
        if not rdir or not rdir.exists():
            continue

        for m in MODELS:
            candidate_paths = []
            candidate_paths += list(rdir.glob(f"time_series_{m}_*.csv"))
            candidate_paths += list(rdir.glob(f"physics_timeseries_{m}_*.csv"))
            candidate_paths += list(rdir.glob(f"hydrostatic_timeseries_{m}_*.csv"))

            # Fallback to evaluation file if no dedicated timeseries files exist
            if not candidate_paths:
                fallback = rdir / f"physics_evaluation_{m}_2020.csv"
                if fallback.exists():
                    candidate_paths = [fallback]

            for path in candidate_paths:
                df = pd.read_csv(path)
                ts = _extract_hydro_timeseries(df, baseline=baselines[(m, v)])
                if ts.empty:
                    continue
                ts["model"] = m
                ts["variant"] = v
                frames.append(ts)

    if not frames:
        print("No hydrostatic timeseries data found for the selected models.")
        return
    df_all = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    for model in MODELS:
        for variant in VARIANTS:
            mdf = df_all[(df_all["model"] == model) & (df_all["variant"] == variant)]
            if mdf.empty:
                continue

            agg = mdf.groupby("forecast_hour")["hydrostatic_rmse"].agg(["mean", "std"]).reset_index()
            if agg.empty:
                continue

            style = MODEL_STYLES.get(model, {"color": "grey", "marker": "."})
            label = f"{NICE.get(model, model)} ({variant})"
            ls = "-" if variant == "Tv" else "--"

            x = agg["forecast_hour"].values
            y = agg["mean"].values
            y_sigma = agg["std"].fillna(0.0).values

            ax.plot(x, y, label=label, color=style["color"], marker=style["marker"], markersize=5, linewidth=2, linestyle=ls)
            ax.fill_between(x, y - y_sigma, y + y_sigma, color=style["color"], alpha=0.15, linewidth=0)

    ax.set_title("Hydrostatic Balance Comparison (4 Models)", fontsize=28)
    ax.set_xlabel("Forecast Hour", fontsize=24)
    ax.set_ylabel("Δ RMSE (m²/s²)", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    
    fig.tight_layout()
    fig.savefig(outdir / "hydrostatic_models_timeseries.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved hydrostatic timeseries plot")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(BASE_DIR / "plots_ifs_hydro_ablation"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {outdir}")
    print("Plotting Hydrostatic Ablation Table...")
    plot_hydrostatic_table(outdir)
    
    print("Plotting Hydrostatic Ablation Timeseries...")
    plot_hydrostatic_timeseries(outdir)
    
    print("Done!")

if __name__ == "__main__":
    main()
