#!/usr/bin/env python3
"""
Render a colour-coded summary table for `q_spectrum` and `850hpa ke_spectrum`.
It calculates spectral metrics on the fly using the time-averaged spectra.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# Companion library (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from physics_metrics import _find_effective_resolution, compute_spectral_scores

# ── Config ───────────────────────────────────────────────────────────────────

# Enforce the correct plotting and table order
MODELS = ["hres", "pangu", "graphcast", "neuralgcm", "fuxi", "aurora"]
NICE = {
    "hres": "HRES", "pangu": "Pangu", "graphcast": "GraphCast",
    "neuralgcm": "NeuralGCM", "fuxi": "FuXi", "aurora": "Aurora",
}
NATIVE_RESOLUTION_MODELS = ["hres", "neuralgcm"]

# Metrics to show (NeurIPS-style spectral block)
METRICS = {
    "effective_resolution_km": ("Eff. Resolution [km]",),
    "spectral_residual": ("Spec. Residual ↓0",),
    "spectral_wasserstein": ("Spec. W-Dist ↓0",),
}

def load_and_calculate_metrics(results_dir: Path, spectrum_type: str, ifs_mode: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    suffix = "_ifs" if ifs_mode else ""
    
    value_col = "power" if spectrum_type == "q_spectrum" else "energy"
    
    for m in MODELS:
        # Resolve path
        if m == "aurora":
            p = results_dir / f"{spectrum_type}_aurora_s3_2022{suffix}.csv"
        else:
            p = results_dir / f"{spectrum_type}_{m}_2020{suffix}.csv"
            
        if not p.exists() and m == "hres" and not ifs_mode:
            # Fallback for HRES
            pass

        if not p.exists():
            continue
            
        print(f"Loading {p.name} ...")
        df = pd.read_csv(p)
        
        # Calculate mean spectrum across dates
        mean_df = df.groupby(["model", "lead_hours", "wavenumber", "source"], as_index=False)[value_col].mean()
        
        # Compute metrics per lead time
        metrics_rows = []
        for lead in mean_df["lead_hours"].unique():
            sub = mean_df[mean_df["lead_hours"] == lead].sort_values("wavenumber")
            
            # Separate pred and era5
            pred_sub = sub[sub["source"] == "pred"]
            era5_sub = sub[sub["source"] == "era5"]
            
            if pred_sub.empty or era5_sub.empty:
                continue
                
            # align wavenumbers
            common_wn = np.intersect1d(pred_sub["wavenumber"], era5_sub["wavenumber"])
            
            pred_vals = pred_sub.set_index("wavenumber").loc[common_wn, value_col].values
            era5_vals = era5_sub.set_index("wavenumber").loc[common_wn, value_col].values
            
            try:
                eff_out = _find_effective_resolution(common_wn, pred_vals, era5_vals)
                L_eff = eff_out[0] if isinstance(eff_out, tuple) else eff_out
            except Exception:
                L_eff = np.nan
                
            try:
                s_div, s_res = compute_spectral_scores(pred_vals, era5_vals)
                # Use wavenumbers as coordinates and powers as the distribution weights
                w_dist = wasserstein_distance(common_wn, common_wn, u_weights=pred_vals, v_weights=era5_vals)
            except Exception:
                s_div, s_res, w_dist = np.nan, np.nan, np.nan
                
            metrics_rows.append({"metric_name": "effective_resolution_km", "lead_time_hours": lead, "mean_value": L_eff})
            metrics_rows.append({"metric_name": "spectral_wasserstein", "lead_time_hours": lead, "mean_value": w_dist})
            metrics_rows.append({"metric_name": "spectral_residual", "lead_time_hours": lead, "mean_value": s_res})

        if metrics_rows:
            out[m] = pd.DataFrame(metrics_rows)
            print(f"  -> Computed metrics for {m}")
        else:
            if m == "fuxi" and spectrum_type == "q_spectrum":
                print("  -> Skipped fuxi q_spectrum: CSV contains only era5 rows, no pred spectrum to compare.")
            
    return out

def get_value(df: pd.DataFrame, metric: str) -> float:
    row = df[df["metric_name"] == metric]
    if row.empty:
        return np.nan
    return float(row.iloc[0]["mean_value"])

def fmt(val: float, metric: str) -> str:
    if np.isnan(val):
        return "—"
    if metric == "effective_resolution_km":
        return f"{val:.1f}"
    if abs(val) > 0 and abs(val) < 0.0001:
        return f"{val:.2e}"
    return f"{val:.4f}"


def render_table(spectrum_type: str, leads: list[int], summaries: dict[str, pd.DataFrame], outdir: Path):
    metrics_list = list(METRICS.keys())
    model_cols = [m for m in MODELS if m in summaries]
    if not model_cols:
        return

    model_labels = [NICE.get(m, m) for m in model_cols]
    header_color = np.array([0.9, 0.9, 0.9])
    red = np.array([1.0, 0.75, 0.75])
    white = np.array([1.0, 1.0, 1.0])

    # Precompute max abs per metric for red-only intensity scaling.
    max_abs = {}
    for metric in metrics_list:
        vals = []
        for m in model_cols:
            if metric == "effective_resolution_km" and m in NATIVE_RESOLUTION_MODELS:
                continue
            df = summaries[m]
            for lead in leads:
                df_lt = df[df["lead_time_hours"] == lead]
                if df_lt.empty:
                    avail = sorted(df["lead_time_hours"].dropna().unique())
                    if avail:
                        nearest = min(avail, key=lambda x: abs(x - lead))
                        df_lt = df[df["lead_time_hours"] == nearest]
                if not df_lt.empty:
                    val = get_value(df_lt, metric)
                    if not np.isnan(val):
                        vals.append(val)
        max_abs[metric] = max([abs(v) for v in vals]) if vals else 1.0
        if max_abs[metric] == 0:
            max_abs[metric] = 1.0

    cell_texts = [["Metric", "Lead Time"] + model_labels]
    cell_colors = [[header_color] * len(cell_texts[0])]

    for metric in metrics_list:
        metric_label = METRICS[metric][0]
        for l_idx, lead in enumerate(leads):
            row_t = [metric_label if l_idx == len(leads) // 2 else "", f"{lead}h"]
            row_c = [white.copy(), white.copy()]

            for m in model_cols:
                df = summaries[m]
                suffix = ""
                df_lt = df[df["lead_time_hours"] == lead]
                if df_lt.empty:
                    avail = sorted(df["lead_time_hours"].dropna().unique())
                    if avail:
                        nearest = min(avail, key=lambda x: abs(x - lead))
                        df_lt = df[df["lead_time_hours"] == nearest]
                        suffix = " *"

                val = get_value(df_lt, metric) if not df_lt.empty else np.nan
                if np.isnan(val):
                    row_t.append("—")
                    row_c.append(white)
                elif metric == "effective_resolution_km" and m in NATIVE_RESOLUTION_MODELS:
                    row_t.append(fmt(val, metric) + suffix)
                    row_c.append(white)
                else:
                    row_t.append(fmt(val, metric) + suffix)
                    intensity = min(abs(val) / max_abs[metric], 1.0) * 0.8
                    row_c.append(white * (1 - intensity) + red * intensity)

            cell_texts.append(row_t)
            cell_colors.append(row_c)

    n_cols = len(cell_texts[0])
    n_rows = len(cell_texts)
    fig, ax = plt.subplots(figsize=(max(1.8 * n_cols, 12), max(0.3 * n_rows, 3.0)))
    ax.axis("off")

    col_widths = [0.45, 0.12] + [0.15] * len(model_cols)
    table = ax.table(
        cellText=cell_texts,
        cellColours=[[tuple(c) for c in row] for row in cell_colors],
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
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
            if r == row_idx:
                table[r, 0].visible_edges = "LRT"
            elif r == row_idx + len(leads) - 1:
                table[r, 0].visible_edges = "LRB"
            else:
                table[r, 0].visible_edges = "LR"
            table[r, 0].set_text_props(fontweight="bold")
        row_idx += len(leads)

    out = outdir / f"summary_table_{spectrum_type}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional single output directory for both tables (legacy behavior).",
    )
    parser.add_argument(
        "--out-dir-ke850",
        type=str,
        default=None,
        help="Output directory for KE 850 hPa spectral summary table.",
    )
    parser.add_argument(
        "--out-dir-ke500",
        type=str,
        default=None,
        help="Output directory for KE 500 hPa spectral summary table.",
    )
    parser.add_argument(
        "--out-dir-q",
        type=str,
        default=None,
        help="Output directory for Q-spectrum summary table.",
    )
    parser.add_argument("--leads", nargs="+", type=int, default=[12, 120, 240])
    parser.add_argument("--ifs", action="store_true")
    
    args = parser.parse_args()
    
    rdir = Path(args.results_dir)

    # Resolve destination folders.
    if args.out_dir:
        out_dir_q = Path(args.out_dir)
        out_dir_ke = Path(args.out_dir)
        out_dir_ke500 = Path(args.out_dir)
    else:
        out_dir_q = Path(args.out_dir_q) if args.out_dir_q else (rdir / "plots_q_spec")
        out_dir_ke = Path(args.out_dir_ke850) if args.out_dir_ke850 else (rdir / "plots_ke_850")
        out_dir_ke500 = Path(args.out_dir_ke500) if args.out_dir_ke500 else (rdir / "plots_ke_spectrum")

    out_dir_q.mkdir(parents=True, exist_ok=True)
    out_dir_ke.mkdir(parents=True, exist_ok=True)
    out_dir_ke500.mkdir(parents=True, exist_ok=True)

    for stype in ["q_spectrum", "ke_spectrum_850hpa", "ke_spectrum"]:
        print(f"Processing {stype}...")
        summaries = load_and_calculate_metrics(rdir, stype, args.ifs)
        if summaries:
            if stype == "q_spectrum":
                target_out = out_dir_q
            elif stype == "ke_spectrum_850hpa":
                target_out = out_dir_ke
            else:
                target_out = out_dir_ke500
            render_table(stype, args.leads, summaries, target_out)

if __name__ == "__main__":
    main()
