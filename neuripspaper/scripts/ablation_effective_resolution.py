import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("/home/ekasteleyn/aurora_thesis/neuripspaper/results")
PLOTS_DIR = Path("/home/ekasteleyn/aurora_thesis/neuripspaper/plots")
EARTH_RADIUS_KM = 6371.0
MODELS = ["HRES", "Pangu", "GraphCast", "NeuralGCM", "FuXi"]

def _find_effective_resolution(
    k: np.ndarray, e_pred: np.ndarray, e_true: np.ndarray, threshold: float, n_consecutive: int
) -> float:
    """Find effective resolution where e_pred/e_true < threshold for n_consecutive wavenumbers."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = e_pred / e_true
    
    below = ratio < threshold
    n = len(ratio)
    fallback = 2.0 * np.pi * EARTH_RADIUS_KM / (float(k[-1]) if len(k) else 1.0)

    if n < n_consecutive:
        return fallback

    run = 0
    for i in range(n):
        if below[i]:
            run += 1
            if run >= n_consecutive:
                idx = i - n_consecutive + 1
                return 2.0 * np.pi * EARTH_RADIUS_KM / float(k[idx])
        else:
            run = 0
            
    return fallback

def _color_gradient(val, min_val=111.5, max_val_ref=500.0):
    """Return an RGB tuple transitioning from white to light red (#FFBDBD)."""
    if pd.isna(val) or type(val) == str:
        return np.array([1.0, 1.0, 1.0])
        
    white = np.array([1.0, 1.0, 1.0])
    light_red = np.array([1.0, 189/255, 189/255])  # #FFBDBD
    
    intensity = min(max(val - min_val, 0) / max(max_val_ref - min_val, 1.0), 1.0)
    return white * (1 - intensity) + light_red * intensity

def plot_grouped_ablation_table(
    df: pd.DataFrame, models: list[str], leads: list[int],
    param_col: str, param_vals: list, param_label: str, out_path: Path
):
    """Render a colored Matplotlib table with parameter changing on rows and models on columns."""
    # Find max visualization value for gradient map (excluding nans/strings)
    max_val = df["eff_res"].max()
    if pd.isna(max_val) or max_val <= 111.5:
        max_val = 500.0

    header_color = np.array([230/255, 230/255, 230/255])  # #E6E6E6
    white = np.array([1.0, 1.0, 1.0])

    cell_texts = [[param_label, "Lead Time"] + models]
    cell_colors = [[header_color] * len(cell_texts[0])]

    for p_val in param_vals:
        for l_idx, lead in enumerate(leads):
            # Center the parameter label vertically by putting it in the middle row of the group
            t_label = f"{p_val}" if l_idx == len(leads)//2 else ""
            row_t = [t_label, f"{lead}h"]
            row_c = [white.copy(), white.copy()]

            for m in models:
                sub = df[(df["Model"] == m) & (df[param_col] == p_val) & (df["lead_hours"] == lead)]
                if sub.empty or pd.isna(sub.iloc[0]["eff_res"]):
                    row_t.append("—")
                    row_c.append(white.copy())
                else:
                    v = sub.iloc[0]["eff_res"]
                    row_t.append(f"{v:.1f}")
                    row_c.append(_color_gradient(v, 111.5, max_val))
                    
            cell_texts.append(row_t)
            cell_colors.append(row_c)

    n_cols = len(cell_texts[0])
    n_rows = len(cell_texts)
    
    fig, ax = plt.subplots(figsize=(max(1.8 * n_cols, 12), max(0.35 * n_rows, 3.0)))
    ax.axis("off")
    
    colWidths = [0.20, 0.15] + [0.15] * len(models)
    table = ax.table(
        cellText=cell_texts,
        cellColours=[[tuple(c) for c in r] for r in cell_colors],
        colWidths=colWidths, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.0, 1.8)

    # Style Headers
    for j in range(n_cols):
        table[0, j].set_text_props(fontweight="bold")

    # Group table borders for the parameter column
    row_idx = 1
    for _ in param_vals:
        for r in range(row_idx, row_idx + len(leads)):
            if r == row_idx: table[r, 0].visible_edges = 'LRT'
            elif r == row_idx + len(leads) - 1: table[r, 0].visible_edges = 'LRB'
            else: table[r, 0].visible_edges = 'LR'
            table[r, 0].set_text_props(fontweight="bold")
        row_idx += len(leads)
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    thresholds = [0.3, 0.5, 0.7, 0.9]
    n_consecutive_vals = [1, 3, 5, 7]
    leads = [12, 120, 240]
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for model in MODELS:
        filepath = RESULTS_DIR / f"spectra_{model.lower()}_2020.csv"
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue
            
        df = pd.read_csv(filepath)
        df = df[df["variable"] == "KE"]
        
        for lead in leads:
            sub_df = df[df["lead_hours"] == lead]
            if sub_df.empty:
                continue
                
            # Average across dates first
            agg = sub_df.groupby("wavenumber")[["power_pred", "power_ref"]].mean().reset_index()
            agg = agg[agg["power_ref"] > 1e-12].sort_values("wavenumber")
            
            k = agg["wavenumber"].values
            e_pred = agg["power_pred"].values
            e_ref = agg["power_ref"].values
            
            for thr in thresholds:
                for n in n_consecutive_vals:
                    eff_res = _find_effective_resolution(k, e_pred, e_ref, thr, n)
                    results.append({
                        "Model": model,
                        "lead_hours": lead,
                        "threshold": thr,
                        "n_consecutive": n,
                        "eff_res": eff_res
                    })
        
    if not results:
        print("No valid data found to run ablation.")
        return
        
    df_results = pd.DataFrame(results)
    
    # Save Raw CSV
    csv_path = PLOTS_DIR / "ablation_eff_res_raw.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved raw ablation data to {csv_path}")
    
    # Table 1: Varying Thresholds (fix n_consecutive = 5)
    df_thr = df_results[df_results["n_consecutive"] == 5]
    if not df_thr.empty:
        png_path_thr = PLOTS_DIR / "ablation_eff_res_thresholds.png"
        plot_grouped_ablation_table(
            df_thr, MODELS, leads, 
            param_col="threshold", param_vals=thresholds, param_label="Threshold (n=5)", 
            out_path=png_path_thr
        )
        print(f"Saved threshold ablation table to {png_path_thr}")

    # Table 2: Varying n_consecutive (fix threshold = 0.5)
    df_n = df_results[df_results["threshold"] == 0.5]
    if not df_n.empty:
        png_path_n = PLOTS_DIR / "ablation_eff_res_n_consec.png"
        plot_grouped_ablation_table(
            df_n, MODELS, leads, 
            param_col="n_consecutive", param_vals=n_consecutive_vals, param_label="Cons. Points", 
            out_path=png_path_n
        )
        print(f"Saved n_consecutive ablation table to {png_path_n}")

if __name__ == "__main__":
    main()
