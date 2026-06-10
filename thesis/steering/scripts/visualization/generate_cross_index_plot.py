#!/usr/bin/env python3
"""
Visualize the cross-index evaluations.
For each steered phenomenon, plot how all 6 calculated indices respond to the steering alpha.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def parse_phenomenon(filename):
    # Filenames look like "base_nao_...", "steered_pna_..."
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[1].upper()
    return "UNKNOWN"

def main():
    csv_path = Path("/home/ekasteleyn/aurora_thesis/thesis/results/all_indices_evaluated.csv")
    out_path = Path("/home/ekasteleyn/aurora_thesis/thesis/results/cross_index_evaluation.png")
    
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Extract the steered phenomenon from the filename
    df['Steered_Phenomenon'] = df['Filename'].apply(parse_phenomenon)
    
    # The 6 evaluated indices
    indices = ["NAO", "PNA", "AAO", "AO", "MJO", "ENSO"]
    phenomena = ["AO", "AAO", "NAO", "PNA", "ENSO", "MJO"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()
    
    # Use a nice distinct color palette for the 6 lines
    colors = sns.color_palette("husl", 6)
    
    for i, phenom in enumerate(phenomena):
        ax = axes[i]
        subset = df[df['Steered_Phenomenon'] == phenom].copy()
        
        if subset.empty:
            ax.set_title(f"Steering {phenom} (No Data)", fontsize=14, fontweight="bold")
            continue
            
        subset = subset.sort_values(by="Alpha")
        
        # Plot each index's response
        for j, idx_name in enumerate(indices):
            # Emphasize the diagonal (i.e. if we steer NAO, make the NAO line thicker)
            is_target = (idx_name == phenom)
            linewidth = 4.0 if is_target else 2.0
            alpha_val = 1.0 if is_target else 0.6
            
            ax.plot(
                subset['Alpha'], 
                subset[idx_name], 
                marker='o', 
                linewidth=linewidth, 
                markersize=6,
                alpha=alpha_val,
                color=colors[j],
                label=idx_name
            )
            
        ax.set_title(f"Steering {phenom}", fontsize=14, fontweight="bold")
        ax.axvline(0, color='black', linestyle='--', alpha=0.5) # Mark alpha=0
        
        if i >= 3:
            ax.set_xlabel("Steering Magnitude (α)", fontsize=12, fontweight="bold")
        if i % 3 == 0:
            ax.set_ylabel("Index Value", fontsize=12, fontweight="bold")
            
        ax.grid(True, alpha=0.3)

    # Add a single shared legend at the top or right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Evaluated Index", loc='center right', 
               bbox_to_anchor=(1.08, 0.5), fontsize=12, title_fontsize=14)
               
    plt.suptitle("Cross-Index Evaluation: How Steering One Phenomenon Affects the Others", 
                 fontsize=18, fontweight="bold", y=1.02)
                 
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
