import pandas as pd
from pathlib import Path

source_dir = Path("/scratch-shared/ekasteleyn/ao_neutral_steered")
output_path = Path("/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/81_eval.csv")

all_dfs = []
for f in source_dir.glob("eval_results_*.csv"):
    date_tag = f.stem.replace("eval_results_", "")
    
    # Format date_tag (YYYYMMDD) back to YYYY-MM-DD
    if len(date_tag) == 8:
        formatted_date = f"{date_tag[:4]}-{date_tag[4:6]}-{date_tag[6:]}"
    else:
        formatted_date = date_tag
        
    df = pd.read_csv(f)
    df["Date"] = formatted_date
    all_dfs.append(df)

if all_dfs:
    master_df = pd.concat(all_dfs, ignore_index=True)
    # Order nicely
    master_df = master_df[["Date", "Alpha", "AO_Index_Diff"]]
    master_df = master_df.sort_values(["Date", "Alpha"])
    master_df.to_csv(output_path, index=False)
    print(f"Successfully merged {len(all_dfs)} files into {output_path}")
else:
    print("No CSV files found!")
