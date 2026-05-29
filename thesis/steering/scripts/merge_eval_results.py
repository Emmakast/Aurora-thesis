import pandas as pd
from pathlib import Path

source_dir = Path("/scratch-shared/ekasteleyn/ao_neutral_steered")
output_file = Path("/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/ao_eval_results_merged.csv")

all_dfs = []

for file_path in source_dir.glob("eval_results_*.csv"):
    date_str = file_path.stem.split("_")[2]
    # date_str is like '20160123'. Let's format it nicely to YYYY-MM-DD
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    
    try:
        df = pd.read_csv(file_path)
        df.insert(0, "Date", formatted_date)
        all_dfs.append(df)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")

if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    # Sort by Date and Alpha
    final_df.sort_values(by=["Date", "Alpha"], inplace=True)
    final_df.to_csv(output_file, index=False)
    print(f"Successfully merged {len(all_dfs)} files into {output_file}")
else:
    print("No CSV files found to merge.")
