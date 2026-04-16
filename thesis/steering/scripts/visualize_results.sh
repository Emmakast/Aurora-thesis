#!/bin/bash

# Activate the environment
source /home/ekasteleyn/aurora_thesis/aurora_env/bin/activate
cd /home/ekasteleyn/aurora_thesis/thesis/steering/scripts

# # Visualize AO (empty variant because we removed the suffix in the job script)
# /home/ekasteleyn/aurora_thesis/aurora_env/bin/python visualize_alphas_grid_global.py \
#     --phenomenon AO \
#     --variant "" \
#     --date 20170308

# # Visualize AAO (variant is "medium" based on the job script suffix)
# /home/ekasteleyn/aurora_thesis/aurora_env/bin/python visualize_alphas_grid_global.py \
#     --phenomenon AAO \
#     --variant medium \
#     --date 20160113

# # Visualize AO Polar (Focuses on North Pole)
# /home/ekasteleyn/aurora_thesis/aurora_env/bin/python visualize_alphas_grid.py \
#     --phenomenon AO \
#     --variant "" \
#     --date 20170308

# # Visualize AAO Polar (Focuses on South Pole)
# /home/ekasteleyn/aurora_thesis/aurora_env/bin/python visualize_alphas_grid.py \
#     --phenomenon AAO \
#     --variant medium \
#     --date 20160113

# ── NEW RESULTS ──────────────────────────────────────────────────────────────

# Visualize AO Global (High Index, variant "ao3")
/home/ekasteleyn/aurora_thesis/aurora_env/bin/python visualize_alphas_grid_global.py \
    --phenomenon AO \
    --variant ao3 \
    --date 20170308

# Visualize AAO Global (High Index, variant "high")
/home/ekasteleyn/aurora_thesis/aurora_env/bin/python visualize_alphas_grid_global.py \
    --phenomenon AAO \
    --variant high \
    --date 20160113

# Visualize AO Polar (High Index, variant "ao3")
/home/ekasteleyn/aurora_thesis/aurora_env/bin/python visualize_alphas_grid.py \
    --phenomenon AO \
    --variant ao3 \
    --date 20170308

# Visualize AAO Polar (High Index, variant "high")
/home/ekasteleyn/aurora_thesis/aurora_env/bin/python visualize_alphas_grid.py \
    --phenomenon AAO \
    --variant high \
    --date 20160113
