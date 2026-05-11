#!/bin/bash
# Make sure to run this from the aurora_thesis directory

# 1. Generate Q spectrum plots
python thesis/benchmark/scripts/plot_spectrum.py q \
    --results-dir thesis/benchmark/results

# 2. Generate KE 850hPa spectrum plots
python thesis/benchmark/scripts/plot_spectrum.py ke_850hpa \
    --results-dir thesis/benchmark/results

# 3. Generate summary tables for both Q and KE 850hPa
python thesis/benchmark/scripts/plot_spectra_tables.py \
    --results-dir thesis/benchmark/results \
    --out-dir-ke850 thesis/benchmark/results/plots_ke_850 \
    --out-dir-q thesis/benchmark/results/plots_q_spec

echo "Done! Plots saved to thesis/benchmark/results/plots_ke_850 and thesis/benchmark/results/plots_q_spec"
