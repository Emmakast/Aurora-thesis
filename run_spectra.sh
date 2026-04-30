#!/bin/bash

echo "Plotting KE 850 hPa spectrum..."
aurora_env/bin/python ./thesis/benchmark/scripts/plot_spectrum.py ke_850hpa --results-dir ./thesis/benchmark/results

echo "Plotting Q spectrum..."
aurora_env/bin/python ./thesis/benchmark/scripts/plot_spectrum.py q --results-dir ./thesis/benchmark/results

echo "Generating summary tables..."
aurora_env/bin/python ./thesis/benchmark/scripts/plot_spectra_tables.py \
    --results-dir ./thesis/benchmark/results \
    --out-dir-ke850 ./thesis/benchmark/results/plots_ke_850 \
    --out-dir-q ./thesis/benchmark/results/plots_q_spec

echo "Done! All plots generated."
