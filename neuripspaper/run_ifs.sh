#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python "$DIR/scripts/plot_neurips_metrics.py" \
    --results-dir "$DIR/results_ifs" \
    --outdir "$DIR/plots_ifs"
