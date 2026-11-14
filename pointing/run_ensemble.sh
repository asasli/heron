#!/usr/bin/env bash
# Population/noise ensemble for the pointing tier (Fig. 5 error bars).
#
# Step 1 (any machine with bbhx): generate N hand-off data sets, one per
#   noise realization (and/or Sangria case). Each carries its own alert
#   posterior + noiseless template.
# Step 2 (a machine with the HyperWave package + a GPU): run the alert-seeded
#   PE on every subtracted file. Then aggregate with aggregate_ensemble.py.
#
# Usage:
#   bash pointing/run_ensemble.sh generate 12        # -> exports/ens_*
#   bash pointing/run_ensemble.sh localize           # -> results/ensemble/*
set -euo pipefail

N="${2:-12}"
EXPORTS="exports/ensemble"
RESULTS="results/ensemble"
BACKEND="${HW_BACKEND:-cuda12x}"

case "${1:-}" in
  generate)
    mkdir -p "$EXPORTS"
    for s in $(seq 0 $((N-1))); do
      for c in 1 2; do
        echo "[gen] case $c seed $s"
        python pointing/export.py --case "$c" --seed "$s" --n-ucb 30 \
          --tag "ens_c${c}_s${s}" --out "$EXPORTS"
      done
    done
    echo "generated $((2*N)) data sets in $EXPORTS/"
    ;;
  localize)
    mkdir -p "$RESULTS"
    for f in "$EXPORTS"/*_subtracted.npz; do
      base=$(basename "$f" _subtracted.npz)
      echo "[loc] $base"
      python pointing/synergy_pe.py --npz "$f" --seed-mode alert \
        --force-backend "$BACKEND" --out-dir "$RESULTS" \
        --outfile "$RESULTS/${base}.npz"
    done
    echo "localized $(ls "$RESULTS"/*.npz | wc -l) events -> $RESULTS/"
    echo "now: python pointing/aggregate_ensemble.py --dir $RESULTS"
    ;;
  *)
    echo "usage: $0 {generate N | localize}"; exit 1 ;;
esac
