#!/bin/bash
# Scale-up data campaign: generate chunks c64..c96 (33 x 200k = 6.6M) to bring
# the heterodyned v6-ridge dataset from 3.4M -> 1e7 (paper-scale). v6 ridge
# kernel (same as existing chunks); fresh seeds; train-only (val/test already in
# data_het_v6). ~10-15 min/chunk, GPU-bound (bbhx). Skips completed chunks.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bbhx_env
cd /home/asasli/NGB/lisa-smbhb-pe
CH=/mnt/wd_hdd_18TB/asasli
for i in $(seq 64 96); do
  OUT=$CH/data_het_v6_c$i
  if [ -f "$OUT/train/metadata.json" ]; then
    echo "[skip] c$i already done"; continue
  fi
  echo "=== generating c$i (seed $((70000+i))) $(date) ==="
  python data_generation/generate_het_1month.py \
      --output-dir $OUT --n-train 200000 --n-val 0 --n-test 0 --seed $((70000+i)) \
      2>&1 | grep -vE "it/s\]|warn|tdi|cupy|Follow|uninstall|conda|pip|http|----|^$" | tail -3
  echo "[done] c$i $(date); free: $(df -h $CH | tail -1 | awk '{print $4}')"
done
echo "CAMPAIGN COMPLETE $(date)"
