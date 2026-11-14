#!/bin/bash
# v7 data campaign: WIDE-SPIN references (uninformative chi refs, env switch).
# Base dir (train+val+test) + 49 train chunks = 1e7 total. Resumable.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bbhx_env
cd /home/asasli/NGB/lisa-smbhb-pe
export WIDE_SPIN_REF=1
CH=/mnt/wd_hdd_18TB/asasli

if [ ! -f "$CH/data_het_v7/test/metadata.json" ]; then
  echo "=== base data_het_v7 (train 200k + val 5k + test 2k) $(date) ==="
  python data_generation/generate_het_1month.py \
      --output-dir $CH/data_het_v7 --n-train 200000 --n-val 5000 --n-test 2000 --seed 90000 \
      2>&1 | grep -vE "it/s\]|warn|tdi|cupy|Follow|uninstall|conda|pip|http|----|^$" | tail -3
fi
for i in $(seq 1 49); do
  OUT=$CH/data_het_v7_c$i
  [ -f "$OUT/train/metadata.json" ] && { echo "[skip] c$i"; continue; }
  echo "=== v7 chunk c$i (seed $((90000+i))) $(date) ==="
  python data_generation/generate_het_1month.py \
      --output-dir $OUT --n-train 200000 --n-val 0 --n-test 0 --seed $((90000+i)) \
      2>&1 | grep -vE "it/s\]|warn|tdi|cupy|Follow|uninstall|conda|pip|http|----|^$" | tail -2
  echo "[done] c$i $(date); free $(df -h $CH | tail -1 | awk '{print $4}')"
done
echo "V7 CAMPAIGN COMPLETE $(date)"
