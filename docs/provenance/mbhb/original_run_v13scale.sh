#!/bin/bash
# het_v13scale: paper-scale NSF on ~1e7 samples. The amortized MEDIAN sky is
# data-limited (memory: v8 data-saturated at small flow; bigger flow alone left
# median ~2700; sky-GNPE broke it to 1700). Route to the paper's ~3-20 deg2
# regime = MORE DATA (1e7) + BIGGER FLOW (40-transform, 108M, toward their
# ResNet-50 + 50-transform). Keeps our wins: sky-GNPE + on-the-fly UCB-robust.
# From scratch (flow shape differs from v11robust). Chunk-rotated over 50 chunks.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bbhx_env
cd /home/asasli/NGB/lisa-smbhb-pe

CH=/mnt/wd_hdd_18TB/asasli
CHUNKS=""
for i in $(seq 48 96); do CHUNKS="$CHUNKS $CH/data_het_v6_c$i"; done

python -m pe.train_sbi_het \
    --data-dir data_het_v6 \
    --train-chunks $CHUNKS \
    --chunk-epochs 4 \
    --tag het_v13scale \
    --epochs 280 --batch-size 256 --lr 1.5e-4 --weight-decay 1e-3 \
    --enc-hidden 384 --enc-blocks 12 --context-dim 768 \
    --n-transforms 40 --flow-hidden 768 --flow-bins 16 --bank-k 256 \
    --sky-weight 0.5 --residual-targets --ucb-cond --sky-gnpe \
    --patience 80 --device cuda:0
