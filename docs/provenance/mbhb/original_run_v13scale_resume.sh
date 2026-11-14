#!/bin/bash
# Resume v13scale after the Jun-26 22:58 machine reboot killed it at E152/280.
# Continues from the E152 bestsky checkpoint to complete the back-third cosine
# anneal (where the sky sharpens most). New tag het_v13scale_r2 so the E152
# checkpoint stays as a safe fallback. Persistent log (logs/, survives reboot).
# Norm stats recomputed deterministically from data_het_v6 (seed 123) == match
# the original v13scale norm stats, so the loaded weights see consistent inputs.
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
    --tag het_v13scale_r2 \
    --resume models/sbi_het_het_v13scale_bestsky.pt \
    --epochs 130 --batch-size 256 --lr 1.0e-4 --weight-decay 1e-3 \
    --enc-hidden 384 --enc-blocks 12 --context-dim 768 \
    --n-transforms 40 --flow-hidden 768 --flow-bins 16 --bank-k 256 \
    --sky-weight 0.5 --residual-targets --ucb-cond --sky-gnpe \
    --patience 60 --device cuda:0
