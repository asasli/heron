#!/bin/bash
# het_v15wspin: wide-spin-reference data (v7) + absolute q/chi targets.
# The complete fix for reference-anchored spins: uninformative chi refs force
# the network to learn spin from the DATA. Warm-start v14abs.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bbhx_env
cd /home/asasli/NGB/lisa-smbhb-pe
CH=/mnt/wd_hdd_18TB/asasli
CHUNKS=""
for i in $(seq 1 49); do CHUNKS="$CHUNKS $CH/data_het_v7_c$i"; done
ln -sfn $CH/data_het_v7 data_het_v7
python -m pe.train_sbi_het \
    --data-dir data_het_v7 \
    --train-chunks $CHUNKS \
    --chunk-epochs 4 \
    --tag het_v15wspin \
    --resume models/sbi_het_het_v14abs_bestsky.pt \
    --epochs 130 --batch-size 256 --lr 6.0e-5 --weight-decay 1e-3 \
    --enc-hidden 384 --enc-blocks 12 --context-dim 768 \
    --n-transforms 40 --flow-hidden 768 --flow-bins 16 --bank-k 256 \
    --sky-weight 0.5 --residual-targets --abs-spins --ucb-cond --sky-gnpe \
    --patience 60 --device cuda:0
