#!/bin/bash
# het_v14abs: absolute spin/q targets. The MCMC comparison (Sangria Case 1)
# proved the residual encoding CAPS the q/chi posteriors at the kernel width
# (model chi1 width ~0.07 vs MCMC ~0.7) and leaks truth via the truth-centered
# reference. Fix: q, c1, c2 become ABSOLUTE targets; Mc, t stay residual (they
# are sharper than spline resolution — the original reason for residuals).
# Warm-start from v13scale_r2 (encoder/bank fully transfer; flow adapts the 3
# re-purposed dims). Same 1e7 data, sky-GNPE + UCB-robust.
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
    --tag het_v14abs \
    --resume models/sbi_het_het_v13scale_r2_bestsky.pt \
    --epochs 110 --batch-size 256 --lr 6.0e-5 --weight-decay 1e-3 \
    --enc-hidden 384 --enc-blocks 12 --context-dim 768 \
    --n-transforms 40 --flow-hidden 768 --flow-bins 16 --bank-k 256 \
    --sky-weight 0.5 --residual-targets --abs-spins --ucb-cond --sky-gnpe \
    --patience 60 --device cuda:0
