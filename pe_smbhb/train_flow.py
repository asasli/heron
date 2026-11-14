"""
The alert flow (stage 2): model, dataset, foreground augmentation, training.

    FlowHet        residual CNN encoder (FiLM-conditioned on the reference, the
                   UCB count and the sky pose) + matched-filter bank + neural
                   spline flow over the eleven MBHB parameters
    HetDataset     heterodyned training samples; noise is added on the fly
    inject_ucb_foreground, draw_n_ucb   resolved and faint UCB lines, resampled every batch
    sky_pose_meta, draw_sky_sigma       blurred sky pose for sky-GNPE
    residual_targets, residual_to_physical   kernel-scaled residual encoding of Mc and t

Training (published configuration and its four phases: config/heron.yaml,
docs/TRAINING_PROVENANCE.md; `python -m heron.run` issues these commands):

    python -m pe_smbhb.train_flow --data-dir data/mbhb \
        --train-chunks data/mbhb_c{1..49} --chunk-epochs 4 --tag mbhb \
        --epochs 130 --batch-size 256 --lr 6e-5 \
        --enc-hidden 384 --enc-blocks 12 --context-dim 768 \
        --n-transforms 40 --flow-hidden 768 --flow-bins 16 --bank-k 256 \
        --sky-weight 0.5 --residual-targets --abs-spins --ucb-cond --sky-gnpe \
        [--resume <checkpoint>] [--seed 0]

Outputs in --save-dir: sbi_het_<tag>_best.pt (best validation loss),
sbi_het_<tag>_bestsky.pt (best auxiliary sky loss; the released model is this
checkpoint) and norm_stats_het_<tag>.pt.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import MBHB_FLOW, MBHB_DATA, UCB_RESPONSE

from pe_smbhb.networks import (PARAM_NAMES, PARAM_RANGES, N_PARAMS,
                              normalize_params, from_model_space,
                              Encoder11D, TransformerEncoder11D)


def make_encoder(encoder_type, in_channels, hidden, n_blocks, context_dim, n_meta):
    """Encoder factory: 'cnn' (Encoder11D) or 'transformer' (TransformerEncoder11D).
    Both share forward(x:[B,n_bins,ch], meta:[B,n_meta]) -> [B,context_dim]."""
    if encoder_type == 'transformer':
        return TransformerEncoder11D(in_channels=in_channels, context_dim=context_dim,
                                     n_meta=n_meta)
    return Encoder11D(in_channels=in_channels, hidden=hidden, n_blocks=n_blocks,
                      context_dim=context_dim, n_meta=n_meta)

N_META = 9
T_PRE_MAX_H = 605.0
KSCALE_MIN, KSCALE_MAX = 0.02, 4.0


def _ks_norm(x):
    return (np.log10(x) - np.log10(KSCALE_MIN)) \
        / (np.log10(KSCALE_MAX) - np.log10(KSCALE_MIN))


# Sky-GNPE
# The coherent sky posterior is near-delta (<3 deg^2) but its LOCATION swings
# wildly across parameter space, so a plain flow hedges broad (median plateaus
# ~2700 deg^2 << ceiling). Fix: condition the network on an APPROXIMATE sky
# pose (true sky + blur, in normalized [0,1]^2 of (sin_beta, lambda)); the
# network then learns a small, smooth RESIDUAL it can place sharply. At
# inference we iterate: broad seed -> posterior -> tighter seed -> ... (Gibbs).
# sigma is in normalized units (0.5 = whole-prior-ish; 0.01 ~ 1 deg near eq).
SKY_SIGMA_MIN, SKY_SIGMA_MAX = 3e-3, 1.0
N_SKY_META = 4
# UCB foreground injected ON THE FLY (resampled every batch) into the existing
# clean data, so the model learns the SMBHB waveform robustly against an
# ever-changing galactic-binary foreground, conditioned only on n_ucb (the
# RESOLVED count YOLO provides). The diffuse Gaussian confusion is negligible
# here (instrument-dominated band, see evaluation/calibrate_ucb.py), so the
# foreground IS the superposition of UCB lines: n_ucb bright resolved + a
# varying number of faint unresolved (the 'SGWB from UCBs').
UCB_N_MAX_COND = 40
N_UCB_META = 1
# per-bin SNR ~ 10^(log10A+21)*30; resolved (YOLO, n_ucb) vs sub-threshold foreground
UCB_LOG10A_RES = (-21.5, -20.3)    # per-bin SNR ~9-150 (resolved, conditioned)
UCB_LOG10A_FAINT = (-23.5, -21.7)  # per-bin SNR ~0.1-6 (unresolved confusion)
UCB_N_FAINT_MAX = 500              # many faint -> 'SGWB from UCBs', varies/sample


def inject_ucb_foreground(fd, n_ucb, g_bin, band_bins, rng=None):
    """Add resolved + faint UCB lines to fd[:, :, :4] (raw whitened-binned
    units) IN PLACE. fd: (B, n_bins, 5); n_ucb: (B,) resolved counts. A UCB at
    output bin k with amplitude 10^a contributes 10^a * g_bin[k] with random
    phase to A and E. Faint population (the SGWB) varies per sample."""
    device = fd.device
    B = fd.shape[0]
    if rng is None:
        randint = lambda hi, sz: torch.randint(0, hi, sz, device=device)
        rand = lambda sz: torch.rand(sz, device=device)
    else:
        randint = lambda hi, sz: torch.randint(0, hi, sz, generator=rng, device=device)
        rand = lambda sz: torch.rand(sz, generator=rng, device=device)
    nb = band_bins.shape[0]
    n_faint = randint(UCB_N_FAINT_MAX + 1, (B,))   # varying SGWB realization
    for b in range(B):
        nres = int(n_ucb[b]); nf = int(n_faint[b])
        tot = nres + nf
        if tot == 0:
            continue
        bins = band_bins[randint(nb, (tot,))]
        a = torch.empty(tot, device=device)
        a[:nres] = rand((nres,)) * (UCB_LOG10A_RES[1] - UCB_LOG10A_RES[0]) + UCB_LOG10A_RES[0]
        a[nres:] = rand((nf,)) * (UCB_LOG10A_FAINT[1] - UCB_LOG10A_FAINT[0]) + UCB_LOG10A_FAINT[0]
        mag = (10.0 ** a) * g_bin[bins]
        phA = rand((tot,)) * 2 * np.pi
        phE = rand((tot,)) * 2 * np.pi
        fd[b, :, 0].index_add_(0, bins, mag * torch.cos(phA))
        fd[b, :, 1].index_add_(0, bins, mag * torch.sin(phA))
        fd[b, :, 2].index_add_(0, bins, mag * torch.cos(phE))
        fd[b, :, 3].index_add_(0, bins, mag * torch.sin(phE))
    return fd


def draw_n_ucb(B, device, p_zero=0.15):
    """Resolved UCB count per sample: 15% none, else U[1, UCB_N_MAX_COND]."""
    n = torch.randint(1, UCB_N_MAX_COND + 1, (B,), device=device)
    n[torch.rand(B, device=device) < p_zero] = 0
    return n


def sky_pose_meta(sky_true01, sigma, gen=None):
    """Build sky-pose conditioning from true sky (B,2 in [0,1]) + blur sigma (B,).
    Returns (B, 4): [sin_beta_approx in [-1,1], cos 2pi lam_a, sin 2pi lam_a,
    log-sigma normalized]. Wraps lambda."""
    B = sky_true01.shape[0]
    dev = sky_true01.device
    noise = torch.randn(B, 2, generator=gen, device=dev) if gen is not None \
        else torch.randn(B, 2, device=dev)
    sb_a = (sky_true01[:, 0] + noise[:, 0] * sigma).clamp(0, 1) * 2 - 1
    lam_a = (sky_true01[:, 1] + noise[:, 1] * sigma) % 1.0
    lsig = (torch.log10(sigma.clamp(SKY_SIGMA_MIN, SKY_SIGMA_MAX))
            - np.log10(SKY_SIGMA_MIN)) / (np.log10(SKY_SIGMA_MAX) - np.log10(SKY_SIGMA_MIN))
    return torch.stack([sb_a, torch.cos(2*np.pi*lam_a),
                        torch.sin(2*np.pi*lam_a), lsig], dim=-1)


def draw_sky_sigma(B, device, p_noinfo=0.2):
    """Per-sample sky-blur ladder: 20% 'no info' (sigma~prior) so the model can
    run seedless on round 0, else logU[min, 0.3] for the refinement regime."""
    u = torch.rand(B, device=device)
    fine = 10 ** (torch.rand(B, device=device)
                  * (np.log10(0.3) - np.log10(SKY_SIGMA_MIN)) + np.log10(SKY_SIGMA_MIN))
    return torch.where(u < p_noinfo, torch.full_like(fine, SKY_SIGMA_MAX), fine)


# Ridge-GNPE (full GNPE over the intrinsic ridge {Mc, t})
# The coherent sky peak is razor sharp but its LOCATION is set by the
# intrinsics: a 0.1% Mc / 30 s t error shifts it ~2.4 deg, flooring the sky at
# ~1700 (typical) / ~9800 (Sangria) deg^2. The oracle-reference probe
# (scripts/probe_oracle_ref.py) showed accurate {Mc,t} ALONE tightens the
# amortized sky ~9x (101% of the full-oracle gain). So we pose the intrinsic
# ridge the same way as the sky: condition on an APPROXIMATE {Mc,t} location +
# blur, in RESIDUAL-coordinate units (the residual targets are already O(1) and
# network-readable — unlike absolute Mc/t which broke earlier reference
# refinement at NSF spline resolution), and Gibbs-iterate jointly with the sky
# at inference. We pose ONLY {Mc,t} (cols 0 and 10) — NOT the spins — so the
# poorly-measured spin posteriors stay free (no GNPE overconfidence).
RIDGE_IDX = [0, 10]  # = [RESIDUAL_IDX['Mc'], RESIDUAL_IDX['t_pre_hours']] (defined below)
RIDGE_SIGMA_MIN, RIDGE_SIGMA_MAX = 3e-3, 1.0
N_RIDGE_META = 3


def ridge_pose_meta(ridge_true01, sigma, gen=None):
    """Ridge pose from the true {Mc,t} residual-target coords (B,2 in [0,1]) +
    blur sigma (B,). Returns (B,3): [Mc_res_approx, t_res_approx, log-sigma norm].
    Both coords live in residual-target [0,1] space (O(1), no precision loss)."""
    B = ridge_true01.shape[0]
    dev = ridge_true01.device
    noise = torch.randn(B, 2, generator=gen, device=dev) if gen is not None \
        else torch.randn(B, 2, device=dev)
    approx = (ridge_true01 + noise * sigma.unsqueeze(-1)).clamp(0, 1)
    lsig = (torch.log10(sigma.clamp(RIDGE_SIGMA_MIN, RIDGE_SIGMA_MAX))
            - np.log10(RIDGE_SIGMA_MIN)) / (np.log10(RIDGE_SIGMA_MAX) - np.log10(RIDGE_SIGMA_MIN))
    return torch.cat([approx, lsig.unsqueeze(-1)], dim=-1)


def draw_ridge_sigma(B, device, p_noinfo=0.2):
    """Per-sample ridge-blur ladder: 20% 'no info' (seedless round 0), else
    logU[min, 0.3] for refinement — mirrors draw_sky_sigma."""
    u = torch.rand(B, device=device)
    fine = 10 ** (torch.rand(B, device=device)
                  * (np.log10(0.3) - np.log10(RIDGE_SIGMA_MIN)) + np.log10(RIDGE_SIGMA_MIN))
    return torch.where(u < p_noinfo, torch.full_like(fine, RIDGE_SIGMA_MAX), fine)


def ref_meta_vec(ref):
    """Normalized conditioning vector from reference params: GNPE param scale,
    phase-coherence scale, and kernel type (0=independent jitter, 1=ridge —
    likelihood-refined references whose (Mc,t) compensate the (q,chi) error).
    Old datasets lacking the v6 fields default to independent (s_phase=scale)."""
    t_pre_ref = (2.6298e6 / 3600.0) - ref['t_ref_hr']  # hours before window end
    lo, hi = PARAM_RANGES['Mc']
    ks = float(ref.get('kernel_scale', 1.0))
    sp = float(ref.get('s_phase', ks))
    kt = float(ref.get('kernel_type', 0))
    return np.array([
        (np.log10(ref['Mc']) - lo) / (hi - lo),
        1.0 / ref['q'],
        (ref['c1'] + 1) / 2,
        (ref['c2'] + 1) / 2,
        np.arcsinh(t_pre_ref) / np.arcsinh(T_PRE_MAX_H),
        t_pre_ref / T_PRE_MAX_H,
        _ks_norm(ks),
        _ks_norm(sp),
        kt,
    ], dtype=np.float32)


# Residual-target mode: the 5 reference-jittered params are predicted as
# kernel-scaled residuals, e.g. (Mc_true - Mc_ref) / (Mc_ref * frac_Mc * s),
# mapped from [-RES_CLIP, RES_CLIP] to [0, 1]. Absolute Mc/t occupy ~1e-3 of
# their normalized range — beyond NSF spline resolution, causing the biased,
# overconfident Mc that broke GNPE iteration. Residuals are O(1) at every
# kernel scale. Widths use REF-based quantities only (known at inference).
RESIDUAL_IDX = {'Mc': 0, 'q': 1, 'c1': 2, 'c2': 3, 't_pre_hours': 10}
RES_CLIP = 4.0


def residual_targets(sp, ref, kbase):
    """Targets in [0,1] for the 5 jittered params, residual to the reference."""
    s = float(ref.get('kernel_scale', 1.0))
    t_pre_ref_h = (2.6298e6 / 3600.0) - ref['t_ref_hr']
    sig_t_h = max(kbase['t_floor_sec'],
                  kbase['t_frac_of_tpre'] * abs(t_pre_ref_h) * 3600.0) * s / 3600.0
    r = {
        'Mc': (sp['Mc'] / ref['Mc'] - 1.0) / (kbase['frac_Mc'] * s),
        'q': np.log(sp['q'] / ref['q']) / (kbase['log_q'] * s),
        'c1': (sp['c1'] - ref['c1']) / (kbase['chi'] * s),
        'c2': (sp['c2'] - ref['c2']) / (kbase['chi'] * s),
        't_pre_hours': (sp['t_pre_hours'] - t_pre_ref_h) / sig_t_h,
    }
    return {k: np.clip((v / RES_CLIP + 1) / 2, 1e-4, 1 - 1e-4) for k, v in r.items()}


def residual_to_physical(samps, ref, kbase, params=None):
    """Map flow samples (n, 11) back to physical Mc, q(m1/m2), c1, c2,
    t_pre_hours. `params` = names encoded as residuals (default: all 5, the
    legacy scheme). Params NOT in the set are decoded ABSOLUTELY (abs-spins
    models: only Mc,t residual; q,c1,c2 absolute so their posteriors are not
    capped at the kernel width)."""
    names = set(params) if params is not None else set(RESIDUAL_IDX)
    s = float(ref.get('kernel_scale', 1.0))
    t_pre_ref_h = (2.6298e6 / 3600.0) - ref['t_ref_hr']
    sig_t_h = max(kbase['t_floor_sec'],
                  kbase['t_frac_of_tpre'] * abs(t_pre_ref_h) * 3600.0) * s / 3600.0
    r = {k: (samps[:, i] * 2 - 1) * RES_CLIP for k, i in RESIDUAL_IDX.items()}
    res = {
        'Mc': ref['Mc'] * (1 + r['Mc'] * kbase['frac_Mc'] * s),
        'q': ref['q'] * np.exp(r['q'] * kbase['log_q'] * s),
        'c1': ref['c1'] + r['c1'] * kbase['chi'] * s,
        'c2': ref['c2'] + r['c2'] * kbase['chi'] * s,
        't_pre_hours': t_pre_ref_h + r['t_pre_hours'] * sig_t_h,
    }
    # absolute decodings (PARAM_RANGES normalization) for non-residual params
    absd = {
        'Mc': 10 ** (samps[:, 0] * (PARAM_RANGES['Mc'][1] - PARAM_RANGES['Mc'][0])
                     + PARAM_RANGES['Mc'][0]),
        'q': 1.0 / np.clip(samps[:, 1], 1e-3, 1.0),       # model m2/m1 -> physical m1/m2
        'c1': samps[:, 2] * 2 - 1,
        'c2': samps[:, 3] * 2 - 1,
        't_pre_hours': 10 ** (samps[:, 10] * (PARAM_RANGES['t_pre_hours'][1]
                              - PARAM_RANGES['t_pre_hours'][0]) + PARAM_RANGES['t_pre_hours'][0]),
    }
    return {k: (res[k] if k in names else absd[k]) for k in RESIDUAL_IDX}


# Post-hoc calibration lives in heron/recal.py (numpy only); re-exported here.
from heron.recal import load_recal, recalibrate_samples   # noqa: E402,F401


class HetDataset(Dataset):
    """Heterodyned binned FD data. Signal-only on disk; noise added on the fly
    (train) or once with a fixed seed (val/test)."""

    def __init__(self, data_dir, split='train', noise_scale=1.0,
                 residual=False, ucb_cond=False):
        root = Path(data_dir)
        self.data_dir = root / split
        self.split = split
        self.residual = residual
        self.ucb_cond = ucb_cond

        info = json.loads((root / 'dataset_info.json').read_text())
        self.kbase = info['kernel'].get('base', info['kernel'].get('narrow'))
        counts = np.load(root / 'bin_counts.npy')
        self.sigma_bins = torch.from_numpy(
            (info['sigma0'] * noise_scale / np.sqrt(counts)).astype(np.float32))

        meta_path = self.data_dir / 'metadata.json'
        self.metadata = json.loads(meta_path.read_text())
        n = len(self.metadata)
        print(f"  [{split}] {n} samples (heterodyned, on-the-fly noise)")

        first = np.load(self.data_dir / f'{self.metadata[0]["idx"]:06d}_het.npy')
        self.fd_tensor = torch.empty((n, *first.shape), dtype=torch.float32)
        print(f"  Preloading {n} arrays ({self.fd_tensor.nbytes/1e9:.1f} GB)...",
              end=' ', flush=True)
        for i, m in enumerate(self.metadata):
            arr = np.load(self.data_dir / f'{m["idx"]:06d}_het.npy')
            arr[:, 4] = np.log1p(np.maximum(arr[:, 4], 0.0))  # compress href channel
            self.fd_tensor[i] = torch.from_numpy(arr)
        print("done")

        tg = np.array([normalize_params(m['smbhb']) for m in self.metadata])
        if residual:
            # residual may be True (legacy: all 5) or a list of param names
            res_names = list(residual) if isinstance(residual, (list, tuple)) \
                else list(RESIDUAL_IDX)
            for i, m in enumerate(self.metadata):
                res = residual_targets(m['smbhb'], m['ref'], self.kbase)
                for name in res_names:
                    tg[i, RESIDUAL_IDX[name]] = res[name]
        self.targets = torch.from_numpy(tg.astype(np.float32))
        self.meta_vecs = torch.from_numpy(
            np.array([ref_meta_vec(m['ref']) for m in self.metadata]).astype(np.float32))
        self.snrs = torch.from_numpy(
            np.array([m['smbhb']['snr'] for m in self.metadata], dtype=np.float32))

        if split != 'train':
            print("  Adding fixed noise to val/test...", end=' ', flush=True)
            g = torch.Generator().manual_seed(42)
            noise = torch.empty(n, first.shape[0], 4).normal_(generator=g)
            self.fd_tensor[:, :, :4] += noise * self.sigma_bins[None, :, None]
            print("done")

    def add_noise_(self, fd):
        """On-the-fly train noise. fd: (B, n_bins, 5) on any device."""
        sig = self.sigma_bins.to(fd.device)
        fd[:, :, :4] += torch.randn_like(fd[:, :, :4]) * sig[None, :, None]
        return fd

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return {
            'fd': self.fd_tensor[idx],
            'target': self.targets[idx],
            'meta': self.meta_vecs[idx],
            'snr': self.snrs[idx],
        }


class GlobalComplexBank(nn.Module):
    """Learned matched-filter bank: K global linear projections of the complex
    heterodyned channels (rA, rE). Linear in the data -> coherent integration
    across all bins; global support -> phase ramps (timing/Doppler/sky) project
    onto single outputs, which local convolutions cannot measure precisely.
    Outputs [Re z, Im z, |z|] per filter, LayerNormed.
    """
    def __init__(self, n_bins=4096, K=96, out_dim=256):
        super().__init__()
        self.K = K
        # one real linear over (re, im) per channel pair == superset of complex-linear
        self.projA = nn.Linear(2 * n_bins, 2 * K, bias=False)
        self.projE = nn.Linear(2 * n_bins, 2 * K, bias=False)
        self.norm = nn.LayerNorm(6 * K)
        self.mlp = nn.Sequential(nn.Linear(6 * K, out_dim), nn.GELU(),
                                 nn.Linear(out_dim, out_dim), nn.GELU())

    def forward(self, x):
        """x: (B, n_bins, >=4) normalized heterodyned channels."""
        B = x.shape[0]
        zA = self.projA(x[:, :, 0:2].reshape(B, -1)).reshape(B, self.K, 2)
        zE = self.projE(x[:, :, 2:4].reshape(B, -1)).reshape(B, self.K, 2)
        z = torch.cat([zA, zE], dim=1)                       # (B, 2K, 2)
        mag = torch.sqrt((z**2).sum(-1).clamp(min=1e-12))    # (B, 2K)
        feats = torch.cat([z.reshape(B, -1), mag], dim=-1)   # (B, 6K)
        return self.mlp(self.norm(feats))


class FlowHet(nn.Module):
    """CNN encoder (FiLM-conditioned on reference params) + global matched-filter
    bank + NSF for 11D PE."""
    def __init__(self, in_channels=5, encoder_hidden=128, encoder_blocks=6,
                 context_dim=256, n_transforms=10, flow_hidden=256, flow_bins=10,
                 n_meta=N_META, bank_k=0, n_bins=4096, encoder_type='cnn'):
        super().__init__()
        from zuko.flows import NSF
        self.encoder = make_encoder(encoder_type, in_channels, encoder_hidden,
                                    encoder_blocks, context_dim, n_meta)
        self.bank = GlobalComplexBank(n_bins=n_bins, K=bank_k,
                                      out_dim=context_dim) if bank_k > 0 else None
        if self.bank is not None:
            self.fuse = nn.Sequential(
                nn.Linear(2 * context_dim, context_dim),
                nn.LayerNorm(context_dim), nn.GELU(),
            )
        self.flow = NSF(
            features=N_PARAMS, context=context_dim,
            transforms=n_transforms,
            hidden_features=[flow_hidden, flow_hidden],
            bins=flow_bins,
        )
        # Auxiliary sky head: dense gradient on the encoder for the 2 sky dims,
        # which otherwise learn slowly (11D NLL is dominated by the sharp
        # residual-encoded intrinsics). Sky NLL of 0 == prior; negative == info.
        self.sky_idx = [PARAM_NAMES.index('sin_beta'), PARAM_NAMES.index('lambda')]
        self.sky_flow = NSF(
            features=2, context=context_dim, transforms=12,
            hidden_features=[384, 384, 384], bins=12,
        )

    def _context(self, fd, meta=None):
        ctx = self.encoder(fd, meta)
        if self.bank is not None:
            ctx = self.fuse(torch.cat([ctx, self.bank(fd)], dim=-1))
        return ctx

    def log_prob(self, fd, target, meta=None):
        ctx = self._context(fd, meta)
        return self.flow(ctx).log_prob(target)

    def log_prob_and_sky(self, fd, target, meta=None):
        ctx = self._context(fd, meta)
        return (self.flow(ctx).log_prob(target),
                self.sky_flow(ctx).log_prob(target[:, self.sky_idx]))

    def sample_sky(self, fd, n_samples, meta=None):
        ctx = self._context(fd, meta)
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        B = ctx.shape[0]
        ctx_rep = ctx.repeat_interleave(n_samples, dim=0)
        return self.sky_flow(ctx_rep).sample().reshape(B, n_samples, 2)

    def sample(self, fd, n_samples, meta=None):
        ctx = self._context(fd, meta)
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        B = ctx.shape[0]
        ctx_rep = ctx.repeat_interleave(n_samples, dim=0)
        return self.flow(ctx_rep).sample().reshape(B, n_samples, -1)


def build_model(cfg):
    """Construct the posterior model from a checkpoint config. Every eval
    script uses this so a new architecture is just a config flag."""
    arch = cfg.get('arch', 'nsf')
    if arch != 'nsf':
        raise ValueError(f"unknown arch '{arch}' (the published model is 'nsf')")
    return FlowHet(n_transforms=cfg['n_transforms'],
                   flow_hidden=cfg['flow_hidden'], flow_bins=cfg['flow_bins'],
                   in_channels=cfg.get('in_channels', 5),
                   encoder_hidden=cfg['enc_hidden'], encoder_blocks=cfg['enc_blocks'],
                   context_dim=cfg['context_dim'], bank_k=cfg.get('bank_k', 0),
                   n_meta=cfg.get('n_meta', N_META),
                   encoder_type=cfg.get('encoder_type', 'cnn'))


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Chunk rotation: --train-chunks lists extra dataset roots (train splits
    # generated with different seeds). Every --chunk-epochs the training set is
    # swapped, multiplying unique-sample count without exceeding RAM.
    chunk_dirs = [args.data_dir] + (args.train_chunks or [])
    # abs-spins: q, c1, c2 are ABSOLUTE targets (their true posteriors are wider
    # than the kernel, so residual encoding caps/overtightens them — the spin
    # discrepancy vs MCMC). Mc, t stay residual (sharper than spline resolution).
    res_params = (['Mc', 't_pre_hours'] if args.abs_spins else list(RESIDUAL_IDX)) \
        if args.residual_targets else False
    train_ds = HetDataset(chunk_dirs[0], 'train', noise_scale=args.noise_scale,
                          residual=res_params, ucb_cond=args.ucb_cond)
    val_ds = HetDataset(args.data_dir, 'val', noise_scale=args.noise_scale,
                        residual=res_params, ucb_cond=args.ucb_cond)

    n_bins = train_ds[0]['fd'].shape[0]
    in_ch = train_ds[0]['fd'].shape[1]
    print(f"FD shape: ({n_bins}, {in_ch})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=min(args.batch_size * 4, 1024),
                            shuffle=False, num_workers=0)

    # Normalization stats on signal+noise
    print("Computing normalization stats...")
    n_norm = min(5000, len(train_ds))
    fd_all = train_ds.fd_tensor[:n_norm].clone()
    g = torch.Generator().manual_seed(123)
    noise = torch.empty(n_norm, n_bins, 4).normal_(generator=g)
    fd_all[:, :, :4] += noise * train_ds.sigma_bins[None, :, None]
    fd_mean = fd_all.mean(dim=(0, 1))
    fd_std = fd_all.std(dim=(0, 1)).clamp(min=1e-6)
    del fd_all, noise
    torch.save({'fd_mean': fd_mean, 'fd_std': fd_std},
               save_dir / f'norm_stats_het_{args.tag}.pt')

    base_meta = train_ds[0]['meta'].shape[0]
    n_meta = base_meta + (N_UCB_META if args.ucb_cond else 0) \
        + (N_SKY_META if args.sky_gnpe else 0) \
        + (N_RIDGE_META if args.full_gnpe else 0)
    sky_idx = [PARAM_NAMES.index('sin_beta'), PARAM_NAMES.index('lambda')]
    if args.full_gnpe and not args.sky_gnpe:
        raise SystemExit("--full-gnpe requires --sky-gnpe (joint pose over sky + Mc,t)")
    model_cfg = {
        'arch': args.arch, 'encoder_type': args.encoder, 'in_channels': in_ch,
        'enc_hidden': args.enc_hidden, 'enc_blocks': args.enc_blocks,
        'context_dim': args.context_dim, 'n_transforms': args.n_transforms,
        'flow_hidden': args.flow_hidden, 'flow_bins': args.flow_bins,
        'bank_k': args.bank_k, 'n_meta': n_meta,
        'residual_targets': args.residual_targets,
        'residual_params': (res_params if res_params else None),
        'sky_gnpe': args.sky_gnpe, 'ucb_cond': args.ucb_cond,
        'full_gnpe': args.full_gnpe,
    }
    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FlowHet(NSF): {n_params:,} "
          f"params | {N_PARAMS}D PE | {n_bins} bins x {in_ch} ch | n_meta={n_meta}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        own = model.state_dict()
        sd, padded = {}, 0
        for k, v in ckpt['model_state_dict'].items():
            if k not in own:
                continue
            if own[k].shape == v.shape:
                sd[k] = v
            elif (v.dim() == 2 and own[k].dim() == 2 and own[k].shape[0] == v.shape[0]
                  and own[k].shape[1] > v.shape[1]):
                # meta_embed grew n_meta (e.g. +3 ridge dims): keep old columns,
                # new input dims start at the fresh init (-> ~0 contribution) so
                # the warm-started sky-GNPE behaviour is preserved exactly.
                w = own[k].clone()
                w[:, :v.shape[1]] = v
                w[:, v.shape[1]:] = 0.0   # ridge dims start at zero contribution
                sd[k] = w
                padded += 1
        skipped = len(ckpt['model_state_dict']) - len(sd)
        missing, _ = model.load_state_dict(sd, strict=False)
        print(f"Resumed from {args.resume} (epoch {ckpt.get('epoch', '?')}); "
              f"loaded {len(sd)} tensors ({padded} col-padded), skipped {skipped}, "
              f"{len(missing)} fresh")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    warmup = 10
    def lr_sched(epoch):
        if epoch < warmup:
            return epoch / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / max(1, args.epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    fd_mean_d = fd_mean.to(device)
    fd_std_d = fd_std.to(device)
    sigma_d = train_ds.sigma_bins.to(device)

    if args.ucb_cond:
        resp = np.load(args.ucb_response)
        g_bin_d = torch.from_numpy(resp['g_bin'].astype(np.float32)).to(device)
        band_bins = torch.from_numpy(
            np.where(resp['g_bin'] > 0)[0].astype(np.int64)).to(device)
        print(f"UCB on-the-fly foreground: {len(band_bins)} band bins, "
              f"resolved n_ucb 0-{UCB_N_MAX_COND} + faint 0-{UCB_N_FAINT_MAX}")

    best_val = float('inf')
    best_sky = float('inf')
    patience_count = 0
    for epoch in range(1, args.epochs + 1):
        if args.stop_epoch and epoch > args.stop_epoch:
            print(f"Stopping after epoch {args.stop_epoch} (--stop-epoch)")
            break
        if len(chunk_dirs) > 1 and epoch > 1 and (epoch - 1) % args.chunk_epochs == 0:
            nxt = chunk_dirs[((epoch - 1) // args.chunk_epochs) % len(chunk_dirs)]
            print(f"  [chunk] switching train set -> {nxt}", flush=True)
            del train_ds, train_loader
            train_ds = HetDataset(nxt, 'train', noise_scale=args.noise_scale,
                                  residual=res_params, ucb_cond=args.ucb_cond)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=True, num_workers=0, pin_memory=True)
        model.train()
        train_loss_sum, sky_loss_sum, n_train = 0, 0, 0
        for batch in train_loader:
            fd = batch['fd'].to(device, non_blocking=True)
            tgt = batch['target'].to(device)
            meta = batch['meta'].to(device)
            if args.ucb_cond:
                n_ucb = draw_n_ucb(len(fd), device)
                inject_ucb_foreground(fd, n_ucb, g_bin_d, band_bins)
                meta = torch.cat([meta, (n_ucb.float() / UCB_N_MAX_COND)[:, None]], dim=-1)
            fd[:, :, :4] += torch.randn_like(fd[:, :, :4]) * sigma_d[None, :, None]
            fd = (fd - fd_mean_d) / fd_std_d
            if args.sky_gnpe:
                sig = draw_sky_sigma(len(fd), device)
                meta = torch.cat([meta, sky_pose_meta(tgt[:, sky_idx], sig)], dim=-1)
            if args.full_gnpe:
                rsig = draw_ridge_sigma(len(fd), device)
                meta = torch.cat([meta, ridge_pose_meta(tgt[:, RIDGE_IDX], rsig)], dim=-1)
            if args.sky_weight > 0:
                lp, sky_lp = model.log_prob_and_sky(fd, tgt, meta=meta)
                sky_loss = -sky_lp.clamp(min=-100, max=100).mean()
                main_loss = -lp.clamp(min=-100, max=100).mean()
            else:
                main_loss = -model.log_prob(fd, tgt, meta=meta).clamp(min=-100, max=100).mean()
                sky_loss = torch.zeros((), device=device)
            loss = main_loss + args.sky_weight * sky_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(fd)
            sky_loss_sum += sky_loss.item() * len(fd)
            n_train += len(fd)
        scheduler.step()
        train_loss = train_loss_sum / n_train
        train_sky = sky_loss_sum / n_train

        model.eval()
        val_nll_all, val_sky_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                fd = batch['fd'].to(device).clone()
                tgt = batch['target'].to(device)
                meta = batch['meta'].to(device)
                if args.ucb_cond:
                    g = torch.Generator(device=device).manual_seed(7)
                    n_ucb = torch.randint(0, UCB_N_MAX_COND + 1, (len(fd),),
                                          generator=g, device=device)
                    inject_ucb_foreground(fd, n_ucb, g_bin_d, band_bins, rng=g)
                    meta = torch.cat([meta, (n_ucb.float() / UCB_N_MAX_COND)[:, None]], dim=-1)
                fd = (fd - fd_mean_d) / fd_std_d
                if args.sky_gnpe:
                    g = torch.Generator(device=device).manual_seed(99)
                    sig = torch.full((len(fd),), 0.05, device=device)  # fixed mid-blur val
                    meta = torch.cat([meta, sky_pose_meta(tgt[:, sky_idx], sig, gen=g)], dim=-1)
                if args.full_gnpe:
                    g = torch.Generator(device=device).manual_seed(101)
                    rsig = torch.full((len(fd),), 0.05, device=device)
                    meta = torch.cat([meta, ridge_pose_meta(tgt[:, RIDGE_IDX], rsig, gen=g)], dim=-1)
                lp, sky_lp = model.log_prob_and_sky(fd, tgt, meta=meta)
                lp = lp.clamp(min=-100, max=100)
                valid = torch.isfinite(lp)
                if valid.any():
                    val_nll_all.append(-lp[valid].cpu())
                    val_sky_all.append(-sky_lp.clamp(min=-100, max=100)[valid].cpu())
        if val_nll_all:
            val_nll_cat = torch.cat(val_nll_all)
            sorted_nll, _ = torch.sort(val_nll_cat)
            trim = max(1, len(val_nll_cat) // 10)
            val_loss = sorted_nll[trim:-trim].mean().item()
            val_sky = torch.cat(val_sky_all).mean().item()
        else:
            val_loss, val_sky = float('inf'), float('inf')

        ckpt_payload = None
        if val_sky < best_sky:
            best_sky = val_sky
            ckpt_payload = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': val_loss, 'val_sky': val_sky, 'n_bins': n_bins,
                'tag': args.tag, 'config': dict(model_cfg, n_bins=n_bins),
            }
            torch.save(ckpt_payload, save_dir / f'sbi_het_{args.tag}_bestsky.pt')
        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': best_val, 'n_bins': n_bins, 'tag': args.tag,
                'config': dict(model_cfg, n_bins=n_bins),
            }, save_dir / f'sbi_het_{args.tag}_best.pt')
            star = ' *'
        else:
            patience_count += 1
            star = ''

        if epoch % 5 == 0 or epoch == 1:
            print(f"E{epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"best={best_val:.4f}  skyNLL(tr/val)={train_sky:.3f}/{val_sky:.3f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}{star}",
                  flush=True)
        if patience_count >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nDone [{args.tag}]. Best val NLL: {best_val:.4f}")


def sky_area_90(lam, sinb, k=30, level=0.90):
    """Grid-free 90% HPD sky area [deg^2] from posterior samples via k-NN density.

    Raw HEALPix histograms of n samples saturate at ~level*n*pixarea for broad
    posteriors (one sample per pixel) — pure artifact. Here: p_i ~ k/(n*A_i)
    with A_i the spherical-cap area to the k-th neighbour, and
    area(HPD) = sum_{top level by p_i} A_i / k. Handles multimodality.
    """
    b = np.arcsin(np.clip(sinb, -1, 1))
    v = np.stack([np.cos(b) * np.cos(lam), np.cos(b) * np.sin(lam),
                  np.sin(b)], axis=-1).astype(np.float32)
    n = len(v)
    G = np.clip(v @ v.T, -1, 1)
    # k-th largest cosine per row (excluding self) -> k-NN angular radius
    cos_k = np.partition(G, n - 1 - k, axis=1)[:, n - 1 - k]
    A = 2 * np.pi * (1 - cos_k)            # cap area, steradian
    dens = 1.0 / np.maximum(A, 1e-12)      # ~ p_i up to constants
    order = np.argsort(dens)[::-1]
    top = order[:int(np.ceil(level * n))]
    return A[top].sum() / k * (180 / np.pi)**2


def evaluate(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import healpy as hp

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tag = ckpt.get('tag', args.tag)
    norm = torch.load(save_dir / f'norm_stats_het_{tag}.pt',
                      map_location=device, weights_only=False)
    fd_mean = norm['fd_mean'].to(device)
    fd_std = norm['fd_std'].to(device)

    res_cfg = cfg.get('residual_params') or cfg.get('residual_targets', False)
    dataset = HetDataset(args.data_dir, args.eval_split, noise_scale=args.noise_scale,
                         residual=res_cfg, ucb_cond=cfg.get('ucb_cond', False))
    N = len(dataset)

    all_nll, all_snr = [], []
    all_pcts = {name: [] for name in PARAM_NAMES}
    areas_90 = []

    print(f"Evaluating {N} samples on {args.eval_split}...")
    for i in tqdm(range(N)):
        entry = dataset[i]
        fd = (entry['fd'].unsqueeze(0).to(device) - fd_mean) / fd_std
        meta = entry['meta'].unsqueeze(0).to(device)
        with torch.no_grad():
            lp = model.log_prob(fd, entry['target'].unsqueeze(0).to(device), meta=meta)
            if torch.isfinite(lp):
                all_nll.append(-lp.item())
            samps = model.sample(fd, 5000, meta=meta).cpu().numpy().squeeze()
        samps = np.clip(samps, 1e-4, 1 - 1e-4)

        target = entry['target'].numpy()
        for j, name in enumerate(PARAM_NAMES):
            all_pcts[name].append((samps[:, j] <= target[j]).mean())
        all_snr.append(float(entry['snr']))

        # sky area: k-NN HPD estimator on a 4000-sample subset
        lam_s = samps[:4000, 7] * 2 * np.pi
        sinb_s = np.clip(samps[:4000, 6] * 2 - 1, -1 + 1e-6, 1 - 1e-6)
        areas_90.append(sky_area_90(lam_s, sinb_s))

    snr_arr = np.array(all_snr)
    areas_90 = np.array(areas_90)

    print(f"\n=== Heterodyned 11D PE ({args.eval_split}, N={N}) ===")
    print(f"  Mean NLL: {np.mean(all_nll):.4f}")
    from scipy.stats import kstest
    ks = {name: kstest(np.array(all_pcts[name]), 'uniform') for name in PARAM_NAMES}
    print("  KS calibration (p-value, D): " + "  ".join(
        f"{n}=({r.pvalue:.3f},{r.statistic:.3f})" for n, r in ks.items()))
    for level in [0.5, 0.68, 0.90]:
        covs = []
        for name in PARAM_NAMES:
            pcts = np.array(all_pcts[name])
            covs.append(((pcts > (1-level)/2) & (pcts < 1-(1-level)/2)).mean())
        print(f"  CI {level:.0%}: " + "  ".join(
            f"{n}={c:.0%}" for n, c in zip(PARAM_NAMES, covs)))
    print(f"  Median 90% sky area: {np.median(areas_90):.1f} deg^2")
    for lo, hi in [(10, 50), (50, 200), (200, 1000), (1000, 1e9)]:
        m = (snr_arr >= lo) & (snr_arr < hi)
        if m.sum() > 3:
            print(f"    SNR [{lo:.0f},{hi:.0f}): median {np.median(areas_90[m]):.1f} "
                  f"deg^2 (n={m.sum()})")

    # PP plot
    fig, ax = plt.subplots(figsize=(8, 8))
    xp_ = np.linspace(0, 1, 200)
    s_ci = np.sqrt(xp_ * (1 - xp_) / N)
    ax.fill_between(xp_, np.clip(xp_-3*s_ci, 0, 1), np.clip(xp_+3*s_ci, 0, 1),
                    color='gray', alpha=0.15)
    y = np.arange(1, N+1) / N
    colors = plt.cm.tab10(np.linspace(0, 1, N_PARAMS))
    for j, name in enumerate(PARAM_NAMES):
        ax.plot(np.sort(np.array(all_pcts[name])), y,
                label=f"{name} (p={ks[name].pvalue:.2f})",
                color=colors[j], lw=1.2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Theoretical CDF',
           ylabel='Empirical CDF',
           title=f'P-P Plot — heterodyned 11D PE ({args.eval_split}, N={N})')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / f'pp_plot_het_{args.eval_split}.png', dpi=300)
    plt.close(fig)

    # sky area vs SNR
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(snr_arr, areas_90, alpha=0.4, s=15)
    ax.set(xlabel='SNR', ylabel='90% CR [deg^2]', xscale='log', yscale='log',
           title='Sky Area vs SNR (heterodyned 11D PE)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / f'sky_vs_snr_het_{args.eval_split}.png', dpi=200)
    plt.close(fig)
    print(f"Plots saved to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=MBHB_DATA,
                        help='dataset root with train/val splits (pe_smbhb/data_mbhb.py)')
    parser.add_argument('--save-dir', type=str, default='runs/train/mbhb')
    parser.add_argument('--tag', type=str, default='mbhb')
    parser.add_argument('--seed', type=int, default=0,
                        help='seeds torch / numpy (initialization, shuffling, noise, foreground)')
    parser.add_argument('--ucb-response', type=str, default=UCB_RESPONSE,
                        help='calibrated UCB line response (evaluation/calibrate_ucb.py)')
    parser.add_argument('--stop-epoch', type=int, default=0,
                        help='stop after this epoch while keeping the --epochs schedule '
                             '(reproduces an interrupted run; 0 = run to --epochs)')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--noise-scale', type=float, default=1.0)
    parser.add_argument('--sky-weight', type=float, default=0.5,
                        help='Weight of auxiliary sky-flow NLL (0 disables)')
    parser.add_argument('--ucb-cond', action='store_true', default=False,
                        help='Condition on n_ucb + confusion amplitude (robustness)')
    parser.add_argument('--sky-gnpe', action='store_true', default=False,
                        help='Condition on approximate sky pose (GNPE); learn sharp residual')
    parser.add_argument('--full-gnpe', action='store_true', default=False,
                        help='Also pose the intrinsic ridge {Mc,t} (residual-coord GNPE); '
                             'joint Gibbs over sky+Mc+t at inference. Requires --sky-gnpe.')
    parser.add_argument('--residual-targets', action='store_true', default=True,
                        help='Predict jittered params as kernel-scaled residuals')
    parser.add_argument('--no-residual-targets', dest='residual_targets',
                        action='store_false')
    parser.add_argument('--abs-spins', action='store_true', default=False,
                        help='q,c1,c2 as ABSOLUTE targets (not residual); fixes '
                             'kernel-capped spin posteriors. Mc,t stay residual.')
    parser.add_argument('--bank-k', type=int, default=96,
                        help='Global complex matched-filter bank size (0 disables)')
    parser.add_argument('--train-chunks', type=str, nargs='*', default=None,
                        help='Extra dataset roots; train split rotates over them')
    parser.add_argument('--chunk-epochs', type=int, default=8,
                        help='Epochs between train-chunk swaps')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--enc-hidden', type=int, default=128)
    parser.add_argument('--enc-blocks', type=int, default=6)
    parser.add_argument('--context-dim', type=int, default=256)
    parser.add_argument('--arch', type=str, default='nsf', choices=['nsf'],
                        help='Posterior estimator (the published model is the coupling NSF)')
    parser.add_argument('--encoder', type=str, default='cnn', choices=['cnn', 'transformer'],
                        help='Embedding network: cnn (Encoder11D) or transformer')
    parser.add_argument('--n-transforms', type=int, default=10)
    parser.add_argument('--flow-hidden', type=int, default=256)
    parser.add_argument('--flow-bins', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval-split', type=str, default='val')
    parser.add_argument('--model-path', type=str, default=MBHB_FLOW)
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
