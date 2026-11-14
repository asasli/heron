"""
Building blocks of the alert flow (stage 2): the eleven parameters and their
model-space ranges, and the embedding networks.

    PARAM_NAMES, PARAM_RANGES     parameter order and model-space ranges
    to_model_space, from_model_space, normalize_params, denormalize_params
    FiLMLayer, ResBlock1D         conditioned residual 1-D convolution block
    Encoder11D                    residual CNN encoder with FiLM conditioning (released model)
    TransformerEncoder11D         alternative encoder (not used by the released model)

The model itself (`FlowHet`), the dataset class and the training loop are in
pe_smbhb/train_flow.py.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



# 11D PARAMETER DEFINITIONS


PARAM_NAMES = ['Mc', 'q', 'c1', 'c2', 'logD', 'cos_inc',
               'sin_beta', 'lambda', 'psi', 'phi0', 't_pre_hours']
# Mc and t_pre_hours span large, heavily skewed dynamic ranges, so the model
# works in log10 space (spreads the mass, fixes the over-wide posteriors).
# q is redefined to m2/m1 in (0,1]: the metadata stores m1/m2 (m1>=m2), which
# ranges to ~43; inverting bounds it cleanly. PARAM_RANGES are in MODEL space.
PARAM_RANGES = {
    'Mc':          (5.7, 7.45),        # log10(Mc); Mc in ~[6e5, 2.6e7]
    'q':           (0.0, 1.0),         # m2/m1 (<=1); metadata m1/m2 is inverted on load
    'c1':          (-1.0, 1.0),        # Paper: chi1 in [-1, 1]
    'c2':          (-1.0, 1.0),        # Paper: chi2 in [-1, 1]
    'logD':        (3.69, 5.5),        # data range [3.699, 5.301]
    'cos_inc':     (-1.0, 1.0),
    'sin_beta':    (-1.0, 1.0),
    'lambda':      (0.0, 2*np.pi),
    'psi':         (0.0, np.pi),
    'phi0':        (0.0, 2*np.pi),
    't_pre_hours': (-2.05, 2.78),      # log10(hours); hours in [0.01, 600]
}
N_PARAMS = len(PARAM_NAMES)


def to_model_space(name, phys):
    """Map a physical value (as stored in metadata) into model space."""
    if name in ('Mc', 't_pre_hours'):
        return np.log10(phys)
    if name == 'q':
        return 1.0 / phys          # metadata stores m1/m2; model uses m2/m1 in (0,1]
    return phys


def from_model_space(name, m):
    """Map a model-space value back to a physical/display value (vectorized-safe)."""
    if name in ('Mc', 't_pre_hours'):
        return 10.0 ** m
    return m                        # q model-space IS the (redefined) physical m2/m1


def normalize_params(params_dict):
    out = np.zeros(N_PARAMS, dtype=np.float32)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        val = to_model_space(name, params_dict[name])
        out[i] = np.clip((val - lo) / (hi - lo), 1e-4, 1 - 1e-4)
    return out


def denormalize_params(arr):
    out = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        out[name] = from_model_space(name, float(arr[i]) * (hi - lo) + lo)
    return out



# DATASET



class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: scale and shift conv features using metadata."""
    def __init__(self, n_meta, n_channels):
        super().__init__()
        self.proj = nn.Linear(n_meta, 2 * n_channels)

    def forward(self, x, meta):
        """
        x: (batch, channels, n_bins)
        meta: (batch, n_meta)
        """
        gamma_beta = self.proj(meta)  # (batch, 2*channels)
        gamma = gamma_beta[:, :x.shape[1]].unsqueeze(-1) + 1  # (batch, channels, 1)
        beta = gamma_beta[:, x.shape[1]:].unsqueeze(-1)
        return gamma * x + beta


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, n_meta=0):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.film = FiLMLayer(n_meta, channels) if n_meta > 0 else None

    def forward(self, x, meta=None):
        residual = x
        x = nn.functional.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.film is not None and meta is not None:
            x = self.film(x, meta)
        return nn.functional.gelu(x + residual)


class Encoder11D(nn.Module):
    """1D CNN encoder with strided downsampling, FiLM conditioning and attention pooling.

    The frequency axis is ~5600 bins but the sky/intrinsic information is smooth
    across it, so we downsample heavily: a strided stem (stem_stride) plus one
    mid-network stride-2 conv. This cuts the conv FLOPs of the ResBlock stack by
    ~5-6x vs running everything at full resolution, with negligible accuracy cost.
    """
    def __init__(self, in_channels=4, hidden=128, n_blocks=6,
                 kernel_size=7, context_dim=128, n_meta=2, stem_stride=4):
        super().__init__()
        self.n_meta = n_meta
        # Multi-scale (Inception-style) strided stem: parallel branches with
        # different odd kernels capture both the slow sky-modulation (large
        # kernels) and the fast chirp structure (small kernels) in one pass.
        stem_kernels = [7, 15, 31, 63]
        assert hidden % len(stem_kernels) == 0, "hidden must be divisible by 4"
        bch = hidden // len(stem_kernels)
        self.stem = nn.ModuleList([
            nn.Conv1d(in_channels, bch, k, stride=stem_stride, padding=k // 2)
            for k in stem_kernels
        ])
        self.input_bn = nn.BatchNorm1d(hidden)

        # Embed metadata (orbital_phase, ...) to higher dim for FiLM
        self.meta_embed = nn.Sequential(
            nn.Linear(n_meta, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        ) if n_meta > 0 else None
        film_dim = 32 if n_meta > 0 else 0

        self.blocks = nn.ModuleList([
            ResBlock1D(hidden, kernel_size=kernel_size, dilation=2**i,
                       n_meta=film_dim)
            for i in range(n_blocks)
        ])

        # One extra stride-2 downsample halfway through the block stack.
        self.downsample_at = n_blocks // 2
        self.mid_pool = nn.Conv1d(hidden, hidden, kernel_size=kernel_size,
                                  stride=2, padding=kernel_size // 2)
        self.mid_bn = nn.BatchNorm1d(hidden)

        self.attn_query = nn.Linear(hidden, 1)
        # Context includes both pooled features and metadata embedding
        self.output_proj = nn.Sequential(
            nn.Linear(hidden + film_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
        )

    def forward(self, x, meta=None):
        """
        x: (batch, n_bins, in_channels)
        meta: (batch, n_meta) — orbital_phase, etc.
        """
        x = x.permute(0, 2, 1)  # (batch, channels, n_bins)
        x = torch.cat([branch(x) for branch in self.stem], dim=1)  # multi-scale
        x = nn.functional.gelu(self.input_bn(x))

        meta_emb = None
        if self.meta_embed is not None and meta is not None:
            meta_emb = self.meta_embed(meta)  # (batch, 32)

        for i, block in enumerate(self.blocks):
            x = block(x, meta_emb)
            if i == self.downsample_at - 1:
                x = nn.functional.gelu(self.mid_bn(self.mid_pool(x)))

        x = x.permute(0, 2, 1)  # (batch, n_bins_reduced, hidden)
        attn_weights = torch.softmax(self.attn_query(x), dim=1)
        pooled = (x * attn_weights).sum(dim=1)  # (batch, hidden)

        if meta_emb is not None:
            pooled = torch.cat([pooled, meta_emb], dim=-1)

        return self.output_proj(pooled)


class TransformerEncoder11D(nn.Module):
    """Attention encoder over the frequency axis — same interface as Encoder11D:
    forward(x:[B,n_bins,in_channels], meta:[B,n_meta]) -> [B,context_dim].

    Patchify the bins (Conv1d patch embed) into tokens, prepend a meta token
    (the conditioning vector) and a CLS token, run a pre-norm transformer, pool
    the CLS token. Self-attention captures the long-range frequency correlations
    (the global phase ramps) that the strided CNN compresses away — the
    diagnosed sky-info bottleneck.
    """
    def __init__(self, in_channels=5, hidden=128, n_blocks=6, context_dim=256,
                 n_meta=2, d_model=384, depth=6, heads=6, patch=32):
        super().__init__()
        self.n_meta = n_meta
        self.patch_embed = nn.Conv1d(in_channels, d_model, kernel_size=patch, stride=patch)
        self.meta_embed = nn.Sequential(
            nn.Linear(n_meta, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        ) if n_meta > 0 else None
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        n_patch = 4096 // patch
        self.pos = nn.Parameter(torch.zeros(1, n_patch + 2, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            activation='gelu', batch_first=True, norm_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        meta_extra = d_model if n_meta > 0 else 0
        self.output_proj = nn.Sequential(
            nn.Linear(d_model + meta_extra, context_dim),
            nn.LayerNorm(context_dim), nn.GELU())

    def forward(self, x, meta=None):
        B = x.shape[0]
        tok = self.patch_embed(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, n_patch, d)
        cls = self.cls.expand(B, -1, -1)
        m_emb = (self.meta_embed(meta) if (self.meta_embed is not None and meta is not None)
                 else torch.zeros(B, self.cls.shape[-1], device=x.device))
        seq = torch.cat([cls, m_emb[:, None, :], tok], dim=1) + self.pos[:, :tok.shape[1] + 2]
        h = self.norm(self.blocks(seq)[:, 0])               # CLS token
        if self.meta_embed is not None and meta is not None:
            h = torch.cat([h, m_emb], dim=-1)
        return self.output_proj(h)
