"""
Inference with the alert flow (stage 2): the three operations every script
shares.

    draw_ref       heterodyne reference at kernel scale s around a source
    het_features   heterodyne + relative-binning of whitened A/E data against a reference
    gnpe_marg      sky-GNPE: Gibbs-iterate the sky pose, with K_POSE pose draws
                   in the final round (pose-marginalized posterior)

The reference is the entry point a search stage provides. In this release it is
drawn around the injected source with the training kernel (Supplementary Note
"Stage-2 amortized inference and the amortization floor"); `wide_spin=True`
draws the reference spins from the prior, as in the training of the released
model.
"""
import numpy as np
import torch
import cupy as cp

from pe_smbhb.data_mbhb import FIDUCIAL_EXT
from pe_smbhb.train_flow import sky_pose_meta, SKY_SIGMA_MIN, SKY_SIGMA_MAX

K_POSE = 8          # pose draws in the final GNPE round (published configuration)


def circ_median(l01):
    a = l01 * 2 * np.pi
    return (np.angle(np.mean(np.exp(1j * a))) % (2 * np.pi)) / (2 * np.pi)


def draw_ref(truth_row, t_pre_h, kbase, rng, s=1.0, wide_spin=False):
    """Stage-1-like reference draw around the truth (independent kernel), at
    kernel scale s. s<1 tightens the reference-marginalization dispersion so the
    conditioned params (Mc,q,chi,t) are not double-broadened -> nominal coverage.
    wide_spin=True: reference spins drawn UNINFORMATIVELY from U[-0.99,0.99]
    (matches the WIDE_SPIN_REF-trained v15 model, whose data used prior-drawn
    reference spins so the network extracts chi from the data, not the ref)."""
    Mc, q, c1, c2 = truth_row[0], truth_row[1], truth_row[2], truth_row[3]
    k = kbase
    t_ref_hr = truth_row[5]
    if wide_spin:
        rc1 = float(rng.uniform(-0.99, 0.99)); rc2 = float(rng.uniform(-0.99, 0.99))
    else:
        rc1 = float(np.clip(c1 + rng.randn() * k['chi'] * s, -0.99, 0.99))
        rc2 = float(np.clip(c2 + rng.randn() * k['chi'] * s, -0.99, 0.99))
    return dict(
        Mc=Mc * (1 + rng.randn() * k['frac_Mc'] * s),
        q=max(q * np.exp(rng.randn() * k['log_q'] * s), 1.0001),
        c1=rc1, c2=rc2,
        t_ref_hr=t_ref_hr + rng.randn() * max(k['t_floor_sec'],
                 k['t_frac_of_tpre'] * t_pre_h * 3600) * s / 3600,
        kernel_scale=s, s_phase=s, kernel_type=0)


def het_features(gen, dA, dE, ref):
    pr = np.array([[ref['Mc'], ref['q'], ref['c1'], ref['c2'], FIDUCIAL_EXT['logD'],
                    ref['t_ref_hr'], FIDUCIAL_EXT['ci'], FIDUCIAL_EXT['sb'],
                    FIDUCIAL_EXT['lam'], FIDUCIAL_EXT['psi'], FIDUCIAL_EXT['phi0']]])
    Ar, _ = gen.gen_wf(pr); Ar = Ar[0]; mag = cp.abs(Ar)
    c = cp.where(mag > 0, cp.conj(Ar) / cp.where(mag > 0, mag, 1.0), 0.0)
    rAb = gen.bin_mean((dA * c)[None, :])[0]; rEb = gen.bin_mean((dE * c)[None, :])[0]
    href = gen.bin_mean((mag / gen.sqrt_psd_g).astype(cp.complex128)[None, :])[0].real
    return cp.stack([rAb.real, rAb.imag, rEb.real, rEb.imag,
                     cp.log1p(cp.maximum(href, 0))], -1)


def gnpe_marg(m, fd, base, device, rounds, n_per, inflate, seed, k_pose=None):
    """Sky-GNPE with pose-MARGINALIZED final round: `k_pose` (default K_POSE)
    pose draws, n_per samples each. Returns the stacked unit-cube samples."""
    k_pose = K_POSE if k_pose is None else k_pose
    sb, lam, s = 0.5, 0.5, SKY_SIGMA_MAX
    g = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for r in range(rounds - 1):
            pose = sky_pose_meta(torch.tensor([[sb, lam]], device=device, dtype=torch.float32),
                                 torch.tensor([s], device=device, dtype=torch.float32), gen=g)
            samp = m.sample(fd, n_per, meta=torch.cat([base, pose], dim=-1)).cpu().numpy().squeeze()
            samp = np.clip(samp, 1e-4, 1 - 1e-4)
            sb = float(np.median(samp[:, 6])); lam = circ_median(samp[:, 7])
            s = float(np.clip(max(np.std(samp[:, 6]), np.std(samp[:, 7])) * inflate,
                              SKY_SIGMA_MIN, SKY_SIGMA_MAX))
        # final round: ensemble over K_POSE pose draws
        outs = []
        for _ in range(k_pose):
            pose = sky_pose_meta(torch.tensor([[sb, lam]], device=device, dtype=torch.float32),
                                 torch.tensor([s], device=device, dtype=torch.float32), gen=g)
            o = m.sample(fd, n_per, meta=torch.cat([base, pose], dim=-1)).cpu().numpy().squeeze()
            outs.append(np.clip(o, 1e-4, 1 - 1e-4))
    return np.concatenate(outs, axis=0)
