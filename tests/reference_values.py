"""
Waveform fingerprints: a few scalar summaries of the MBHB (BBHx) and UCB
(GBGPU) waveforms at fixed parameters, as produced by the environment that
generated the published results. A different BBHx/GBGPU build changes them.
"""
import numpy as np


def waveform_fingerprint():
    import cupy as cp
    from data_utils import ucb
    from heron.sources import SANGRIA
    from pe_smbhb.data_mbhb import HetDataGenerator
    gen = HetDataGenerator(output_dir=None)
    S = SANGRIA[1]['src']
    m1, m2 = S['m1'], S['m2']; eta = m1 * m2 / (m1 + m2) ** 2
    Mc = (m1 + m2) * eta ** 0.6; q = m1 / m2
    t_ref = (gen.t_obs - 24.0 * 3600) / 3600
    p = np.array([[Mc, q, S['chi1'], S['chi2'], np.log10(S['dist_Mpc']), t_ref,
                   np.cos(S['iota']), np.sin(S['beta']), S['lam'], S['psi'], S['phi0']]])
    A, E = gen.gen_wf(p)
    hA, hE = A[0], E[0]
    snr = float(cp.sqrt(4 * gen.df * cp.sum((cp.abs(hA) ** 2 + cp.abs(hE) ** 2) / gen.psd_g)))
    k = 5000                                           # one mid-band bin
    f0 = 2.5e-3
    s0 = int(round(f0 / gen.df - 64))
    m = ucb(f_lims=np.array([s0 * gen.df, (s0 + 128) * gen.df]), f_inds=[s0, s0 + 128],
            gpu=True, tobs=gen.t_obs, dt=gen.dt)
    a, e, _ = m.eval(np.array([[-21.0, f0 * 1e3, -16.5, 1.0, 0.3, 0.7, 2.0, -0.4]]))
    a = cp.asarray(a).ravel(); e = cp.asarray(e).ravel()
    return dict(
        mbhb_snr=snr,
        mbhb_A_bin=np.array([float(hA[k].real), float(hA[k].imag)]),
        mbhb_E_bin=np.array([float(hE[k].real), float(hE[k].imag)]),
        ucb_norm=np.array([float(cp.linalg.norm(a)), float(cp.linalg.norm(e))]),
        ucb_peak=np.array([float(cp.abs(a).max()), int(cp.argmax(cp.abs(a)))]),
        sigma0=float(gen.sigma0), n_bins=int(gen.n_bins))


REFERENCE = {
    'mbhb_snr': 357.0375069106725,
    'mbhb_A_bin': np.array([3.924709305233995e-18, -9.51533812576419e-19]),
    'mbhb_E_bin': np.array([-1.3892297060509323e-18, 3.440313956395406e-19]),
    'ucb_norm': np.array([4.449260401815587e-17, 1.5270246021565032e-17]),
    'ucb_peak': np.array([4.390391710603096e-17, 65.0]),
    'sigma0': 810.8329051043747,
    'n_bins': 39421,
}
