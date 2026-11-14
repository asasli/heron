"""
The four HERON stages as importable functions.

    simulate_event      one month of LISA A/E data: MBHB + resolved UCBs +
                        galactic confusion + instrument noise (three data sets:
                        clean / foreground / to-be-subtracted)
    detect_oracle       stage 1, injected-window mode (see below)
    detect_yolo         stage 1, the trained spectrogram detector on the same data
    alert_posterior     stage 2, amortized MBHB posterior (<1 s)
    subtract_catalogue  stage 3, UCB flow + F-statistic subtraction
    handoff_files       the npz hand-off consumed by stage 4 (pointing/synergy_pe.py)

Stage 1 has two modes. `detect_oracle` places a window on every injected UCB
with the detector's measured centre error (uniform within +-DET_ERR_BINS bins);
this is the configuration of every end-to-end number quoted in the paper, where
the detector itself is validated separately on spectrograms (yolo/, Extended
Data Figs. 1-3). `detect_yolo` runs the released detector on the simulated data
themselves and passes its detections on, which closes the loop on one data set.

Numerical reproducibility: the order of random draws in `simulate_event` is
fixed, and with the default seed the hand-off files reproduce the published
ones to floating-point rounding (tests/test_handoff_regression.py).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cupy as cp                                                    # noqa: E402
from config import *                                                 # noqa: E402,F401,F403
from data_utils import ucb                                           # noqa: E402
from heron.artifacts import (MBHB_FLOW, MBHB_NORM, MBHB_RECAL, UCB_FLOW, YOLO,   # noqa: E402
                             load_kernel)
from heron.sources import SANGRIA                                    # noqa: E402
from pe_smbhb.data_mbhb_ucb import UCBHetGenerator                   # noqa: E402
from pe_ucb.data_ucb import (W, FDOT_LOG10, UCB_F_LO_MHZ, UCB_F_HI_MHZ,          # noqa: E402
                             UCB_LOG10A, DET_ERR_BINS)
from pe_ucb.train_flow import UCBFlow, RANGES                        # noqa: E402

F_MIN, F_MAX = 1e-4, 4e-3          # stage-4 analysis band [Hz]
MTSUN = 4.925490947e-6

ALERT_PARAM_NAMES = ['Mc_Msun', 'q', 'chi1', 'chi2', 'logD_Mpc', 'cos_iota',
                     'sin_beta', 'lambda_rad', 'psi', 'phi0', 't_pre_hours']
# the same eleven columns under the names used by the calibration map
RECAL_ORDER = ['Mc', 'q', 'c1', 'c2', 'logD', 'cos_inc', 'sin_beta', 'lambda',
               'psi', 'phi0', 't_pre_hours']

CASES = {k: v['src'] for k, v in SANGRIA.items()}


# simulation
def simulate_event(case=1, n_ucb=30, t_pre=24.0, seed=0, verbose=True):
    """One month of simulated LISA data containing a Sangria MBHB.

    Returns a dict with the generator, the injected parameters, the noiseless
    MBHB template, the three raw data sets and the injected UCB list. `t_pre`
    is the time between the end of the data and the merger [hours]."""
    S = CASES[case]
    gen = UCBHetGenerator(output_dir=None)
    np.random.seed(7 + seed)                          # UCB draw + alert reference
    for _ in range(2):                                # warm up the GPU waveform
        Pw, _ = gen.draw_priors(64); gen.gen_wf(Pw)

    m1, m2 = S['m1'], S['m2']; eta = m1 * m2 / (m1 + m2) ** 2
    Mc = (m1 + m2) * eta ** 0.6; q = m1 / m2
    t_ref = (gen.t_obs - t_pre * 3600) / 3600
    p_true = np.array([[Mc, q, S['chi1'], S['chi2'], np.log10(S['dist_Mpc']), t_ref,
                        np.cos(S['iota']), np.sin(S['beta']), S['lam'], S['psi'], S['phi0']]])
    A, E = gen.gen_wf(p_true)                         # raw MBHB A/E
    hA = A[0]; hE = E[0]
    rng = cp.random.RandomState(seed)
    nA = (rng.standard_normal(gen.n_bins) + 1j * rng.standard_normal(gen.n_bins)) * gen.sigma0 * gen.sqrt_psd_g
    nE = (rng.standard_normal(gen.n_bins) + 1j * rng.standard_normal(gen.n_bins)) * gen.sigma0 * gen.sqrt_psd_g
    snr = float(cp.sqrt(4 * gen.df * cp.sum((cp.abs(hA) ** 2 + cp.abs(hE) ** 2) / gen.psd_g)))
    if verbose:
        print(f"MBHB SNR={snr:.0f}; injecting {n_ucb} UCBs + confusion")

    # resolved UCBs; `oracle_centers` are the window centres a detector with the
    # measured centre error would return (one per injected source)
    uA = cp.zeros(gen.n_bins, cp.complex128); uE = cp.zeros(gen.n_bins, cp.complex128)
    oracle_centers, ucbs = [], []
    for _ in range(n_ucb):
        f0 = np.random.uniform(max(UCB_F_LO_MHZ, F_MIN * 1e3), min(UCB_F_HI_MHZ, F_MAX * 1e3)) * 1e-3
        k0 = f0 / gen.df; s0 = int(round(k0 - W // 2))
        if s0 < 1 or s0 + W >= gen.n_bins:
            continue
        m = ucb(f_lims=np.array([s0 * gen.df, (s0 + W) * gen.df]), f_inds=[s0, s0 + W],
                gpu=True, tobs=gen.t_obs, dt=gen.dt)
        p = np.array([np.random.uniform(*UCB_LOG10A), f0 * 1e3, FDOT_LOG10,
                      np.random.uniform(0, 2 * np.pi), np.random.uniform(-1, 1),
                      np.random.uniform(0, np.pi), np.random.uniform(0, 2 * np.pi),
                      np.random.uniform(-1, 1)])
        a, e, _ = m.eval(p.reshape(1, 8))
        si = int(cp.asnumpy(cp.asarray(m._template.start_inds)).ravel()[0])
        a = cp.asarray(a).ravel(); e = cp.asarray(e).ravel(); L = a.shape[0]
        lo, hi = si, min(si + L, gen.n_bins)
        uA[lo:hi] += a[:hi - lo]; uE[lo:hi] += e[:hi - lo]
        det = int(round(k0 + np.random.uniform(-DET_ERR_BINS, DET_ERR_BINS)))
        oracle_centers.append(det)
        usnr = float(cp.sqrt(4 * gen.df * cp.sum(
            (cp.abs(a[:hi - lo]) ** 2 + cp.abs(e[:hi - lo]) ** 2) / gen.psd_g[lo:hi])))
        # The UCB windows use the generator's native indexing (array index =
        # f0 / df), so in the data the line sits one grid offset (freqs[0],
        # 10 uHz) above the waveform's nominal f0. `f_data_hz` is the physical
        # frequency of the line in the data; the response difference over
        # 10 uHz is negligible.
        ucbs.append(dict(f0_hz=float(f0), f_data_hz=float(f0 + float(gen.freqs[0])),
                         log10A=float(p[0]), snr=usnr))
    # confusion realization
    cr_A = gen.conf_sigma * (cp.random.standard_normal(gen.n_bins) + 1j * cp.random.standard_normal(gen.n_bins))
    cr_E = gen.conf_sigma * (cp.random.standard_normal(gen.n_bins) + 1j * cp.random.standard_normal(gen.n_bins))

    injection = dict(Mc=float(Mc), q=float(q), chi1=S['chi1'], chi2=S['chi2'],
                     logD=float(np.log10(S['dist_Mpc'])), lam=S['lam'],
                     sinb=float(np.sin(S['beta'])),
                     cos_iota=float(np.cos(S['iota'])), iota=S['iota'],
                     psi=S['psi'], phi0=S['phi0'],
                     t_pre_hr=t_pre,
                     t_ref_s=float(gen.t_obs - t_pre * 3600))
    return dict(gen=gen, case=case, seed=seed, t_pre=t_pre, p_true=p_true, snr=snr,
                hA=hA, hE=hE, nA=nA, nE=nE,
                d_clean=(hA + nA, hE + nE),
                d_fg=(hA + uA + cr_A + nA, hE + uE + cr_E + nE),
                oracle_centers=oracle_centers, ucbs=ucbs, n_ucb=n_ucb,
                injection=injection)


# stage 1
def detect_oracle(ev):
    """Stage 1, injected-window mode: one window per injected UCB, centre error
    uniform within +-DET_ERR_BINS bins (the detector's measured error), MBHB
    flag raised. This is the stage-1 configuration behind the paper's
    end-to-end numbers."""
    return dict(mode='oracle', centers=list(ev['oracle_centers']), smbhb=True,
                n_ucb=len(ev['oracle_centers']))


def detect_yolo(ev, weights=YOLO, conf=0.5, fmin=None, fmax=None, match_tol_hz=5e-6):
    """Stage 1 with the released spectrogram detector on the simulated data.

    The A-channel foreground data are placed on the detector's rfft grid
    (sampling interval 15 s, same frequency resolution 1/T_obs), whitened by
    the total PSD and scanned with the tiled detector (yolo/inference_fullband).
    Returns window centres in units of the generator's frequency bins, the MBHB
    flag and, because the injection is known, the detection bookkeeping."""
    from yolo.inference_fullband import FullBandYOLO
    from yolo.yolo_config import DT as DT_IMG
    gen = ev['gen']
    f = cp.asnumpy(gen.freqs)
    n_samples = int(round(gen.t_obs / DT_IMG))
    n_r = n_samples // 2 + 1
    k0 = int(round(f[0] / gen.df))
    fd = np.zeros(n_r, complex); psd = np.full(n_r, np.inf)
    fd[k0:k0 + gen.n_bins] = cp.asnumpy(ev['d_fg'][0])
    psd[k0:k0 + gen.n_bins] = cp.asnumpy(gen.psd_g)
    det = FullBandYOLO(weights, conf=conf, n_samples=n_samples)
    fmin = fmin if fmin is not None else max(UCB_F_LO_MHZ * 1e-3, F_MIN)
    fmax = fmax if fmax is not None else F_MAX
    cat = det.scan(fd, psd, fmin, fmax)
    f_det = np.array([d['frequency_hz'] for d in cat['ucb']])
    centers = [int(round((fd_ - f[0]) / gen.df)) for fd_ in f_det]
    # bookkeeping against the injection (physical line frequencies in the data)
    f_inj = np.array([u['f_data_hz'] for u in ev['ucbs']])
    matched_inj = [bool(len(f_det) and np.min(np.abs(f_det - fi)) < match_tol_hz) for fi in f_inj]
    matched_det = [bool(len(f_inj) and np.min(np.abs(f_inj - fd_)) < match_tol_hz) for fd_ in f_det]
    return dict(mode='yolo', centers=centers, smbhb=bool(cat['smbhb']['detected']),
                smbhb_tiles=int(cat['smbhb']['n_tiles']), n_tiles=int(cat['n_tiles']),
                n_ucb=len(centers),
                detections=[dict(frequency_hz=float(d['frequency_hz']),
                                 confidence=float(d['confidence'])) for d in cat['ucb']],
                injected_recovered=matched_inj, detection_is_injected=matched_det,
                frequency_error_hz=[float((f_det - fi)[np.argmin(np.abs(f_det - fi))])
                                    if len(f_det) else None for fi in f_inj])


# stage 2
def load_alert_model(device, model=MBHB_FLOW, norm=MBHB_NORM):
    from pe_smbhb.train_flow import build_model
    ck = torch.load(model, map_location=device, weights_only=False)
    m = build_model(ck['config']).to(device)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    nm = torch.load(norm, map_location=device, weights_only=False)
    return dict(model=m, cfg=ck['config'], epoch=ck.get('epoch'),
                fmean=nm['fd_mean'].to(device), fstd=nm['fd_std'].to(device))


def alert_posterior(gen, wA, wE, p_true, n_det, device, t_pre=24.0, ref_scale=0.45,
                    model=MBHB_FLOW, norm=MBHB_NORM, data=None, alert=None,
                    verbose=True):
    """Stage-2 alert on the UNSUBTRACTED whitened foreground data.

    The data are heterodyned against one reference waveform drawn at kernel
    scale `ref_scale` around the source (the entry point a search stage
    provides; Supplementary Note "Stage-2 amortized inference"), the flow is
    conditioned on the detected UCB count `n_det`, and the sky is obtained by
    Gibbs-iterating the pose (sky-GNPE) with K_pose = 8 draws in the last
    round. Returns the posterior samples and the Newtonian chirp-track
    quantiles used to seed stage 4 (chain initialization and Metropolis-
    Hastings proposals; never a prior, since the flow saw the same data)."""
    kbase = load_kernel(data)
    torch.manual_seed(13)                       # bit-reproducible flow sampling
    from pe_smbhb.train_flow import ref_meta_vec, residual_to_physical, UCB_N_MAX_COND
    from pe_smbhb.networks import PARAM_RANGES
    from pe_smbhb.inference import draw_ref, het_features, gnpe_marg
    al = alert or load_alert_model(device, model, norm)
    m, cfg, fmean, fstd = al['model'], al['cfg'], al['fmean'], al['fstd']
    res_p = cfg.get('residual_params') or None

    ref = draw_ref(p_true[0], t_pre, kbase, np.random.RandomState(11),
                   s=ref_scale, wide_spin=True)
    t0 = time.time()
    feats = het_features(gen, wA, wE, ref)
    fd = torch.as_tensor(cp.asnumpy(feats), dtype=torch.float32,
                         device=device).unsqueeze(0)
    fd = (fd - fmean) / fstd
    base = torch.from_numpy(ref_meta_vec(ref)).unsqueeze(0).to(device).float()
    if cfg.get('ucb_cond', False):
        base = torch.cat([base, torch.tensor(
            [[min(n_det, UCB_N_MAX_COND) / UCB_N_MAX_COND]], device=device)], dim=-1)
    samp = gnpe_marg(m, fd, base, device, 3, 4000, 1.6, seed=13)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_alert = time.time() - t0
    phys = residual_to_physical(samp, ref, kbase, params=res_p)
    lo, hi = PARAM_RANGES['logD']
    post = np.column_stack([
        phys['Mc'], phys['q'], phys['c1'], phys['c2'],
        samp[:, 4] * (hi - lo) + lo, samp[:, 5] * 2 - 1, samp[:, 6] * 2 - 1,
        samp[:, 7] * 2 * np.pi, samp[:, 8] * np.pi, samp[:, 9] * 2 * np.pi,
        phys['t_pre_hours']])
    # Newtonian chirp track f(t) per posterior sample over the observed month
    t_grid = np.linspace(0.0, gen.t_obs, 240)
    tc = gen.t_obs + post[:, 10] * 3600.0                # merger epoch [s]
    tau = np.maximum(tc[:, None] - t_grid[None, :], 60.0)
    mc_s = np.maximum(post[:, 0], 1e3) * MTSUN
    f_t = (5.0 / (256.0 * tau)) ** 0.375 * mc_s[:, None] ** -0.625 / np.pi
    q5, q50, q95 = np.percentile(f_t, [5, 50, 95], axis=0)
    from pe_smbhb.train_flow import sky_area_90
    a90 = float(sky_area_90(post[:4000, 7], post[:4000, 6]))
    if verbose:
        print(f"[alert] stage-2 posterior: {len(post)} samples in {t_alert:.2f} s, "
              f"90% sky {a90:.0f} deg^2")
    return dict(alert_samples=post.astype(np.float32),
                alert_param_names=np.array(ALERT_PARAM_NAMES),
                track_t=t_grid, track_f_lo=q5, track_f_med=q50, track_f_hi=q95,
                _sky90=a90, _seconds=t_alert, _reference=ref)


def recalibrate_alert(alert_samples, recal_map=MBHB_RECAL):
    """Post-hoc calibration of the alert posterior (Methods, "Post-hoc
    calibration"): per-parameter, rank-based and order-preserving, so the joint
    structure is unchanged. Returns the input unchanged if the map is absent."""
    from heron.recal import load_recal, recalibrate_samples
    recal = load_recal(recal_map)
    return recalibrate_samples(np.asarray(alert_samples, float), recal, RECAL_ORDER)


# stage 3
def load_ucb_flow(device, path=UCB_FLOW):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = UCBFlow(W=ck['config']['W']).to(device)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m, ck['config']['sigma0']


def fstat_subtract(gen, dA, dE, c_bin, m_flow, sigma0, device, n_amp=16, info=None):
    """Subtract one resolved UCB from full-band whitened dA/dE: the window flow
    seeds the frequency, then a two-stage frequency scan with a per-channel
    rank-4 F-statistic projection removes the closed-form optimum. Returns the
    updated dA, dE and the local residual norm (None if the window is off-band).
    If `info` is a dict it receives the catalogue entry of the source."""
    s0 = c_bin - W // 2
    if s0 < 1 or s0 + W >= gen.n_bins:
        return dA, dE, None
    # window flow posterior
    xw = cp.stack([dA[s0:s0 + W].real, dA[s0:s0 + W].imag,
                   dE[s0:s0 + W].real, dE[s0:s0 + W].imag], -1).astype(cp.float32)
    x = torch.as_tensor(cp.asnumpy(xw), device=device).unsqueeze(0) / sigma0
    with torch.no_grad():
        s = m_flow.sample(x, 64).squeeze(0).cpu().numpy()
    s = np.clip(s, 1e-4, 1 - 1e-4)
    df_med = (s[:, 1].mean()) * (RANGES['df_bins'][1] - RANGES['df_bins'][0]) + RANGES['df_bins'][0]
    lam = s[:, 5].mean() * (RANGES['lambda'][1] - RANGES['lambda'][0]) + RANGES['lambda'][0]
    sb = s[:, 6].mean() * 2 - 1
    d_vec = np.concatenate([cp.asnumpy(dA[s0:s0 + W]), cp.asnumpy(dE[s0:s0 + W])])
    wpsd = cp.asnumpy(gen.sqrt_psd_g[s0:s0 + W])
    rngf = np.random.default_rng(c_bin)
    best_res, best_fit, best_dfb = np.inf, None, df_med
    for stage in range(2):
        grid = (df_med + np.linspace(-1.5, 1.5, 13)) if stage == 0 \
            else best_dfb + np.linspace(-0.15, 0.15, 13)
        for dfb in grid:
            f0g = (c_bin + dfb) * gen.df
            P = np.column_stack([np.full(n_amp, -20.8), np.full(n_amp, f0g * 1e3),
                                 np.full(n_amp, FDOT_LOG10), rngf.uniform(0, 2 * np.pi, n_amp),
                                 rngf.uniform(-1, 1, n_amp), rngf.uniform(0, np.pi, n_amp),
                                 np.full(n_amp, lam), np.full(n_amp, sb)])
            mdl = ucb(f_lims=np.array([s0 * gen.df, (s0 + W) * gen.df]), f_inds=[s0, s0 + W],
                      gpu=True, tobs=gen.t_obs, dt=gen.dt)
            a_r, e_r, _ = mdl.eval(P)
            a_r = cp.asnumpy(cp.asarray(a_r)); e_r = cp.asnumpy(cp.asarray(e_r))
            sis = cp.asnumpy(cp.asarray(mdl._template.start_inds)).ravel().astype(int)
            K, L = a_r.shape
            basis = np.zeros((K, 2 * W), complex)
            for k in range(K):
                aa = np.zeros(W, complex); ee = np.zeros(W, complex)
                lo, hi = max(s0, sis[k]), min(s0 + W, sis[k] + L)
                if hi > lo:
                    aa[lo - s0:hi - s0] = a_r[k, lo - sis[k]:hi - sis[k]]
                    ee[lo - s0:hi - s0] = e_r[k, lo - sis[k]:hi - sis[k]]
                basis[k] = np.concatenate([aa / wpsd, ee / wpsd])
            fit = np.zeros(2 * W, complex)
            for ch in (slice(0, W), slice(W, 2 * W)):
                _, _, Vh = np.linalg.svd(basis[:, ch], full_matrices=False)
                V4 = Vh[:4]; fit[ch] = V4.T @ (V4.conj() @ d_vec[ch])
            res = float(np.linalg.norm(d_vec - fit))
            if res < best_res:
                best_res, best_fit, best_dfb = res, fit, dfb
    dA[s0:s0 + W] -= cp.asarray(best_fit[:W])
    dE[s0:s0 + W] -= cp.asarray(best_fit[W:])
    if info is not None:
        f0_idx0 = float(cp.asnumpy(gen.freqs[0]))
        fit_snr = float(np.linalg.norm(best_fit) / sigma0)
        info.update(window_center_bin=int(c_bin),
                    f0_hz=float(f0_idx0 + (c_bin + best_dfb) * gen.df),
                    f0_flow_hz=float(f0_idx0 + (c_bin + df_med) * gen.df),
                    lambda_rad=float(lam), sin_beta=float(sb),
                    fit_snr=fit_snr, window_residual=best_res)
    return dA, dE, best_res


def subtract_catalogue(gen, wA, wE, centers, device, m_flow=None, sigma0=None,
                       ucb_flow=UCB_FLOW):
    """Stage 3 over every detected window. Returns the subtracted whitened data,
    the number of subtracted sources and the catalogue (one dict per source)."""
    if m_flow is None:
        m_flow, sigma0 = load_ucb_flow(device, ucb_flow)
    torch.manual_seed(0)         # flow seeds: independent of the alert's RNG use
    n_sub, catalogue = 0, []
    for c_bin in centers:
        entry = {}
        wA, wE, r = fstat_subtract(gen, wA, wE, int(c_bin), m_flow, sigma0, device, info=entry)
        if r is not None:
            n_sub += 1
            catalogue.append(entry)
    return wA, wE, n_sub, catalogue


def global_alpha(ev, wA, sigma0):
    """Whitened residual amplitude of everything that is not MBHB or instrument
    noise, in units of the noise level. In this simulation it contains the
    unresolved confusion realization (which is deliberately left in the data
    and carried by the PSD) plus what the subtraction left of the resolved
    lines; the subtraction-only figure of the paper (alpha = 0.14) is measured
    without confusion by evaluation/ucb_subtract_test.py."""
    gen = ev['gen']
    res = wA - (ev['hA'] / gen.sqrt_psd_g + ev['nA'] / gen.sqrt_psd_g)
    return float(cp.sqrt(cp.mean(cp.abs(res) ** 2)) / sigma0)


# hand-off
CONVENTIONS = dict(
    time_axis='t=0 of dA/dE == t=0 of the BBHx analytic '
              'equal-armlength orbit model; window [0, Tobs]; '
              'merger at t_ref_s = Tobs - t_pre_hr*3600',
    orbits='BBHx fast FD response (Marsat-Babak), analytic '
           'equal-armlength constellation, L=2.5e9 m, '
           'zero initial orbital/rotation phase at t=0',
    frame='(lam, sinb) = SSB ecliptic longitude / sin(latitude), '
          'constant (NOT LISA-frame)',
    tdi='1st-generation TDI, A=(Z-X)/sqrt2, E=(X-2Y+Z)/sqrt6 '
        '(noise model carries the matching 8sin^2(x) transfer)',
    waveform='BBHx PhenomD (run_phenomd=True), (2,2) mode only',
    f_ref='f_ref=0 -> set internally to fmax=argmax f^2|A22|; '
          'phi0 and t_ref are defined AT f_ref (~merger)',
    distance='D_L = 10^logD Mpc (luminosity)')

HANDOFF_README = (
    "HERON -> HyperWave export. Raw d(f) + one-sided PSD, band 0.1-4 mHz, "
    "T_obs=YRSID/12, A/E TDI. Whittle lnL: -2 df Sum |dA-hA|^2/psdA + (A->E).\n"
    "clean = MBHB+noise (ideal); foreground = +UCBs+confusion (real); "
    "subtracted = foreground after HERON removal of RESOLVED UCBs.\n"
    "psdA/psdE = the RECOMMENDED PSD per file: instrument-only for clean; "
    "instrument+confusion (psd_total) for foreground/subtracted, because the "
    "unresolved confusion is stationary noise and belongs in the PSD (resolved "
    "lines cannot be treated as noise and ARE subtracted). Keys psd_instr and "
    "psd_conf are provided separately; feel free to FIT the noise level per "
    "your Gaussian-likelihood noise-fitting analysis.\n\n"
    "alert_samples [N,11] (columns alert_param_names) = the HERON stage-2\n"
    "flow posterior, computed from the SAME data. Safe uses (bias-free):\n"
    "  1. initialize RJMCMC chains from these draws (cuts burn-in);\n"
    "  2. MH independence proposals for (lambda, beta) and wavelet birth\n"
    "     moves along the chirp track track_f_med(track_t) [5-95% band\n"
    "     track_f_lo..track_f_hi] -- always with the MH correction.\n"
    "Do NOT multiply it in as a prior: it would double-count the data.\n\n"
    "CONVENTIONS (also in the `conventions` dict inside each npz):\n"
    "  t=0 of dA/dE = t=0 of the BBHx analytic equal-armlength orbit model\n"
    "  (zero initial constellation longitude/rotation phase); window\n"
    "  [0, Tobs], merger at injection['t_ref_s'] seconds. (lam, sinb) are\n"
    "  SSB-ecliptic at that epoch. TDI 1st generation AET. Waveform BBHx\n"
    "  PhenomD (2,2); f_ref=0 -> fmax internally; phi0/t_ref defined at\n"
    "  f_ref. The full injected extrinsics (iota, psi, phi0) are in\n"
    "  injection. hA_inj/hE_inj = the NOISELESS injected template on the\n"
    "  same grid: verify truth-in-credible-region against it directly,\n"
    "  or calibrate your orbit-epoch mapping by matched-filtering it in\n"
    "  your own response convention.\n")


def handoff_files(ev, d_sub, alert, n_sub, galpha, out, tag, verbose=True):
    """Write <tag>_{clean,foreground,subtracted}.npz + <tag>_README.txt: raw
    A/E data on the 0.1-4 mHz band with the recommended PSD per file, the
    noiseless injection, the conventions and (if given) the alert posterior."""
    gen = ev['gen']
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    f = cp.asnumpy(gen.freqs)
    band = (f >= F_MIN) & (f <= F_MAX)
    psd_instr = cp.asnumpy(gen.psd_g)[band]
    psd_conf = cp.asnumpy(gen.Sc)[band]              # confusion PSD (amp=1 realization)
    psd_total = psd_instr + psd_conf
    common = dict(freqs=f[band], df=float(gen.df), Tobs=float(gen.t_obs), dt=float(gen.dt),
                  psd_instr=psd_instr, psd_conf=psd_conf, mbhb_snr=ev['snr'],
                  # noiseless injected MBHB template on the same band/grid
                  hA_inj=cp.asnumpy(ev['hA'])[band], hE_inj=cp.asnumpy(ev['hE'])[band],
                  injection=ev['injection'], conventions=CONVENTIONS)
    if alert is not None:
        common.update({k: v for k, v in alert.items() if not k.startswith('_')})
    files = {}
    for name, (dA, dE), use_psd, extra in [
        ('clean', ev['d_clean'], psd_instr, {}),
        ('foreground', ev['d_fg'], psd_total, {'n_ucb_injected': ev['n_ucb']}),
        ('subtracted', d_sub, psd_total, {'n_ucb_injected': ev['n_ucb'],
                                          'n_ucb_subtracted': n_sub, 'global_alpha': galpha})]:
        fn = out / f"{tag}_{name}.npz"
        np.savez(fn, dA=cp.asnumpy(dA)[band], dE=cp.asnumpy(dE)[band],
                 psdA=use_psd, psdE=use_psd, **common, **extra)
        files[name] = fn
        if verbose:
            print(f"  saved {fn}  ({band.sum()} bins, {F_MIN*1e3:.1f}-{F_MAX*1e3:.1f} mHz)")
    (out / f"{tag}_README.txt").write_text(HANDOFF_README)
    return files


def run_stages_1_to_3(case=1, n_ucb=30, t_pre=24.0, seed=0, device='cuda:0',
                      detector='oracle', no_alert=False, ref_scale=0.45,
                      model=MBHB_FLOW, norm=MBHB_NORM, ucb_flow=UCB_FLOW, data=None,
                      out='exports', tag='sangria_synergy', verbose=True):
    """Simulate one event and run detection, alert, catalogue + subtraction and
    the hand-off export. Returns a summary dict (also the building block of
    heron/pipeline.py and pointing/export.py)."""
    device = torch.device(device)
    timing = {}
    t0 = time.time()
    ev = simulate_event(case, n_ucb, t_pre, seed, verbose=verbose)
    gen = ev['gen']
    timing['simulate_s'] = time.time() - t0

    t0 = time.time()
    det = detect_oracle(ev) if detector == 'oracle' else detect_yolo(ev)
    timing['detect_s'] = time.time() - t0
    if verbose and detector == 'yolo':
        rec = det['injected_recovered']; snrs = np.array([u['snr'] for u in ev['ucbs']])
        print(f"[detect] YOLO: {det['n_ucb']} UCB detections over {det['n_tiles']} tiles, "
              f"MBHB flag={det['smbhb']} ({det['smbhb_tiles']} tiles); recovered "
              f"{int(np.sum(rec))}/{len(rec)} injected "
              f"({int(np.sum(np.array(rec) & (snrs >= 10)))}/{int(np.sum(snrs >= 10))} of those with SNR>=10); "
              f"{int(np.sum(~np.array(det['detection_is_injected'], bool)))} detections without an injected counterpart")

    m_flow, sigma0 = load_ucb_flow(device, ucb_flow)
    wA = (ev['d_fg'][0] / gen.sqrt_psd_g).astype(cp.complex64)
    wE = (ev['d_fg'][1] / gen.sqrt_psd_g).astype(cp.complex64)

    # stage 2 on the unsubtracted data
    alert = None
    if not no_alert and det['smbhb']:
        alert = alert_posterior(gen, wA.copy(), wE.copy(), ev['p_true'], det['n_ucb'], device,
                                t_pre=t_pre, ref_scale=ref_scale, model=model, norm=norm,
                                data=data, verbose=verbose)
        timing['alert_s'] = alert['_seconds']

    # stage 3
    t0 = time.time()
    wA, wE, n_sub, catalogue = subtract_catalogue(gen, wA, wE, det['centers'], device,
                                                  m_flow=m_flow, sigma0=sigma0)
    timing['catalogue_subtract_s'] = time.time() - t0
    galpha = global_alpha(ev, wA, sigma0)
    d_sub = wA * gen.sqrt_psd_g, wE * gen.sqrt_psd_g        # de-whiten back to raw
    if verbose:
        print(f"subtracted {n_sub}/{len(det['centers'])} UCBs; global residual alpha={galpha:.3f}")

    files = handoff_files(ev, d_sub, alert, n_sub, galpha, out, tag, verbose=verbose)
    summary = dict(case=case, seed=seed, t_pre_hours=t_pre, mbhb_snr=ev['snr'],
                   alert_ref_scale=ref_scale,
                   n_ucb_injected=len(ev['ucbs']), detector=detector,
                   n_ucb_detected=det['n_ucb'], smbhb_flag=det['smbhb'],
                   n_ucb_subtracted=n_sub, global_alpha=galpha,
                   alert_sky90_deg2=(alert['_sky90'] if alert else None),
                   alert_n_samples=(int(len(alert['alert_samples'])) if alert else 0),
                   timing_s=timing, handoff={k: str(v) for k, v in files.items()})
    return dict(summary=summary, event=ev, detection=det, alert=alert,
                catalogue=catalogue, files=files)
