"""Realistic full-band detection figure: the Sangria Case-1 MBHB (SNR 357)
plus the ACTUAL month-one resolved UCB population (SNR>10), with galactic
confusion, rendered as a whitened spectrogram and run through the tiled YOLO
detector. Replaces the toy 6-UCB demo (fig1b_detection / Fig. A3).

Data: sangria_1month_snr.npz (evaluation/resolved_population_1month.py;
      published file in paper_products/).
Out: results/figures/fig1b_detection.png  (+ .pdf)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import YOLO as MODEL, product_in, figure_out

from yolo.inference_fullband import FullBandYOLO
from yolo.yolo_config import T_OBS, DT, SA, SI, NGB_PATH
from yolo.dataset_generator import compute_snr
sys.path.insert(0, NGB_PATH)
from data_utils import noise_model, ucb as ucb_gen, smbhb as smbhb_gen
from config import galactic_confusion_psd

OUT = figure_out('fig1b_detection')
FMIN, FMAX = 0.5e-3, 6e-3

# Sangria Case-1 MBHB (Table I of the noise paper), scaled to SNR 357
CASE1 = dict(m1=4956676.2876, m2=4067166.60352, chi1=-0.523732715,
             chi2=-0.117412144, dist_Mpc=61097.116076, iota=1.420048341,
             beta=-1.081082148, lam=4.052962883, psi=1.22844350008,
             phi0=0.6417162631, t_pre_h=144.0)


def main():
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n = int(T_OBS / DT)
    freqs = np.fft.rfftfreq(n, DT)
    df = 1.0 / T_OBS

    # instrument + confusion PSD; whiten with the total (real-scenario)
    nm = noise_model(freqs)
    psd_i = nm.eval([SA, SI])
    psd_i = np.asarray(psd_i.get() if hasattr(psd_i, 'get') else psd_i).ravel()
    psd_i[0] = np.inf; psd_i[psd_i <= 0] = np.inf
    S_conf = galactic_confusion_psd(freqs)
    psd = psd_i + S_conf

    # noise + confusion realization
    noise, _ = nm.noise_gen([SA, SI])
    fd = np.asarray(noise.get() if hasattr(noise, 'get') else noise).ravel()
    band = (freqs > 2e-3) & (freqs < 3e-3)
    C = float(np.mean(np.abs(fd[band]) ** 2 / psd_i[band]))
    fd = fd.astype(complex)
    fd += (rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))) \
        * np.sqrt(C * S_conf / 2)
    fd[0] = 0

    # --- the real resolved UCB population (SNR>10, in the scan band) ---
    pop = np.load(product_in('sangria_1month_snr.npz'))
    m = (pop['snr'] > 10) & (pop['f0'] > FMIN) & (pop['f0'] < FMAX)
    f0s, snrs = pop['f0'][m], pop['snr'][m]
    print(f"injecting {m.sum()} real resolved UCBs ({f0s.min()*1e3:.2f}-"
          f"{f0s.max()*1e3:.2f} mHz, SNR {snrs.min():.0f}-{snrs.max():.0f})")
    for f0, snr_t in zip(f0s, snrs):
        hw = 5e-6
        s0 = int((f0 - hw) / df); s1 = int((f0 + hw) / df)
        if s0 < 1 or s1 + 1 >= len(freqs):
            continue
        u = ucb_gen(f_lims=np.array([f0 - hw, f0 + hw]), f_inds=[s0, s1 + 1],
                    gpu=True, tobs=T_OBS, dt=DT)
        p = np.array([-21.0, f0 * 1e3, -16.5, rng.uniform(0, 2 * np.pi),
                      rng.uniform(-1, 1), rng.uniform(0, np.pi),
                      rng.uniform(0, 2 * np.pi), rng.uniform(-1, 1)])
        a, _, _ = u.eval(p.reshape(1, 8))
        a = np.asarray(a.get() if hasattr(a, 'get') else a).ravel()
        sig = np.zeros(len(freqs), complex); sig[s0:s0 + len(a)] = a
        snr0 = np.sqrt(4 * df * np.sum(np.abs(sig) ** 2 / psd))   # A-only
        fd += sig * snr_t / max(snr0, 1e-30)

    # --- the Case-1 MBHB scaled to SNR 357 ---
    S = CASE1
    eta = S['m1'] * S['m2'] / (S['m1'] + S['m2']) ** 2
    Mc, q = (S['m1'] + S['m2']) * eta ** 0.6, S['m1'] / S['m2']
    t_ref = (T_OBS - S['t_pre_h'] * 3600) / 3600
    p0 = np.array([[Mc, q, S['chi1'], S['chi2'], np.log10(S['dist_Mpc']), t_ref,
                    np.cos(S['iota']), np.sin(S['beta']), S['lam'], S['psi'],
                    S['phi0']]])
    tmpl = smbhb_gen(f=freqs, modes=[(2, 2)], tstart=T_OBS, tend=0.0, gpu=True)
    As, _, _ = tmpl.eval(p0)
    As = np.asarray(As.get() if hasattr(As, 'get') else As).ravel(); As[0] = 0
    # "SNR 357" is the clean (instrument-only) definition; normalize with
    # psd_i so the injected MBHB has its true brightness in the foreground data
    snr0 = np.sqrt(4 * df * np.sum(np.abs(As) ** 2 / psd_i))      # A+E ~ x sqrt2
    fd += As * 357.0 / snr0
    print(f"MBHB scaled to clean SNR 357 (A-only raw {snr0:.0f}, peak "
          f"{freqs[np.argmax(np.abs(As)**2/psd_i)]*1e3:.2f} mHz)")

    # --- tiled YOLO detection ---
    det = FullBandYOLO(MODEL, conf=0.5)
    Sxx = det._stft_all(fd, psd)
    det._S_cache = Sxx
    det._stft_all = lambda *a, **k: Sxx
    cat = det.scan(fd, psd, FMIN, FMAX)
    print(f"detected {len(cat['ucb'])} UCBs, SMBHB={cat['smbhb']['detected']}")

    # --- figure ---
    from evaluation.plotstyle import paper_style
    paper_style()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    i0 = max(1, int(np.searchsorted(det.f_stft, FMIN)))
    i1 = int(np.searchsorted(det.f_stft, FMAX))
    Sl = np.log10(Sxx[i0:i1] + 1e-10)
    ax.imshow(Sl, aspect='auto', origin='lower',
              extent=[0, T_OBS / 86400, det.f_stft[i0] * 1e3, det.f_stft[i1] * 1e3],
              cmap='magma', vmin=np.percentile(Sl, 25), vmax=np.percentile(Sl, 99.7))
    for d in cat['ucb']:
        ax.axhline(d['frequency_hz'] * 1e3, color='cyan', lw=0.5, ls='--', alpha=0.7)
    # the true Newtonian MBHB chirp track f(t) — the SNR-357 source is faint
    # under the confusion but its inspiral runs along this line
    t_merge_day = (T_OBS - S['t_pre_h'] * 3600) / 86400
    MTSUN = 4.925490947e-6
    td = np.linspace(0, t_merge_day * 0.999, 300)
    tau = np.maximum((t_merge_day - td) * 86400, 60.0)
    f_track = (5.0 / (256 * tau)) ** 0.375 * (Mc * MTSUN) ** -0.625 / np.pi * 1e3
    ax.plot(td, np.clip(f_track, FMIN * 1e3, FMAX * 1e3), color='#ffd24d',
            lw=1.3, ls='-', alpha=0.85, label='MBHB chirp track (SNR 357)')
    ax.plot([], [], color='cyan', lw=1.2, ls='--', label=f'UCB detections ({len(cat["ucb"])})')
    ax.legend(loc='upper left', frameon=True, framealpha=0.75, fontsize=9)
    ax.set_xlim(0, T_OBS / 86400); ax.set_ylim(FMIN * 1e3, FMAX * 1e3)
    ax.set_xlabel('time [days]'); ax.set_ylabel('frequency [mHz]')
    ax.set_title(f"Realistic month-1 LISA data: MBHB (SNR 357) + "
                 f"{len(cat['ucb'])}/{m.sum()} resolved UCBs detected")
    fig.tight_layout()
    fig.savefig(str(OUT) + '.png', dpi=200, bbox_inches='tight')
    fig.savefig(str(OUT) + '.pdf', bbox_inches='tight')
    print(f"saved {OUT}.png / .pdf")


if __name__ == '__main__':
    main()
