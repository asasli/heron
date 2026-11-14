"""
Full-band YOLO inference by band tiling — no retrain.

The image detector (weights/yolo_lisa_best.pt) is trained on 1-month whitened
spectrograms of a single 110-uHz band (2.53-2.64 mHz). Every training image is
per-image percentile-normalized, so a whitened tile from ANY band has the same
statistics as a training image. This module slides that same 110-uHz window
across the LISA band (50% overlap), renders each tile in the exact training
convention (whiten -> STFT -> crop -> log10 -> 1/99-percentile norm -> flip ->
640x640 viridis), runs the detector per tile, and stitches the detections back
into a global-frequency catalog (duplicate detections from overlapping tiles
are clustered and the highest-confidence one kept).

A broadband SMBHB chirp appears as a vertical line in every tile it crosses;
the global SMBHB flag is raised when >= --smbhb-min-tiles tiles agree.

Usage:
    # simulated validation: UCBs injected across the band (+ optional SMBHB)
    python -m yolo.inference_fullband --model weights/yolo_lisa_best.pt \
        --simulate --ucb-freqs-mhz 1.0 1.7 2.58 3.4 4.9 --smbhb \
        --fmin-mhz 0.5 --fmax-mhz 6 --out results/fullband

    # real data: 1-month FD npz with keys `data` (or `dA`) on the full rfft
    # grid (df = 1/Tobs), optional `psd` (else the SciRD model is used)
    python -m yolo.inference_fullband --model weights/yolo_lisa_best.pt \
        --fd-npz exports/event.npz --fmin-mhz 0.5 --fmax-mhz 10
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.signal.windows import tukey
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yolo.yolo_config import (T_OBS, DT, F_MIN, F_MAX, NPERSEG, IMG_WIDTH,
                              IMG_HEIGHT, CONF_THRESHOLD, IOU_THRESHOLD,
                              CLASS_NAMES, SA, SI, NGB_PATH)
from yolo.dataset_generator import whiten, compute_snr

BAND_WIDTH = F_MAX - F_MIN                    # 110 uHz — the trained window


class FullBandYOLO:
    """Tile the trained 110-uHz detector across an arbitrary frequency range."""

    def __init__(self, model_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                 n_samples=None):
        """`n_samples` is the length of the time series at the detector's
        sampling interval DT (default: the training segment, T_OBS / DT); set
        it to round(T / DT) to scan data of a different duration T on the rfft
        grid with spacing 1 / T."""
        self.model = YOLO(model_path)
        self.conf, self.iou = conf, iou
        self.dt = DT
        self.fs = 1.0 / DT
        self.n_samples = int(n_samples) if n_samples else int(T_OBS / DT)
        self.nperseg = NPERSEG
        self.noverlap = NPERSEG - 512
        self.f_stft = np.fft.rfftfreq(self.nperseg, DT)
        # tile height in STFT rows = exactly the training band mask
        self.n_rows = int(np.sum((self.f_stft >= F_MIN) & (self.f_stft <= F_MAX)))

    # rendering
    def _stft_all(self, fd_full, psd_full):
        """Full FD data -> whitened STFT magnitude (all frequencies, once)."""
        fd = np.nan_to_num(fd_full, nan=0.0, posinf=0.0, neginf=0.0)
        fd[0] = 0
        ht = np.fft.irfft(fd, n=self.n_samples)
        ht_w = whiten(ht, self.dt, psd_full)
        ht_w *= tukey(len(ht_w), alpha=0.1)
        _, _, Zxx = stft(ht_w, fs=self.fs, nperseg=self.nperseg,
                         noverlap=self.noverlap)
        return np.abs(Zxx)

    def _tile_image(self, S_rows):
        """Render one tile of |STFT| rows in the training image convention."""
        S_log = np.log10(S_rows + 1e-10)
        vmin, vmax = np.percentile(S_log, 1), np.percentile(S_log, 99)
        S_norm = np.clip((S_log - vmin) / (vmax - vmin + 1e-12), 0, 1)
        S_norm = np.flipud(S_norm)
        img = Image.fromarray((S_norm * 255).astype(np.uint8), 'L')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
        cmap = plt.get_cmap('viridis')
        rgb = (cmap(np.array(img) / 255.0)[..., :3] * 255).astype(np.uint8)
        return Image.fromarray(rgb)

    # scan
    def scan(self, fd_full, psd_full, fmin, fmax, overlap=0.5, merge_tol=4e-6,
             smbhb_min_tiles=2, smbhb_conf=0.5):
        """Run the detector over [fmin, fmax].

        Returns dict(ucb=[{frequency_hz, confidence, tile}...],
                     smbhb=dict(detected, n_tiles, t_merge_frac, detections),
                     n_tiles=...).
        """
        S = self._stft_all(fd_full, psd_full)
        step = max(1, int(self.n_rows * (1 - overlap)))
        i_lo = max(1, int(np.searchsorted(self.f_stft, fmin)))
        i_hi = min(len(self.f_stft) - self.n_rows,
                   int(np.searchsorted(self.f_stft, fmax)))
        raw_ucb, raw_smbhb, n_tiles = [], [], 0
        for i0 in range(i_lo, i_hi + 1, step):
            rows = slice(i0, i0 + self.n_rows)
            f_lo, f_hi = self.f_stft[i0], self.f_stft[i0 + self.n_rows - 1]
            img = self._tile_image(S[rows, :])
            res = self.model(img, conf=self.conf, iou=self.iou, verbose=False)[0]
            n_tiles += 1
            for box in res.boxes:
                cls = CLASS_NAMES[int(box.cls[0])]
                conf = float(box.conf[0])
                xc, yc = float(box.xywhn[0][0]), float(box.xywhn[0][1])
                if cls == 'ucb':
                    raw_ucb.append(dict(
                        frequency_hz=f_lo + (1.0 - yc) * (f_hi - f_lo),
                        confidence=conf, tile=n_tiles - 1,
                        tile_band_mhz=[f_lo * 1e3, f_hi * 1e3]))
                else:
                    raw_smbhb.append(dict(
                        t_frac=xc, confidence=conf, tile=n_tiles - 1,
                        tile_band_mhz=[f_lo * 1e3, f_hi * 1e3]))
        ucb = self._merge_ucb(raw_ucb, merge_tol)
        # a real chirp crosses MANY tiles at high confidence (median ~0.96);
        # per-tile artifacts sit just above the box threshold — gate on conf
        strong = [d for d in raw_smbhb if d['confidence'] >= smbhb_conf]
        smbhb_tiles = len({d['tile'] for d in strong})
        smbhb = dict(detected=smbhb_tiles >= smbhb_min_tiles,
                     n_tiles=smbhb_tiles,
                     t_merge_frac=(float(np.median([d['t_frac'] for d in strong]))
                                   if strong else None),
                     detections=raw_smbhb)
        return dict(ucb=ucb, smbhb=smbhb, n_tiles=n_tiles)

    @staticmethod
    def _merge_ucb(dets, tol):
        """Cluster detections within `tol` Hz (duplicates from tile overlap),
        keep the highest-confidence member of each cluster."""
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: d['frequency_hz'])
        merged, cur = [], [dets[0]]
        for d in dets[1:]:
            if d['frequency_hz'] - cur[-1]['frequency_hz'] < tol:
                cur.append(d)
            else:
                merged.append(max(cur, key=lambda x: x['confidence']))
                cur = [d]
        merged.append(max(cur, key=lambda x: x['confidence']))
        return merged


# simulate
def simulate(ucb_freqs_mhz, ucb_snrs, with_smbhb, seed, conf_noise=False,
             n_crowd=0, crowd_band_mhz=(0.5, 4.0)):
    """Inject UCBs at arbitrary frequencies (+ optional SMBHB, + optional
    unresolved confusion and a crowd of random resolved UCBs) into SciRD
    noise on the full 1-month rfft grid. Returns (fd, psd, truth).

    With conf_noise the data gain a Gaussian confusion realization and the
    whitening PSD becomes S_instr + S_conf (unresolved foreground belongs in
    the PSD); source SNRs are then defined against the TOTAL PSD."""
    sys.path.insert(0, NGB_PATH)
    from data_utils import noise_model, ucb as ucb_gen, smbhb as smbhb_gen
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    n_samples = int(T_OBS / DT)
    freqs = np.fft.rfftfreq(n_samples, DT)
    nm = noise_model(freqs)
    psd = nm.eval([SA, SI])
    psd = np.asarray(psd.get() if hasattr(psd, 'get') else psd).ravel()
    psd[0] = np.inf
    psd[psd <= 0] = np.inf
    noise, _ = nm.noise_gen([SA, SI])
    fd = np.asarray(noise.get() if hasattr(noise, 'get') else noise).ravel()
    fd[0] = 0

    if conf_noise:
        from config import galactic_confusion_psd
        S_conf = galactic_confusion_psd(freqs)
        # calibrate the FD-amplitude<->PSD convention on the instrument
        # realization itself, then draw the confusion at the same convention
        band = (freqs > 2e-3) & (freqs < 3e-3)
        C = float(np.mean(np.abs(fd[band]) ** 2 / psd[band]))
        cr = (rng.standard_normal(len(freqs))
              + 1j * rng.standard_normal(len(freqs))) * np.sqrt(C * S_conf / 2)
        fd += cr
        psd = psd + S_conf

    df = 1.0 / T_OBS
    truth = dict(ucb=[], smbhb=None)

    if n_crowd:
        lo, hi = crowd_band_mhz
        crowd_f, guard = [], 20e-3  # min separation 20 uHz (in mHz units)
        while len(crowd_f) < n_crowd:
            f = rng.uniform(lo, hi)
            if all(abs(f - g) > guard for g in crowd_f + list(ucb_freqs_mhz)):
                crowd_f.append(f)
        ucb_freqs_mhz = list(ucb_freqs_mhz) + crowd_f
        ucb_snrs = list(ucb_snrs) + list(10 ** rng.uniform(1.0, 1.8, n_crowd))

    for f0_mhz, snr_t in zip(ucb_freqs_mhz, ucb_snrs):
        f0 = f0_mhz * 1e-3
        hw = 5e-6
        s0 = int((f0 - hw) / df)
        s1 = int((f0 + hw) / df)
        m = ucb_gen(f_lims=np.array([f0 - hw, f0 + hw]), f_inds=[s0, s1 + 1],
                    gpu=True, tobs=T_OBS, dt=DT)
        p = np.array([-21.5, f0_mhz, -16.5, rng.uniform(0, 2 * np.pi),
                      rng.uniform(-1, 1), rng.uniform(0, np.pi),
                      rng.uniform(0, 2 * np.pi), rng.uniform(-1, 1)])
        A, _, _ = m.eval(p.reshape(1, 8))
        A = np.asarray(A.get() if hasattr(A, 'get') else A).ravel()
        sig = np.zeros(len(freqs), dtype=np.complex128)
        sig[s0:s0 + len(A)] = A
        snr0 = compute_snr(freqs, sig, psd, fmin=f0 - hw, fmax=f0 + hw)
        sig *= snr_t / max(snr0, 1e-30)
        fd += sig
        truth['ucb'].append(dict(f0_mhz=f0_mhz, snr=snr_t))

    if with_smbhb:
        m1 = rng.uniform(1e6, 5e6)
        m2 = rng.uniform(5e5, min(m1, 3e6))
        eta = m1 * m2 / (m1 + m2) ** 2
        Mc, q = (m1 + m2) * eta ** 0.6, m1 / m2
        t_merge_h = rng.uniform(48, 600)
        p0 = np.array([[Mc, q, rng.uniform(-.5, .5), rng.uniform(-.5, .5),
                        np.log10(rng.uniform(200, 2000)),
                        (T_OBS - t_merge_h * 3600) / 3600,
                        rng.uniform(-1, 1), rng.uniform(-1, 1),
                        rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi),
                        rng.uniform(0, 2 * np.pi)]])
        tmpl = smbhb_gen(f=freqs, modes=[(2, 2)], tstart=T_OBS, tend=0.0, gpu=True)
        As, _, _ = tmpl.eval(p0)
        As = np.asarray(As.get() if hasattr(As, 'get') else As).ravel()
        As[0] = 0
        fd += As
        truth['smbhb'] = dict(t_merge_hours=float(t_merge_h),
                              snr=float(compute_snr(freqs, As, psd,
                                                    fmin=1e-4, fmax=1e-2)))
    return fd, psd, truth


def overview_plot(det, catalog, truth, fmin, fmax, out_png):
    """Full-band whitened spectrogram with detected/injected UCB lines."""
    fig, ax = plt.subplots(figsize=(14, 8))
    S = det._S_cache
    i0 = max(1, int(np.searchsorted(det.f_stft, fmin)))
    i1 = int(np.searchsorted(det.f_stft, fmax))
    ax.imshow(np.log10(S[i0:i1] + 1e-10), aspect='auto', origin='lower',
              extent=[0, T_OBS / 86400, det.f_stft[i0] * 1e3, det.f_stft[i1] * 1e3],
              cmap='viridis')
    for d in catalog['ucb']:
        ax.axhline(d['frequency_hz'] * 1e3, color='cyan', lw=0.8, ls='--')
    if truth:
        for t in truth['ucb']:
            ax.axhline(t['f0_mhz'], color='w', lw=0.5, ls=':')
    if catalog['smbhb']['detected']:
        ax.axvline(catalog['smbhb']['t_merge_frac'] * T_OBS / 86400,
                   color='magenta', lw=1.2, ls='--')
    ax.set_xlabel('time [days]')
    ax.set_ylabel('frequency [mHz]')
    ax.set_title(f"full-band scan: {len(catalog['ucb'])} UCBs "
                 f"(cyan dashed = detected, white dotted = injected), "
                 f"SMBHB={'yes' if catalog['smbhb']['detected'] else 'no'}")
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Full-band tiled YOLO inference")
    ap.add_argument('--model', required=True)
    ap.add_argument('--fd-npz', help='real-data npz: data/dA on the full rfft grid')
    ap.add_argument('--simulate', action='store_true')
    ap.add_argument('--ucb-freqs-mhz', type=float, nargs='+',
                    default=[1.0, 1.7, 2.58, 3.4, 4.9])
    ap.add_argument('--ucb-snr', type=float, nargs='+', default=None,
                    help='per-source SNR (default 25 each)')
    ap.add_argument('--smbhb', action='store_true')
    ap.add_argument('--conf-noise', action='store_true',
                    help='add galactic confusion (realization in the data, '
                         'S_conf in the whitening PSD)')
    ap.add_argument('--crowd', type=int, default=0,
                    help='additionally inject N random resolved UCBs '
                         '(SNR 10-63) in --crowd-band')
    ap.add_argument('--crowd-band-mhz', type=float, nargs=2, default=[0.5, 4.0])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fmin-mhz', type=float, default=0.5)
    ap.add_argument('--fmax-mhz', type=float, default=6.0)
    # 0.5 (not the single-image CONF_THRESHOLD=0.25): a 105-tile scan multiplies
    # the false-alarm chances, and true detections score >=0.92 down to SNR 10
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--out', default='results/fullband')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    truth = None
    if args.simulate:
        snrs = args.ucb_snr or [25.0] * len(args.ucb_freqs_mhz)
        fd, psd, truth = simulate(args.ucb_freqs_mhz, snrs, args.smbhb,
                                  args.seed, conf_noise=args.conf_noise,
                                  n_crowd=args.crowd,
                                  crowd_band_mhz=tuple(args.crowd_band_mhz))
    else:
        z = np.load(args.fd_npz)
        fd = np.asarray(z['data'] if 'data' in z else z['dA']).ravel()
        if 'psd' in z:
            psd = np.asarray(z['psd']).ravel()
        else:
            sys.path.insert(0, NGB_PATH)
            from data_utils import noise_model
            freqs = np.fft.rfftfreq(int(T_OBS / DT), DT)
            p = noise_model(freqs).eval([SA, SI])
            psd = np.asarray(p.get() if hasattr(p, 'get') else p).ravel()
            psd[0] = np.inf
            psd[psd <= 0] = np.inf

    det = FullBandYOLO(args.model, conf=args.conf)
    S = det._stft_all(fd, psd)
    det._S_cache = S

    # reuse the precomputed STFT inside scan
    det._stft_all = lambda *a, **k: S
    catalog = det.scan(fd, psd, args.fmin_mhz * 1e-3, args.fmax_mhz * 1e-3)

    print(f"\nscanned {catalog['n_tiles']} tiles "
          f"({args.fmin_mhz}-{args.fmax_mhz} mHz, 110-uHz window, 50% overlap)")
    print(f"UCBs detected: {len(catalog['ucb'])}")
    for d in catalog['ucb']:
        print(f"  f = {d['frequency_hz']*1e3:.4f} mHz  (conf {d['confidence']:.2f})")
    sm = catalog['smbhb']
    print(f"SMBHB: {'DETECTED' if sm['detected'] else 'no'} "
          f"({sm['n_tiles']} tiles)")
    if truth:
        print("\n--- truth ---")
        for t in truth['ucb']:
            match = min((abs(d['frequency_hz'] * 1e3 - t['f0_mhz'])
                         for d in catalog['ucb']), default=np.inf)
            print(f"  injected {t['f0_mhz']:.4f} mHz (SNR {t['snr']:.0f}) -> "
                  f"nearest detection {match*1e3:.1f} uHz away "
                  f"{'OK' if match < 5e-3 else 'MISS'}")
        if truth['smbhb']:
            print(f"  injected SMBHB (SNR {truth['smbhb']['snr']:.0f}, "
                  f"merger T+{truth['smbhb']['t_merge_hours']:.0f}h) -> "
                  f"{'OK' if sm['detected'] else 'MISS'}")

    (out / 'catalog.json').write_text(json.dumps(
        dict(catalog=catalog, truth=truth), indent=2, default=float))
    overview_plot(det, catalog, truth, args.fmin_mhz * 1e-3,
                  args.fmax_mhz * 1e-3, out / 'fullband_scan.png')
    print(f"\nsaved: {out}/catalog.json, {out}/fullband_scan.png")


if __name__ == '__main__':
    main()
