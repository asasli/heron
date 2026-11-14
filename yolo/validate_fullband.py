"""
Full-band validation of the tiled detector (stage 1): the set of simulated
scans behind the full-band statements of the paper and Extended Data Fig. 2.

Each scan injects galactic binaries (and optionally a massive black-hole
binary) into one month of noise, runs the tiled detector over 0.5-6 mHz
(105 tiles) at the full-band operating point (confidence 0.5; MBHB flag when
at least two tiles agree) and scores the result against the injection:
recovered sources, detections without an injected counterpart, the largest
frequency error and the MBHB flag.

    python -m yolo.validate_fullband                 # all scans, ~3 min on one GPU
    python -m yolo.validate_fullband --only crowd snr10

Scans:
    val       5 UCBs at SNR 25 + MBHB, instrument noise
    conf      the same with galactic confusion in the data and in the whitening PSD
    crowd     25 UCBs (5 + 20 at random frequencies, SNR 10-63) with confusion
    lowsnr    6 UCBs at SNR 12-15
    snr10     4 UCBs at SNR 10-11 (the detection floor)
    noise     noise only (false-alarm check)
    dense20   20 UCBs at 5.5 uHz spacing inside one 110 uHz band (density ceiling)
    dense13   13 UCBs at 7 uHz spacing inside one band

Writes results/products/fullband_validation.json and one catalog + image per
scan under results/fullband/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import YOLO, RESULTS, product_out, require   # noqa: E402

DENSE20_SNR = [15, 22, 11, 35, 18, 50, 12, 27, 14, 40, 16, 30, 11, 45, 20, 13, 55, 17, 25, 12]
DENSE13_SNR = [15, 11, 30, 12, 45, 18, 24, 13, 55, 16, 20, 11, 35]
SCANS = {
    'val':     dict(freqs=[1.0, 1.7, 2.58, 3.4, 4.9], snrs=[25.0] * 5, smbhb=True, seed=0),
    'conf':    dict(freqs=[1.0, 1.7, 2.58, 3.4, 4.9], snrs=[25.0] * 5, smbhb=True, seed=1, conf_noise=True),
    'crowd':   dict(freqs=[0.8, 1.6, 2.4, 3.2, 4.6], snrs=[15.0, 20.0, 25.0, 30.0, 40.0], seed=2,
                    conf_noise=True, n_crowd=20),
    'lowsnr':  dict(freqs=[0.8, 1.3, 2.1, 3.0, 4.2, 5.5], snrs=[12.0, 13.0, 15.0, 12.0, 14.0, 13.0], seed=3),
    'snr10':   dict(freqs=[0.9, 2.4, 3.8, 5.2], snrs=[10.0, 10.0, 11.0, 10.0], seed=4),
    'noise':   dict(freqs=[], snrs=[], seed=5),
    'dense20': dict(freqs=[3.302 + 0.0055 * i for i in range(20)], snrs=DENSE20_SNR, seed=6,
                    band=(3.25, 3.46)),
    'dense13': dict(freqs=[3.310 + 0.0072 * i for i in range(13)], snrs=DENSE13_SNR, seed=7,
                    band=(3.25, 3.46)),
}


def run_scan(name, cfg, weights, conf, out):
    from yolo.inference_fullband import FullBandYOLO, simulate, overview_plot
    fd, psd, truth = simulate(cfg['freqs'], cfg['snrs'], cfg.get('smbhb', False), cfg['seed'],
                              conf_noise=cfg.get('conf_noise', False), n_crowd=cfg.get('n_crowd', 0))
    fmin, fmax = cfg.get('band', (0.5, 6.0))
    det = FullBandYOLO(weights, conf=conf)
    S = det._stft_all(fd, psd)
    det._S_cache = S
    det._stft_all = lambda *a, **k: S
    cat = det.scan(fd, psd, fmin * 1e-3, fmax * 1e-3)
    f_det = np.array([d['frequency_hz'] for d in cat['ucb']])
    f_inj = np.array([t['f0_mhz'] * 1e-3 for t in truth['ucb']])
    snr_inj = np.array([t['snr'] for t in truth['ucb']])
    tol = 5e-6
    rec = np.array([bool(len(f_det) and np.min(np.abs(f_det - f)) < tol) for f in f_inj], bool)
    spurious = int(sum(not (len(f_inj) and np.min(np.abs(f_inj - f)) < tol) for f in f_det))
    err = [float(np.min(np.abs(f_det - f))) for f, r in zip(f_inj, rec) if r]
    sm = cat['smbhb']
    confs = [d['confidence'] for d in sm['detections'] if d['confidence'] >= 0.5]
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'catalog.json').write_text(json.dumps(dict(catalog=cat, truth=truth), indent=1, default=float))
    overview_plot(det, cat, truth, fmin * 1e-3, fmax * 1e-3, d / 'fullband_scan.png')
    return dict(scan=name, n_tiles=int(cat['n_tiles']), injected=int(len(f_inj)),
                snr_min=float(snr_inj.min()) if len(snr_inj) else None,
                snr_max=float(snr_inj.max()) if len(snr_inj) else None,
                recovered=int(rec.sum()), no_counterpart=spurious,
                max_freq_error_uhz=float(max(err) * 1e6) if err else None,
                mbhb_injected=bool(truth['smbhb']), mbhb_flag=bool(sm['detected']),
                mbhb_tiles=int(sm['n_tiles']),
                mbhb_median_conf=float(np.median(confs)) if confs else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model', default=YOLO)
    ap.add_argument('--conf', type=float, default=0.5, help='full-band operating threshold')
    ap.add_argument('--only', nargs='+', choices=list(SCANS), default=None)
    ap.add_argument('--out', default=str(RESULTS / 'fullband'))
    args = ap.parse_args()
    require('yolo')
    rows = [run_scan(n, c, args.model, args.conf, args.out)
            for n, c in SCANS.items() if not args.only or n in args.only]
    print(f"\n{'scan':8s} {'tiles':>5s} {'inj':>4s} {'SNR':>9s} {'rec':>4s} {'extra':>5s} "
          f"{'max|df| uHz':>11s}  MBHB")
    for r in rows:
        snr = f"{r['snr_min']:.0f}-{r['snr_max']:.0f}" if r['injected'] else '-'
        df = f"{r['max_freq_error_uhz']:.1f}" if r['max_freq_error_uhz'] is not None else '-'
        mb = (f"flag={r['mbhb_flag']} ({r['mbhb_tiles']} tiles, median conf "
              f"{r['mbhb_median_conf']:.2f})" if r['mbhb_injected']
              else f"not injected, flag={r['mbhb_flag']}")
        print(f"{r['scan']:8s} {r['n_tiles']:5d} {r['injected']:4d} {snr:>9s} {r['recovered']:4d} "
              f"{r['no_counterpart']:5d} {df:>11s}  {mb}")
    wide = [r for r in rows if not r['scan'].startswith('dense')]
    if wide:
        print(f"\nfull-band scans: {sum(r['recovered'] for r in wide)}/{sum(r['injected'] for r in wide)} "
              f"injected UCBs recovered, {sum(r['no_counterpart'] for r in wide)} detections without "
              f"counterpart, largest frequency error "
              f"{max((r['max_freq_error_uhz'] or 0) for r in wide):.1f} uHz")
    out = product_out('fullband_validation.json')
    Path(out).write_text(json.dumps(rows, indent=1))
    print(f"saved {out}")


if __name__ == '__main__':
    main()
