"""
LISA-YOLO Dataset Generator
============================

Correct approach:
1. Generate noise + signals in frequency domain
2. iFFT to time domain
3. Whiten in time domain (hf / sqrt(PSD) / dt)
4. STFT on whitened data
5. Crop to band and normalize

SNR filtering: only include sources with SNR > MIN_SNR

Classes:
- 0: UCB (horizontal line)
- 1: SMBHB (vertical line near merger)
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.signal.windows import tukey

from yolo.yolo_config import *

# Add NGB path
sys.path.insert(0, NGB_PATH)
from data_utils import ucb, smbhb, noise_model


def compute_snr(f, d, S, fmin=None, fmax=None):
    """
    Compute SNR of a signal.
    
    Parameters:
    -----------
    f : array
        Frequency array
    d : array
        Signal in frequency domain (complex)
    S : array
        Noise PSD
    fmin, fmax : float, optional
        Frequency limits for integration
    
    Returns:
    --------
    snr : float
        Signal-to-noise ratio
    """
    imax = len(f) - 1
    df = f[1] - f[0]
    
    if fmax is not None:
        imax_arr = np.argwhere(f >= fmax)
        if len(imax_arr) > 0:
            imax = imax_arr[0][0]
    
    imin = 1
    if fmin is not None:
        imin_arr = np.argwhere(f >= fmin)
        if len(imin_arr) > 0:
            imin = imin_arr[0][0]
    
    # SNR formula
    snr = 2 * np.sqrt(df * np.sum(
        np.real(d[imin:imax] * np.conjugate(d[imin:imax]) / S[imin:imax])
    ))
    
    return snr


def whiten(data, dt, psd):
    """
    Whiten time-domain data.
    
    Parameters:
    -----------
    data : array
        Time-domain data
    dt : float
        Sampling interval
    psd : array
        Noise PSD at rfft frequencies
    
    Returns:
    --------
    white_ht : array
        Whitened time-domain data
    """
    Nt = len(data)
    hf = np.fft.rfft(data)
    white_hf = hf / np.sqrt(psd) / dt
    white_hf[0] = 0.0  # Remove DC
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


class LISADatasetGenerator:
    """Generate LISA spectrograms with UCB and SMBHB sources."""
    
    def __init__(self, output_dir: str = DATA_DIR, use_gpu: bool = True):
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu
        
        # Time parameters
        self.t_obs = T_OBS
        self.dt = DT
        self.fs = 1.0 / DT
        self.df = 1.0 / self.t_obs
        
        # Data size
        self.n_samples = int(self.t_obs / self.dt)
        
        # FULL frequency array
        self.freqs = np.fft.rfftfreq(self.n_samples, self.dt)
        
        # Noise model for FULL array
        self.noise_model_full = noise_model(self.freqs)
        
        # PSD for whitening and SNR
        psd = self.noise_model_full.eval([SA, SI])
        self.psd = np.asarray(psd.get() if hasattr(psd, 'get') else psd).ravel()
        # Handle DC and any zeros
        self.psd[0] = np.inf
        self.psd[self.psd <= 0] = np.inf
        
        # STFT parameters
        self.nperseg = NPERSEG
        self.noverlap = self.nperseg - 512
        
        # STFT frequency array and band mask
        self.f_stft = np.fft.rfftfreq(self.nperseg, 1/self.fs)
        self.f_mask = (self.f_stft >= F_MIN) & (self.f_stft <= F_MAX)
        self.f_stft_band = self.f_stft[self.f_mask]
        
        # Create directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        print(f"LISA Dataset Generator initialized")
        print(f"  T_obs: {self.t_obs/86400:.1f} days")
        print(f"  Band: {F_MIN*1e3:.3f} - {F_MAX*1e3:.3f} mHz")
        print(f"  Full freq bins: {len(self.freqs)}")
        print(f"  STFT band bins: {len(self.f_stft_band)}")
        print(f"  Min SNR: {MIN_SNR}")
    
    def _gen_noise(self):
        """Generate noise for FULL frequency array."""
        noise, _ = self.noise_model_full.noise_gen([SA, SI])
        noise = np.asarray(noise.get() if hasattr(noise, 'get') else noise).ravel()
        noise[0] = 0  # Remove DC
        return noise
    
    def _compute_snr(self, signal):
        """Compute SNR of a signal in the band."""
        return compute_snr(self.freqs, signal, self.psd, fmin=F_MIN, fmax=F_MAX)
    
    def _gen_ucb(self, f0_mHz, log10A):
        """
        Generate one UCB source in FULL frequency array.
        
        Returns:
        --------
        ucb_full : array
            Signal in frequency domain
        f0 : float
            Central frequency in Hz
        snr : float
            Signal-to-noise ratio
        """
        f0 = f0_mHz * 1e-3
        hw = 5e-6
        
        f_lims = np.array([f0 - hw, f0 + hw])
        s0 = int(f_lims[0] / self.df)
        s1 = int(f_lims[1] / self.df)
        
        m = ucb(f_lims=f_lims, f_inds=[s0, s1+1], 
                gpu=self.use_gpu, tobs=self.t_obs, dt=self.dt)
        
        # Random parameters
        p = np.array([log10A, f0_mHz, -16.5, 
                      np.random.uniform(0, 2*np.pi),
                      np.random.uniform(-1, 1),
                      np.random.uniform(0, np.pi),
                      np.random.uniform(0, 2*np.pi),
                      np.random.uniform(-1, 1)])
        
        A, _, _ = m.eval(p.reshape(1, 8))
        A = np.asarray(A.get() if hasattr(A, 'get') else A).ravel()
        
        # Place in FULL array
        ucb_full = np.zeros(len(self.freqs), dtype=np.complex128)
        ucb_full[s0:s0+len(A)] = A
        ucb_full[0] = 0  # Remove DC
        
        # Compute SNR
        snr = self._compute_snr(ucb_full)
        
        return ucb_full, f0, snr
    
    def _gen_smbhb(self):
        """
        Generate one SMBHB source in FULL frequency array.
        
        Returns:
        --------
        signal : array
            Signal in frequency domain
        t_merge_hours : float
            Time to merger in hours
        snr : float
            Signal-to-noise ratio
        """
        # Random masses
        m1 = np.random.uniform(SMBHB_M1_MIN, SMBHB_M1_MAX)
        m2 = np.random.uniform(SMBHB_M2_MIN, min(m1, SMBHB_M2_MAX))
        
        # Derived
        eta = m1 * m2 / (m1 + m2)**2
        Mc = (m1 + m2) * eta**0.6
        q = m1 / m2
        
        # Distance
        dist = np.random.uniform(SMBHB_DIST_MIN, SMBHB_DIST_MAX)
        logD = np.log10(dist)
        
        # Time to merger
        t_merge_hours = np.random.uniform(SMBHB_T_MERGE_MIN, SMBHB_T_MERGE_MAX)
        t_ref = self.t_obs - t_merge_hours * 3600
        t_ref_hr = t_ref / 3600
        
        # Random orientation
        ci = np.random.uniform(-1, 1)
        sb = np.random.uniform(-1, 1)
        lam = np.random.uniform(0, 2*np.pi)
        psi = np.random.uniform(0, np.pi)
        phi0 = np.random.uniform(0, 2*np.pi)
        c1, c2 = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
        
        p0 = np.array([[Mc, q, c1, c2, logD, t_ref_hr, ci, sb, lam, psi, phi0]])
        
        # Generate SMBHB
        modes = [(2, 2)]
        template = smbhb(f=self.freqs, modes=modes, tstart=self.t_obs, tend=0.0, 
                        gpu=self.use_gpu)
        As, _, _ = template.eval(p0)
        As = np.asarray(As.get() if hasattr(As, 'get') else As).ravel()
        As[0] = 0  # Remove DC
        
        # Compute SNR
        snr = self._compute_snr(As)
        
        return As, t_merge_hours, snr
    
    def _to_image(self, data):
        """
        Convert FULL frequency data to spectrogram image.
        
        Correct approach:
        1. iFFT to time domain
        2. Whiten time-domain data
        3. STFT on whitened data
        4. Crop to band
        5. Normalize and convert to image
        """
        # Safety: remove any NaN/inf
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data[0] = 0  # Ensure DC is zero
        
        # iFFT to time domain
        time_data = np.fft.irfft(data, n=self.n_samples)
        
        # Whiten in time domain (CORRECT approach)
        time_data_white = whiten(time_data, self.dt, self.psd)
        
        # Apply taper
        taper = tukey(len(time_data_white), alpha=0.1)
        time_data_white = time_data_white * taper
        
        # STFT on whitened data
        f_stft, t_stft, Zxx = stft(time_data_white, fs=self.fs, 
                                   nperseg=self.nperseg, 
                                   noverlap=self.noverlap)
        
        # Crop to band
        Zxx_band = Zxx[self.f_mask, :]
        
        # Magnitude (not power, since already whitened)
        S = np.abs(Zxx_band)
        
        # Log scaling for better dynamic range (shows both strong and weak signals)
        S_log = np.log10(S + 1e-10)
        
        # Normalize to [0, 1]
        vmin = np.percentile(S_log, 1)
        vmax = np.percentile(S_log, 99)
        S_norm = np.clip((S_log - vmin) / (vmax - vmin + 1e-12), 0, 1)
        
        # Flip so low frequency is at BOTTOM
        S_norm = np.flipud(S_norm)
        
        # To image
        img_gray = Image.fromarray((S_norm * 255).astype(np.uint8), 'L')
        img_gray = img_gray.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
        
        # Viridis colormap
        cmap = plt.get_cmap('viridis')
        img_rgb = (cmap(np.array(img_gray) / 255.0)[..., :3] * 255).astype(np.uint8)
        
        return img_rgb
    
    def _ucb_bbox(self, f0_hz):
        """UCB bounding box: horizontal, full width."""
        freq_norm = (f0_hz - F_MIN) / (F_MAX - F_MIN)
        y_center = 1.0 - freq_norm  # flip
        
        return (CLS_UCB, 0.5, np.clip(y_center, 0.03, 0.97), 1.0, 0.05)
    
    def _smbhb_bbox(self, t_merge_hours):
        """SMBHB bounding box: vertical, full height."""
        x_center = 1.0 - (t_merge_hours * 3600) / self.t_obs
        x_center = np.clip(x_center, 0.02, 0.98)
        
        return (CLS_SMBHB, x_center, 0.5, 0.02, 1.0)
    
    def generate(self, 
                 n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                 train_ucb_range=(TRAIN_UCB_MIN, TRAIN_UCB_MAX),
                 val_ucb_range=(VAL_UCB_MIN, VAL_UCB_MAX),
                 train_smbhb_prob=TRAIN_SMBHB_PROB,
                 val_smbhb_prob=VAL_SMBHB_PROB,
                 min_snr=MIN_SNR, seed=None):
        """Generate the dataset with SNR filtering."""
        np.random.seed(SEED if seed is None else seed)
        
        # UCB frequency range (in mHz, with margin)
        f_lo = UCB_F_MIN * 1e3
        f_hi = UCB_F_MAX * 1e3
        
        # Minimum separation
        min_sep_mHz = (F_MAX - F_MIN) * 1e3 * 0.06
        
        splits = {
            'train': (n_train, train_ucb_range, train_smbhb_prob),
            'val': (n_val, val_ucb_range, val_smbhb_prob),
            'test': (n_test, val_ucb_range, val_smbhb_prob),
        }
        
        for split, (n_images, ucb_range, smbhb_prob) in splits.items():
            ucb_min, ucb_max = ucb_range
            print(f"\n{split.upper()}: {n_images} images, {ucb_min}-{ucb_max} UCBs, {smbhb_prob:.0%} SMBHB, SNR>{min_snr}")
            
            for idx in tqdm(range(n_images), desc=split):
                # Start with noise (FULL array)
                data = self._gen_noise()
                
                bboxes = []
                ucb_freqs = []
                
                # Number of UCBs
                n_ucb = np.random.randint(ucb_min, ucb_max + 1)
                
                # Generate UCBs with SNR filtering
                attempts = 0
                max_attempts = n_ucb * 50
                while len(ucb_freqs) < n_ucb and attempts < max_attempts:
                    f0_mHz = np.random.uniform(f_lo, f_hi)
                    
                    # Check separation
                    too_close = any(abs(f0_mHz - f) < min_sep_mHz for f in ucb_freqs)
                    
                    if not too_close:
                        amp = np.random.uniform(UCB_LOG10A_MIN, UCB_LOG10A_MAX)
                        ucb_signal, f0_hz, snr = self._gen_ucb(f0_mHz, amp)
                        
                        # SNR filter
                        if snr >= min_snr:
                            data += ucb_signal
                            ucb_freqs.append(f0_mHz)
                            bboxes.append(self._ucb_bbox(f0_hz))
                    
                    attempts += 1
                
                # Add SMBHB with probability and SNR filtering
                if np.random.random() < smbhb_prob:
                    smbhb_attempts = 0
                    while smbhb_attempts < 20:
                        smbhb_signal, t_merge_hours, snr = self._gen_smbhb()
                        
                        if snr >= min_snr:
                            data += smbhb_signal
                            bboxes.append(self._smbhb_bbox(t_merge_hours))
                            break
                        
                        smbhb_attempts += 1
                
                # Generate image
                img = self._to_image(data)
                
                # Save image
                Image.fromarray(img).save(
                    self.output_dir / 'images' / split / f'{idx:06d}.png'
                )
                
                # Save labels
                with open(self.output_dir / 'labels' / split / f'{idx:06d}.txt', 'w') as f:
                    for b in bboxes:
                        f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
        
        # Write YAML
        self._write_yaml()
        print(f"\ndone: {self.output_dir}")
    
    def _write_yaml(self):
        """Write dataset YAML."""
        yaml_content = f"""path: {self.output_dir.absolute()}
train: images/train
val: images/val
test: images/test

names:
  0: ucb
  1: smbhb

nc: 2
"""
        (self.output_dir / 'dataset.yaml').write_text(yaml_content)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-train', type=int, default=N_TRAIN)
    parser.add_argument('--n-val', type=int, default=N_VAL)
    parser.add_argument('--n-test', type=int, default=N_TEST)
    parser.add_argument('--output', type=str, default=DATA_DIR)
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--min-snr', type=float, default=MIN_SNR)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    
    gen = LISADatasetGenerator(output_dir=args.output, use_gpu=not args.no_gpu)
    gen.generate(n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
                 min_snr=args.min_snr, seed=args.seed)
