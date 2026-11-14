"""
The canonical sky-area estimator of the paper (pointing tier).

Full-sky histogram on a 0.5 degree grid in ecliptic longitude and
(2/360)-wide cells in sin(latitude) (equal-area cells), Gaussian smoothing with
sigma = 1 cell (longitude wrapped), highest-posterior-density cell count, both
sky modes included. Every stage-4 area quoted in the paper uses these defaults;
a raw 1-degree histogram reads about 30% low and a 1.5-cell smoothing about
2.5 times high, so quote the estimator with the number.
"""
import numpy as np

NLAM, NSB, SMOOTH = 720, 360, 1.0


def credible_areas(lam_rad, sin_beta, levels=(0.90, 0.50)):
    """HPD credible sky areas [deg^2] at the given credibility levels."""
    from scipy.ndimage import gaussian_filter
    lam = np.degrees(np.asarray(lam_rad, float)) % 360.0
    sb = np.clip(np.asarray(sin_beta, float), -1, 1)
    H, xe, ye = np.histogram2d(lam, sb, bins=[NLAM, NSB], range=[[0., 360.], [-1., 1.]])
    Hs = gaussian_filter(H.T, SMOOTH, mode=('nearest', 'wrap'))
    flat = np.sort(Hs.ravel())[::-1]
    csum = np.cumsum(flat) / flat.sum()
    cell = np.radians(xe[1] - xe[0]) * (ye[1] - ye[0]) * (180 / np.pi) ** 2
    return [float(np.sum(Hs >= flat[np.searchsorted(csum, q)]) * cell) for q in levels]


def mode_fraction(lam_rad, sin_beta, lam_deg=(255., 345.), sin_beta_max=0.1):
    """Posterior mass inside the sky island that contains the (frame-mapped)
    Case-1 injection: 255 < lambda < 345 deg, sin(beta) < 0.1 in the stage-4
    frame (the antipodal island is the complement)."""
    lam = np.degrees(np.asarray(lam_rad, float)) % 360.0
    sb = np.asarray(sin_beta, float)
    return float(((lam > lam_deg[0]) & (lam < lam_deg[1]) & (sb < sin_beta_max)).mean())
