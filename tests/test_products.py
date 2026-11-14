"""The numbers quoted in the paper, recomputed from the published products
(paper_products/). CPU only: no GPU, no weights except the calibration map."""
import numpy as np
import pytest
from scipy.stats import kstest

from heron.artifacts import ART, PAPER_PRODUCTS
from heron.skyarea import credible_areas, mode_fraction

P = PAPER_PRODUCTS
ELEVEN = ['Mc', 'q', 'c1', 'c2', 'logD', 'cos_inc', 'sin_beta', 'lambda', 'psi', 'phi0', 't_pre_hours']


def test_calibration_ks_range():
    """Fig. 3a: KS p-values of the 11 recalibrated marginals span 0.08-0.94."""
    z = np.load(P / 'pp_ranks_recal.npz')
    assert int(z['N']) == 500
    pv = [kstest(z[n], 'uniform').pvalue for n in ELEVEN]
    assert round(min(pv), 2) == 0.08 and round(max(pv), 2) == 0.94


def test_recalibration_map_refit_matches_released():
    """The released map is the empirical rank CDF of the 3000-event calibration set."""
    if not ART['mbhb_recal'].exists():
        pytest.skip("weights not installed")
    cal = np.load(P / 'pp_ranks_cal.npz'); rel = np.load(ART['mbhb_recal'])
    assert int(cal['N']) == 3000
    for p in rel['params']:
        assert np.array_equal(np.sort(cal[str(p)]), rel[f'g_{p}'])


def test_recalibrated_test_ranks():
    cal = np.load(P / 'pp_ranks_cal.npz'); test = np.load(P / 'pp_ranks.npz')
    rec = np.load(P / 'pp_ranks_recal.npz')
    for p in ELEVEN:
        g = np.sort(cal[p])
        assert np.allclose(np.searchsorted(g, test[p], side='right') / len(g), rec[p])


def test_sky_coverage():
    """Two-dimensional sky credible regions: 55.2% / 89.8% (main text)."""
    sc = np.load(P / 'pp_ranks_skycov.npz')['sky_cred']
    assert len(sc) == 500
    assert round(100 * (sc <= 0.5).mean(), 1) == 55.2
    assert round(100 * (sc <= 0.9).mean(), 1) == 89.8


@pytest.mark.parametrize('tag,a90,a50,fdom', [
    ('subtracted_alert', 143, 28, 0.81), ('foreground_alert', 134, 32, 0.53),
    ('clean_alert', 191, 41, 0.72), ('subtracted_none', 772, 109, 0.23)])
def test_pointing_single_realization(tag, a90, a50, fdom):
    """Fig. 5b: 90/50% areas and dominant-mode probability of the stage-4 chains."""
    s = np.load(P / 'pointing' / f'synergy_{tag}.npz', allow_pickle=True)['samples']
    A90, A50 = credible_areas(s[:, 8], s[:, 7])
    assert round(A90) == a90 and round(A50) == a50
    assert round(mode_fraction(s[:, 8], s[:, 7]), 2) == fdom


def test_pointing_ensemble_summary():
    """Fig. 5c: 180+-24 / 119+-94 / 132+-82 deg^2 over N = 4 noise realizations."""
    z = np.load(P / 'pointing_ensemble_canonical.npz')
    for ds, mean, std in [('clean', 180, 24), ('foreground', 119, 94), ('subtracted', 132, 82)]:
        a90 = z[f'hcl_{ds}'][:, 0]
        assert len(a90) == 4 and round(a90.mean()) == mean and round(a90.std()) == std


def test_resolved_population():
    """Methods: 307 resolved (SNR > 10) binaries in 0.1-6 mHz, 94 below 4 mHz."""
    z = np.load(P / 'sangria_1month_snr.npz')
    m = z['snr'] > 10
    assert int(m.sum()) == 307
    assert int((m & (z['f0'] > 1e-4) & (z['f0'] < 4e-3)).sum()) == 94
    h, _ = np.histogram(z['f0'][m], bins=np.arange(1e-4, 6e-3, 110e-6))
    assert h.max() == 20 and np.median(h[h > 0]) == 11
