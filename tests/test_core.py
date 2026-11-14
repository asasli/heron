"""CPU unit tests of the pieces that do not need the waveform packages."""
import json

import numpy as np
import pytest

from heron.artifacts import ART, WEIGHTS, load_kernel, sha256
from heron.skyarea import credible_areas


def test_weights_checksums():
    sums = WEIGHTS / 'SHA256SUMS'
    if not sums.exists():
        pytest.skip("weights not installed")
    for line in sums.read_text().split('\n'):
        if line.strip():
            digest, name = line.split()
            assert sha256(WEIGHTS / name) == digest, name


def test_kernel_released_equals_code_constant():
    if not ART['mbhb_info'].exists():
        pytest.skip("weights not installed")
    released = json.loads(ART['mbhb_info'].read_text())['kernel']['base']
    assert load_kernel() == released
    assert released == dict(frac_Mc=0.003, log_q=0.05, chi=0.05, t_floor_sec=120.0,
                            t_frac_of_tpre=0.02)


def test_sky_area_of_a_gaussian_blob():
    """A 2-degree Gaussian on the equator: 90% area = 2 pi sigma^2 ln(10) = 57.9 deg^2
    (the one-cell smoothing adds a few per cent)."""
    rng = np.random.default_rng(0)
    s = np.radians(2.0)
    lam = np.pi + rng.normal(0, s, 400000); sb = np.sin(rng.normal(0, s, 400000))
    a90, a50 = credible_areas(lam, sb)
    assert abs(a90 / (2 * np.pi * 2.0 ** 2 * np.log(10)) - 1) < 0.15
    assert a50 < a90


def test_recalibration_is_order_preserving():
    torch = pytest.importorskip('torch')                     # train_flow imports torch
    pytest.importorskip('zuko')
    from pe_smbhb.train_flow import recalibrate_samples
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5000, 2))
    recal = {'a': np.sort(rng.beta(2.0, 2.5, 3000))}
    y = recalibrate_samples(x, recal, ['a', 'b'])
    assert np.all(np.diff(y[np.argsort(x[:, 0], kind='stable'), 0]) >= 0)   # monotone map
    assert np.array_equal(x[:, 1], y[:, 1])                           # unmapped column untouched
    ident = {'a': (np.arange(3000) + 0.5) / 3000}
    z = recalibrate_samples(x, ident, ['a', 'b'])[:, 0]                # identity map: no change
    assert np.median(np.abs(z - x[:, 0])) < 5e-3
