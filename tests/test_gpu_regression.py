"""GPU regression tests against stored values of the published runs:
waveforms, training-data generation and the stage 1-3 hand-off."""
import json

import numpy as np
import pytest

from heron.artifacts import PAPER_PRODUCTS
from tests.conftest import requires_gpu, requires_weights

REF = PAPER_PRODUCTS.parent / 'tests' / 'reference'


@requires_gpu
def test_waveform_fingerprint():
    """BBHx and GBGPU builds reproduce the published waveforms."""
    from tests.reference_values import REFERENCE, waveform_fingerprint
    fp = waveform_fingerprint()
    for k, ref in REFERENCE.items():
        assert np.allclose(fp[k], ref, rtol=1e-9), k


@requires_gpu
def test_mbhb_training_data_is_bit_reproducible(tmp_path):
    """Seed 90000 with prior-drawn reference spins regenerates the first
    samples of the published 10^7-sample training set exactly."""
    import pe_smbhb.data_mbhb as dm
    dm.WIDE_SPIN_REF = True
    gen = dm.HetDataGenerator(output_dir=tmp_path)
    gen.generate(64, 0, 0, seed=90000)
    ref = np.load(REF / 'mbhb_train_seed90000_first4.npz')
    for i in range(4):
        assert np.array_equal(np.load(tmp_path / 'train' / f'{i:06d}_het.npy'), ref[f'feats{i}'])
    meta = json.loads((tmp_path / 'train' / 'metadata.json').read_text())
    assert np.allclose([meta[i]['smbhb']['Mc'] for i in range(4)], ref['Mc'], rtol=0, atol=0)


@requires_gpu
def test_ucb_training_data_is_bit_reproducible(tmp_path):
    from pe_ucb.data_ucb import UCBPEGenerator
    gen = UCBPEGenerator(output_dir=tmp_path)
    gen.generate(32, 0, 0, seed=42)
    ref = np.load(REF / 'ucb_train_seed42_first8.npz')['feats']
    assert np.array_equal(np.load(tmp_path / 'train' / 'feats.npy')[:8], ref)


@requires_gpu
@requires_weights
def test_handoff_reproduces_published(tmp_path):
    """Stages 1-3 on the Sangria Case-1 event, default seed: the simulated data,
    the subtraction and the alert posterior match the published hand-off."""
    from heron.stages import run_stages_1_to_3
    r = run_stages_1_to_3(out=tmp_path, tag='t', verbose=False)
    assert abs(r['summary']['mbhb_snr'] - 357.04) < 0.01
    assert r['summary']['n_ucb_subtracted'] == 30
    for ds in ('clean', 'foreground', 'subtracted'):
        new = np.load(tmp_path / f't_{ds}.npz', allow_pickle=True)
        old = np.load(PAPER_PRODUCTS / 'handoff' / f'sangria_synergy_{ds}.npz', allow_pickle=True)
        assert sorted(new.files) == sorted(old.files)
        for k in ('dA', 'dE', 'psdA', 'psdE', 'hA_inj', 'hE_inj', 'freqs'):
            scale = np.max(np.abs(old[k]))
            diff = np.abs(new[k] - old[k])
            if ds == 'subtracted' and k in ('dA', 'dE'):
                # The frequency scan of the subtraction takes an argmin over a
                # grid. GPU round-off can switch it between two nearly tied
                # grid points for one source, which changes that source's
                # 128-bin window (1.2% of the band) and nothing else.
                assert np.mean(diff > 1e-6 * scale) < 0.03, (ds, k)
            else:
                assert diff.max() <= 1e-6 * scale, (ds, k)
        # flow draws: the same samples up to float32 round-off of the GPU sampler
        d = np.abs(new['alert_samples'] - old['alert_samples']) / old['alert_samples'].std(axis=0)
        assert np.median(d) < 1e-4 and np.percentile(d, 99.9) < 1e-2
        assert new['injection'].item() == old['injection'].item()
    assert abs(float(np.load(tmp_path / 't_subtracted.npz')['global_alpha']) - 0.505) < 2e-2
