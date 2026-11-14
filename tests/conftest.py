"""Shared fixtures. GPU tests are skipped when CUDA or the waveform packages are absent."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _has_gpu_stack():
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    return all(importlib.util.find_spec(m) is not None for m in ('cupy', 'bbhx', 'gbgpu'))


def _has_weights():
    from heron.artifacts import ART
    return all(p.exists() for p in ART.values())


requires_gpu = pytest.mark.skipif(not _has_gpu_stack(),
                                  reason="needs CUDA + cupy + bbhx + gbgpu")
requires_weights = pytest.mark.skipif(not _has_weights(), reason="model weights not installed")
