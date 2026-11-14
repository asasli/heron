"""HERON: HEterodyned RObust Neural inference for LISA.

Four stages on one month of LISA data: spectrogram detection of the resolved
galactic binaries and of the massive black-hole binary (yolo/), an amortized
alert posterior conditioned on the detected foreground (pe_smbhb/), a
galactic-binary catalogue with closed-form subtraction (pe_ucb/), and
alert-seeded exact inference (pointing/).

    heron infer                  the four stages on one event, from the released weights
    heron train                  data, training of the three networks, calibration, validation
    heron check                  installation check
    heron --help                 every command

    from heron import Heron      the same from Python (heron/api.py)
"""
__version__ = "1.0.0"


def __getattr__(name):
    # lazy: importing `heron` must not fix the weights directory or load torch
    if name in ('Heron', 'Result'):
        from heron import api
        return getattr(api, name)
    raise AttributeError(f"module 'heron' has no attribute '{name}'")
