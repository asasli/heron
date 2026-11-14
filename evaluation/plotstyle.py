"""Shared publication-quality matplotlib style for HERON figures.
Falls back gracefully if LaTeX is unavailable."""
import shutil
import matplotlib as mpl


def paper_style():
    """Apply a clean Nature-style rcParams. Uses system LaTeX if present."""
    usetex = shutil.which('latex') is not None
    mpl.rcParams.update({
        'text.usetex': usetex,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
        'mathtext.fontset': 'cm',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'axes.linewidth': 0.9,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'axes.grid': False,
        'savefig.dpi': 300, 'figure.dpi': 120,
    })
    if usetex:
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{bm}'
    return usetex


# HERON house colours
HERON_RED = '#c1121f'    # HERON posterior
MCMC_BLUE = '#1d4e89'    # exact MCMC reference
TRUTH_K = '#111111'      # truth marker

CORNFLOWER = '#6495ED'   # MCMC (requested)
HERON_CORAL = '#E63946'  # HERON flow — watermelon red, pairs with cornflower


def cc_corner(mcmc, ours, labels, truth, save, title=None,
              mcmc_name='MCMC (exact)', ours_name='HERON', figsize=8.0):
    """Publication corner overlay with ChainConsumer: cornflower-blue MCMC vs
    coral HERON, truth markers. Returns the saved path."""
    import numpy as np
    import matplotlib.pyplot as plt
    from chainconsumer import ChainConsumer
    c = ChainConsumer()
    c.add_chain(np.asarray(mcmc), parameters=list(labels), name=mcmc_name)
    c.add_chain(np.asarray(ours), parameters=list(labels), name=ours_name)
    c.configure(colors=[CORNFLOWER, HERON_CORAL],
                linewidths=[1.3, 1.3], shade=[True, True], shade_alpha=[0.55, 0.30],
                sigmas=[1, 2], kde=False, smooth=1, bar_shade=False, serif=True,
                statistics='mean',   # robust to edge-piled/degenerate 1D marginals
                legend_kwargs={'fontsize': 15, 'loc': 'upper right'},
                label_font_size=15, tick_font_size=10, max_ticks=4, summary=False,
                diagonal_tick_labels=False, spacing=0.0)
    fig = c.plotter.plot(figsize=(figsize, figsize),
                         truth={p: t for p, t in zip(labels, truth)})
    if title:
        fig.suptitle(title, fontsize=17, y=1.02)
    fig.savefig(save, dpi=200, bbox_inches='tight')
    pdf = save.rsplit('.', 1)[0] + '.pdf'
    try:
        fig.savefig(pdf, bbox_inches='tight')
    except Exception:
        pdf = None
    plt.close(fig)
    return save, pdf
