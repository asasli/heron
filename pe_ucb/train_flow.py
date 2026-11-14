"""
Amortized per-source UCB PE — the galactic half of HERON and the subtraction
engine for the pointing tier.

Input: (W=128, 4) whitened A/E window around the YOLO-detected frequency.
Output: 7-param posterior [log10A, df_bins, phi0, cos_iota, psi, lambda, sin_beta].
On-the-fly per batch: instrument noise, confusion floor (per-sample Sc/Sn ratio
x amplitude ladder), and NEIGHBOR overlap (other batch samples, shifted/scaled)
so the model is robust to unresolved/imperfectly-detected companions.

Run: python -m pe_ucb.train_flow \
        --data data/ucb --tag ucb
"""
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import UCB_DATA

RANGES = {
    'log10A': (-22.0, -20.3), 'df_bins': (-9.0, 9.0), 'phi0': (0.0, 2*np.pi),
    'cos_iota': (-1.0, 1.0), 'psi': (0.0, np.pi), 'lambda': (0.0, 2*np.pi),
    'sin_beta': (-1.0, 1.0),
}
PARAMS = list(RANGES)
N_PAR = len(PARAMS)


def norm_targets(md):
    out = np.empty((len(md), N_PAR), dtype=np.float32)
    for j, p in enumerate(PARAMS):
        lo, hi = RANGES[p]
        out[:, j] = np.clip([(m[p] - lo) / (hi - lo) for m in md], 1e-4, 1 - 1e-4)
    return out


class UCBFlow(nn.Module):
    def __init__(self, W=128, context=128, transforms=8, hidden=128, bins=8):
        super().__init__()
        from zuko.flows import NSF
        self.enc = nn.Sequential(
            nn.Conv1d(4, 32, 5, stride=1, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * (W // 8), 256), nn.GELU(),
            nn.Linear(256, context), nn.LayerNorm(context),
        )
        self.flow = NSF(features=N_PAR, context=context, transforms=transforms,
                        hidden_features=[hidden, hidden], bins=bins)

    def log_prob(self, x, theta):
        return self.flow(self.enc(x.transpose(1, 2))).log_prob(theta)

    def sample(self, x, n):
        ctx = self.enc(x.transpose(1, 2))
        ctx = ctx.repeat_interleave(n, dim=0)
        return self.flow(ctx).sample().reshape(x.shape[0], n, N_PAR)


def augment_(fd, conf_ratio, sigma0, rng=None, p_neigh=0.5):
    """In-place batch augmentation: noise + confusion + neighbor overlap.
    fd: (B, W, 4) whitened signal windows (raw scale)."""
    B = fd.shape[0]
    dev = fd.device
    g = rng
    randn = (lambda *s: torch.randn(*s, device=dev)) if g is None else \
            (lambda *s: torch.randn(*s, generator=g, device=dev))
    rand = (lambda *s: torch.rand(*s, device=dev)) if g is None else \
           (lambda *s: torch.rand(*s, generator=g, device=dev))
    # neighbor overlap: add shifted/scaled copies of OTHER samples
    m = rand(B) < p_neigh
    if m.any():
        idx = torch.randperm(B, device=dev) if g is None else \
            torch.randperm(B, generator=g, device=dev)
        shifts = (rand(B) * 96 - 48).long()
        amps = 10 ** (rand(B) - 0.5)
        nb = torch.stack([torch.roll(fd[idx[b]], int(shifts[b]), dims=0)
                          for b in range(B)])
        fd[m] += (amps[m, None, None] * nb[m])
    # confusion floor + instrument noise
    conf_amp = 10 ** (rand(B) * (np.log10(2.0) - np.log10(0.5)) + np.log10(0.5))
    sig = sigma0 * torch.sqrt(1.0 + conf_ratio * conf_amp**2)
    fd += randn(*fd.shape) * sig[:, None, None]
    return fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=UCB_DATA)
    ap.add_argument('--tag', default='ucb')
    ap.add_argument('--save-dir', default='runs/train/ucb')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    root = Path(args.data)
    info = json.loads((root / 'dataset_info.json').read_text())
    sigma0 = float(info['sigma0'])

    def load(split):
        feats = torch.from_numpy(np.load(root / split / 'feats.npy'))
        md = json.loads((root / split / 'metadata.json').read_text())
        tg = torch.from_numpy(norm_targets(md))
        cr = torch.tensor([m['conf_ratio'] for m in md], dtype=torch.float32)
        return feats, tg, cr

    trX, trY, trC = load('train')
    vaX, vaY, vaC = load('val')
    print(f"train {len(trX)}, val {len(vaX)}; sigma0={sigma0:.1f}")

    model = UCBFlow(W=info['W']).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"UCBFlow: {n_par:,} params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # fixed val noise
    gval = torch.Generator(device=device).manual_seed(7)
    vaXd = vaX.to(device).clone()
    vaXd = augment_(vaXd, vaC.to(device), sigma0, rng=gval) / sigma0
    vaYd = vaY.to(device)

    best = float('inf')
    save = Path(args.save_dir); save.mkdir(parents=True, exist_ok=True)
    B = args.batch_size
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(trX))
        tot, nb = 0.0, 0
        for a0 in range(0, len(trX) - B + 1, B):
            sel = perm[a0:a0 + B]
            fd = trX[sel].to(device, non_blocking=True).clone()
            fd = augment_(fd, trC[sel].to(device), sigma0) / sigma0
            tg = trY[sel].to(device)
            loss = -model.log_prob(fd, tg).clamp(-50, 50).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vls = []
            for a0 in range(0, len(vaXd), 4096):
                lp = model.log_prob(vaXd[a0:a0+4096], vaYd[a0:a0+4096]).clamp(-50, 50)
                vls.append(-lp.mean().item())
            vl = float(np.mean(vls))
        star = ''
        if vl < best:
            best = vl; star = ' *'
            torch.save({'model_state_dict': model.state_dict(), 'epoch': ep,
                        'val_nll': vl, 'config': {'W': info['W'], 'ranges': RANGES,
                        'sigma0': sigma0, 'params': PARAMS}},
                       save / f'ucb_flow_{args.tag}_best.pt')
        print(f"E{ep:3d} train={tot/nb:.3f} val={vl:.3f} best={best:.3f}"
              f" lr={opt.param_groups[0]['lr']:.1e}{star}", flush=True)
    print(f"Done. best val NLL {best:.3f}")


if __name__ == '__main__':
    main()
