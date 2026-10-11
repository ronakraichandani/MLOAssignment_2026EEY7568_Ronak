"""Receptive-field analysis of tree nodes.

For a node n the scalar  s_n(x) = node_net( standardise( trunk(x) ) )  (the node's own
logit; for the root one logit per class) is differentiable all the way to the input
pixels, because the trunk is frozen and shared.  We visualise it three ways:

1. Gradient receptive field: SmoothGrad |d s_n / d x| averaged over colour channels and over
   the node's *positive* training samples (the samples it was created to repair).
2. Activation maximisation: a synthetic image that maximises s_n (with total-variation and
   random-shift regularisation) -- what pattern does the node "look for"?
3. Top activating real images among its positives.

Summary statistics of the gradient map (how localised / how off-centre it is) are computed
per depth so that root nodes can be compared with deep repair nodes.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .features import to_tensor


def node_scorer(trunk, node, mu, sd, device, cls=None):
    mu_t = torch.as_tensor(mu, device=device)
    sd_t = torch.as_tensor(sd, device=device)
    net = node.fitted.net.to(device).eval()

    def f(x):
        z = net((trunk(x) - mu_t) / sd_t)
        if z.shape[1] > 1:
            return z[:, cls]
        return z[:, 0]
    return f


def smooth_saliency(f, x, n=8, sigma=0.08):
    """mean over images of SmoothGrad |grad| (channel-mean) -> (H, W); also per-image maps."""
    g = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device)
    for _ in range(n):
        xn = (x + sigma * torch.randn_like(x)).clamp(0, 1).requires_grad_(True)
        s = f(xn).sum()
        grad, = torch.autograd.grad(s, xn)
        g += grad.abs().mean(1)
    g /= n
    return g


def act_max(f, shape, steps=250, lr=0.05, tv=2e-3, seed=0, device="cuda"):
    torch.manual_seed(seed)
    x = (0.5 + 0.05 * torch.randn(1, *shape, device=device)).clamp(0, 1).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        dx, dy = np.random.randint(-2, 3, 2)
        xs = torch.roll(x, (int(dx), int(dy)), (2, 3))
        tvl = (xs[:, :, 1:] - xs[:, :, :-1]).abs().mean() + (xs[:, :, :, 1:] - xs[:, :, :, :-1]).abs().mean()
        loss = -f(xs).sum() + tv * tvl * 100
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    return x.detach()[0]


def map_stats(m):
    """m: (H,W) non-negative saliency map -> concentration & centre-of-mass statistics."""
    m = m / (m.sum() + 1e-12)
    flat = np.sort(m.flatten())[::-1]
    n50 = int(np.searchsorted(np.cumsum(flat), 0.5)) + 1
    H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (m * yy).sum(), (m * xx).sum()
    off = float(np.hypot(cy - (H - 1) / 2, cx - (W - 1) / 2) / (np.hypot(H, W) / 2))
    ent = float(-(m * np.log(m + 1e-12)).sum() / np.log(m.size))
    return dict(frac_pixels_for_50pct=n50 / m.size, centre_offset=off, norm_entropy=ent)


def select_nodes(tree, n_children=10, max_depth=3):
    nodes = tree.nodes()
    sel = [(tree.root, k) for k in range(tree.n_classes)]
    by_depth = {}
    for n in nodes:
        if n.role != "root" and n.depth <= max_depth and n.pos_idx is not None and len(n.pos_idx) >= 3:
            by_depth.setdefault(n.depth, []).append(n)
    for d in sorted(by_depth):
        by_depth[d].sort(key=lambda n: -n.n_pos)
        sel += [(n, None) for n in by_depth[d][:n_children]]
    return sel


def visualise(tree, trunk, Ftr, Xtr, ytr, mu, sd, classes, out_dir, device="cuda", n_children=8,
              max_depth=3, n_imgs=6, max_sal_imgs=64):
    os.makedirs(out_dir, exist_ok=True)
    tree.propagate_cls()
    rows, stats = [], []
    for node, k in select_nodes(tree, n_children, max_depth):
        if node.role == "root":
            idx = np.flatnonzero(ytr == k)
            title = f"root / class '{classes[k]}'"
            f = node_scorer(trunk, node, mu, sd, device, cls=k)
            score = node.fitted.logits(Ftr[idx])[:, k]
        else:
            idx = node.pos_idx
            if node.depth == 1:
                title = f"d1 A-child (repairs '{classes[node.cls]}', {node.n_pos} pos.)"
            else:
                title = (f"d{node.depth} {node.role}-child in the repair chain of '{classes[node.cls]}' "
                         f"({node.n_pos} pos.)")
            f = node_scorer(trunk, node, mu, sd, device)
            score = node.fitted.logits(Ftr[idx])
        order = np.argsort(-score)
        top = idx[order[:n_imgs]]
        sal_idx = idx[np.random.RandomState(0).permutation(len(idx))[:max_sal_imgs]]
        x = to_tensor(Xtr[sal_idx], device)
        sal = smooth_saliency(f, x).mean(0).cpu().numpy()
        am = act_max(f, x.shape[1:], device=device).cpu().permute(1, 2, 0).numpy()
        st = map_stats(sal)
        st.update(node_id=node.id, depth=node.depth, role=node.role, n_pos=int(node.n_pos),
                  width=int(node.fitted.width), title=title)
        stats.append(st)
        rows.append((title, Xtr[top], sal, am, st))

    per_page = 8
    paths = []
    for p in range(0, len(rows), per_page):
        chunk = rows[p:p + per_page]
        fig, axs = plt.subplots(len(chunk), n_imgs + 2, figsize=(1.6 * (n_imgs + 2), 1.75 * len(chunk)),
                                squeeze=False)
        for r, (title, tops, sal, am, st) in enumerate(chunk):
            for c in range(n_imgs):
                axs[r, c].axis("off")
                if c < len(tops):
                    axs[r, c].imshow(tops[c])
            axs[r, n_imgs].imshow(sal, cmap="inferno"); axs[r, n_imgs].axis("off")
            axs[r, n_imgs + 1].imshow(am); axs[r, n_imgs + 1].axis("off")
            axs[r, 0].set_title(title, fontsize=7, loc="left")
            if r == 0:
                axs[r, n_imgs].set_title("|grad| RF", fontsize=7)
                axs[r, n_imgs + 1].set_title("act. max", fontsize=7)
        plt.tight_layout()
        path = os.path.join(out_dir, f"rf_page{p // per_page + 1}.png")
        plt.savefig(path, dpi=150); plt.close(fig)
        paths.append(path)

    # summary by depth
    summary = {}
    for st in stats:
        key = "root" if st["role"] == "root" else f"depth{st['depth']}"
        summary.setdefault(key, []).append(st)
    agg = {k: dict(n=len(v), **{m: float(np.mean([s[m] for s in v]))
                                for m in ("frac_pixels_for_50pct", "centre_offset", "norm_entropy")})
           for k, v in summary.items()}
    json.dump(dict(per_node=stats, by_depth=agg, pages=paths), open(os.path.join(out_dir, "rf_summary.json"), "w"),
              indent=1)
    # simple bar plot of localisation vs depth
    keys = sorted(agg, key=lambda s: (s != "root", s))
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    ax[0].bar(keys, [agg[k]["frac_pixels_for_50pct"] for k in keys], color="#4C72B0")
    ax[0].set_ylabel("fraction of pixels holding\n50% of |grad| mass"); ax[0].set_title("RF spread (lower = more local)", fontsize=8)
    ax[1].bar(keys, [agg[k]["centre_offset"] for k in keys], color="#DD8452")
    ax[1].set_ylabel("RF centre offset"); ax[1].set_title("distance of RF centre from image centre", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "rf_vs_depth.png"), dpi=150); plt.close(fig)
    return agg
