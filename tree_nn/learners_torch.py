"""PyTorch node learner: every tree node is a small MLP (or a linear layer).

Minimum-width search: widths are tried in ascending order (default 0 = linear,
16, 64, 256) and the FIRST width that interpolates the node's masked training set
is kept.  Nodes are therefore as small as they can be, which is our first lever
for "minimise the size of the network".  If no width interpolates, the best is
kept and the tree grows grand-children to repair it.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


_CACHE = {"ref": None, "dev": None, "t": None}


def dev_tensor(F, device):
    """float32 copy of F on `device`, cached for the most recent array.  The cache keeps a reference to
    F, so `is` comparison cannot be fooled by a recycled id().  F must not be modified in place."""
    if _CACHE["ref"] is not F or _CACHE["dev"] != device:
        _CACHE.update(ref=F, dev=device, t=torch.as_tensor(F, dtype=torch.float32, device=device))
    return _CACHE["t"]


def make_net(d, width, n_out):
    if width == 0:
        return nn.Linear(d, n_out)
    return nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, n_out))


class TorchFitted:
    def __init__(self, net, width, n_out, device):
        self.net, self.width, self.n_out, self.device = net.eval(), width, n_out, device

    @torch.no_grad()
    def logits(self, F, bs=16384):
        self.net.to(self.device).eval()
        out = []
        Fd = dev_tensor(F, self.device)
        for i in range(0, len(F), bs):
            out.append(self.net(Fd[i:i + bs]).double().cpu())
        z = torch.cat(out).numpy()
        return z[:, 0] if self.n_out == 1 else z

    @property
    def n_params(self):
        return int(sum((p != 0).sum().item() for p in self.net.parameters()))

    @torch.no_grad()
    def sparsify(self, frac):
        """Zero the `frac` smallest-magnitude weights of this node (global within node)."""
        ws = [p for p in self.net.parameters() if p.dim() > 1]
        if not ws or frac <= 0:
            return
        allw = torch.cat([w.abs().flatten() for w in ws])
        k = int(frac * allw.numel())
        if k < 1:
            return
        thr = torch.kthvalue(allw.cpu(), k).values.item()
        for w in ws:
            w[w.abs() <= thr] = 0.0

    def __getstate__(self):                       # keep pickles device-agnostic
        s = dict(self.__dict__)
        s["net"] = self.net.cpu()
        return s


class TorchLearner:
    def __init__(self, device="cuda", widths=(0, 16, 64, 256), steps=2000, lr=5e-3,
                 weight_decay=0.0, check_every=50, seed=0, max_pos_weight=None):
        self.device, self.widths, self.steps, self.lr = device, tuple(widths), steps, lr
        self.wd, self.check_every, self.seed = weight_decay, check_every, seed
        self.max_pos_weight = max_pos_weight
        self.n_fits = 0

    def _train(self, X, y, kind, n_out, width):
        torch.manual_seed(self.seed + self.n_fits)
        net = make_net(X.shape[1], width, n_out).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        if kind == "binary":
            yf = y.float()
            npos = yf.sum().clamp(min=1.0)
            pw = (len(yf) - npos) / npos
            if self.max_pos_weight:
                pw = pw.clamp(max=self.max_pos_weight)
        acc = 0.0
        for step in range(self.steps):
            z = net(X)
            if kind == "multi":
                loss = Fn.cross_entropy(z, y)
            else:
                loss = Fn.binary_cross_entropy_with_logits(z[:, 0], yf, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (step + 1) % self.check_every == 0:
                with torch.no_grad():
                    z = net(X)
                    pred = z.argmax(1) if kind == "multi" else (z[:, 0] > 0).long()
                    acc = (pred == y).float().mean().item()
                if acc >= 1.0:
                    break
        return net, acc

    def fit(self, F, t, mask, kind, n_out):
        idx = np.flatnonzero(mask)
        X = dev_tensor(F, self.device)[torch.as_tensor(idx, device=self.device)]
        y = torch.as_tensor(np.asarray(t)[idx], dtype=torch.long, device=self.device)
        if kind == "binary" and (y.min() == y.max()):
            # nothing to separate: constant output with the right sign
            net = nn.Linear(X.shape[1], 1)
            nn.init.zeros_(net.weight)
            nn.init.constant_(net.bias, 5.0 if int(y[0]) == 1 else -5.0)
            self.n_fits += 1
            return TorchFitted(net, 0, 1, self.device)
        best, best_acc, best_w = None, -1.0, 0
        for w in self.widths:
            net, acc = self._train(X, y, kind, n_out, w)
            self.n_fits += 1
            if acc > best_acc:
                best, best_acc, best_w = net, acc, w
            if acc >= 1.0:
                break
        return TorchFitted(best, best_w, n_out, self.device)
