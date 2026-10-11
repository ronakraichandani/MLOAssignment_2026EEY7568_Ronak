"""Tree-wise double descent: sweep the (fixed) hidden width W of every node, grow each tree to
interpolation on noisy labels, and record test error against total parameters.  A single
wide MLP on the same features ("root only") is the classical model-wise reference.

python scripts/tree_sweep.py --dataset cifar10 --noise 0.15 --n_train 15000 --out results/sweep_cifar10
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tree_nn.data import get_dataset
from tree_nn.features import build_trunk, extract, standardise
from tree_nn.learners_torch import TorchLearner, make_net
from tree_nn.tree import grow_tree, GrowConfig
from tree_nn.utils import set_seed, get_device, ints, save_json, load_json


def mlp_baseline(Ftr, ytr, Fte, yte, K, width, dev, steps=3000, lr=2e-3):
    torch.manual_seed(0)
    X = torch.as_tensor(Ftr, device=dev); y = torch.as_tensor(ytr, device=dev).long()
    Xt = torch.as_tensor(Fte, device=dev); yt = torch.as_tensor(yte, device=dev).long()
    net = make_net(X.shape[1], width, K).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for s in range(steps):
        loss = Fn.cross_entropy(net(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        tr = (net(X).argmax(1) == y).float().mean().item()
        te = (net(Xt).argmax(1) == yt).float().mean().item()
    return dict(width=width, params=sum(p.numel() for p in net.parameters()), train_acc=tr, test_acc=te)


def plot(res, out):
    t, b = res["tree"], res["mlp"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([r["params"] for r in b], [1 - r["test_acc"] for r in b], "s-", label="single MLP (root only)")
    ax[0].plot([r["params"] for r in t], [1 - r["test_acc"] for r in t], "o-", label="tree of MLPs (interpolating)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("total parameters"); ax[0].set_ylabel("test error")
    ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("test error vs size")
    ax[1].plot([r["width"] for r in t], [r["n_nodes"] for r in t], "o-", color="C2")
    ax[1].set_xscale("symlog", linthresh=1); ax[1].set_yscale("log")
    ax[1].set_xlabel("node width W (0 = linear)"); ax[1].set_ylabel("number of nodes"); ax[1].grid(alpha=.3)
    ax[1].set_title("wider nodes -> fewer nodes")
    plt.tight_layout(); plt.savefig(os.path.join(out, "tree_sweep.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "catsdogs"])
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--n_train", type=int, default=15000)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--trunk", default="scratch_cnn")
    ap.add_argument("--trunk_epochs", type=int, default=3)
    ap.add_argument("--widths", default="0,1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--max_depth", type=int, default=12)
    ap.add_argument("--max_nodes", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)
    dev = get_device()
    D = get_dataset(a.dataset, a.data_root, a.n_train, a.noise, a.seed, a.img_size)
    K = D["n_classes"]
    trunk = build_trunk(a.trunk, D["Xtr"], D["ytr"], K, dev, a.trunk_epochs, a.img_size, a.seed)
    Ftr, Fte, _, _ = standardise(extract(trunk, D["Xtr"], dev), extract(trunk, D["Xte"], dev))
    ytr, yte = D["ytr"], D["yte"]
    jpath = os.path.join(a.out, "sweep_results.json")
    res = load_json(jpath) if os.path.exists(jpath) else dict(args=vars(a), tree=[], mlp=[])
    done = {r["width"] for r in res["tree"]}
    for W in ints(a.widths):
        if W in done:
            continue
        t0 = time.time()
        tree = grow_tree(Ftr, ytr, K, TorchLearner(dev, (W,), a.steps, seed=a.seed),
                         GrowConfig(a.max_depth, 1, 0.05, a.max_nodes, False, verbose=False))
        r = dict(width=W, params=tree.n_params(), n_nodes=tree.n_nodes(), depth=tree.depth(),
                 train_acc=tree.accuracy(Ftr, ytr), test_acc=tree.accuracy(Fte, yte),
                 root_test_acc=tree.accuracy(Fte, yte, 0), seconds=time.time() - t0)
        print(r, flush=True)
        res["tree"].append(r)
        if W > 0:
            res["mlp"].append(mlp_baseline(Ftr, ytr, Fte, yte, K, W, dev))
        res["tree"].sort(key=lambda r: r["width"]); res["mlp"].sort(key=lambda r: r["width"])
        save_json(res, jpath)
        plot(res, a.out)
    print("DONE")


if __name__ == "__main__":
    main()
