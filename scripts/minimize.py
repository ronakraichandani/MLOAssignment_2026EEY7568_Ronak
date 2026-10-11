"""Shrink a grown tree.  Three levers, all measured on train and test accuracy:

(a) minimum-width search   -- already built into growth (TorchLearner picks the smallest interpolating
                              width per node); reported here as a width histogram.
(b) child pruning          -- drop every child (with its subtree) that repairs fewer than m samples.
(c) weight sparsification  -- zero the smallest-magnitude fraction of each node's weights.
(b)+(c) are applied post hoc WITHOUT fine-tuning, so the 100%-train-accuracy guarantee is traded
for size; the curves show how gracefully.

python scripts/minimize.py --run results/cifar10_main
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tree_nn.tree import prune_min_pos, sparsify_tree
from tree_nn.utils import save_json, get_device


def ev(t, Ftr, ytr, Fte, yte):
    return dict(params=t.n_params(), nodes=t.n_nodes(), train_acc=t.accuracy(Ftr, ytr), test_acc=t.accuracy(Fte, yte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="directory written by run_tree.py")
    ap.add_argument("--min_pos", default="1,2,3,5,10,20,50,100,200,500")
    ap.add_argument("--sparsity", default="0,0.5,0.8,0.9,0.95,0.98")
    a = ap.parse_args()
    ck = torch.load(os.path.join(a.run, "tree.pt"), weights_only=False)
    tree = ck["tree"]
    d = np.load(os.path.join(a.run, "feats.npz"))
    Ftr, Fte = d["Ftr"].astype(np.float32), d["Fte"].astype(np.float32)
    ytr, yte = d["ytr"], d["yte"]

    full = ev(tree, Ftr, ytr, Fte, yte)
    print("full tree", full, flush=True)
    prune = []
    for m in [int(x) for x in a.min_pos.split(",")]:
        r = dict(min_pos=m, **ev(prune_min_pos(tree, m), Ftr, ytr, Fte, yte)); prune.append(r); print(r, flush=True)
    spars = []
    for s in [float(x) for x in a.sparsity.split(",")]:
        r = dict(sparsity=s, **ev(sparsify_tree(tree, s), Ftr, ytr, Fte, yte)); spars.append(r); print(r, flush=True)
    grid = []
    for m in (1, 5, 20, 100):
        pt = prune_min_pos(tree, m)
        for s in (0.0, 0.5, 0.9):
            r = dict(min_pos=m, sparsity=s, **ev(sparsify_tree(pt, s), Ftr, ytr, Fte, yte)); grid.append(r)
    widths = {}
    for n in tree.nodes():
        widths[int(n.fitted.width)] = widths.get(int(n.fitted.width), 0) + 1
    save_json(dict(full=full, prune=prune, sparsify=spars, grid=grid, width_hist=dict(sorted(widths.items()))),
              os.path.join(a.run, "minimize.json"))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot([r["params"] for r in prune], [r["train_acc"] for r in prune], "o-", label="train")
    ax[0].plot([r["params"] for r in prune], [r["test_acc"] for r in prune], "s-", label="test")
    ax[0].set_xscale("log"); ax[0].set_xlabel("parameters"); ax[0].set_ylabel("accuracy")
    ax[0].set_title("(b) prune children with < m positives"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot([r["sparsity"] for r in spars], [r["train_acc"] for r in spars], "o-", label="train")
    ax[1].plot([r["sparsity"] for r in spars], [r["test_acc"] for r in spars], "s-", label="test")
    ax[1].set_xlabel("fraction of weights zeroed per node"); ax[1].set_title("(c) magnitude sparsification, no fine-tune")
    ax[1].legend(); ax[1].grid(alpha=.3)
    for s in (0.0, 0.5, 0.9):
        g = [r for r in grid if r["sparsity"] == s]
        ax[2].plot([r["params"] for r in g], [r["test_acc"] for r in g], "o-", label=f"sparsity {s}")
    ax[2].set_xscale("log"); ax[2].set_xlabel("non-zero parameters"); ax[2].set_ylabel("test accuracy")
    ax[2].set_title("(b)+(c) Pareto"); ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(a.run, "minimize.png"), dpi=150)
    print("DONE")


if __name__ == "__main__":
    main()
