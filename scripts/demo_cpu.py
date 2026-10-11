"""CPU-only sanity demo (no PyTorch, no downloads): tree of linear nodes on sklearn's 8x8 digits with
15% label noise.  Writes the same files as run_tree.py / minimize.py so that make_report.py can be
tested anywhere.   python scripts/demo_cpu.py --out examples/demo_digits
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tree_nn.data import add_label_noise
from tree_nn.learners_sk import SkLearner
from tree_nn.tree import grow_tree, GrowConfig, prune_min_pos
from tree_nn.utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/demo_digits")
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr_c, yte = train_test_split(X, y, test_size=0.3, random_state=a.seed, stratify=y)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr, flipped = add_label_noise(ytr_c, a.noise, 10, a.seed + 1)

    t0 = time.time()
    tree = grow_tree(Xtr, ytr, 10, SkLearner(), GrowConfig(max_depth=40, verbose=False))
    sec = time.time() - t0
    levels = [dict(depth=d, train_acc=tree.accuracy(Xtr, ytr, d), test_acc=tree.accuracy(Xte, yte, d),
                   params=tree.params_up_to_depth(d)) for d in range(tree.depth() + 1)]
    pred = tree.predict(Xtr)
    rd = tree.repair_depth(Xtr)
    from sklearn.metrics import roc_auc_score
    rep = dict(repair_depth_hist={int(d): int((rd == d).sum()) for d in np.unique(rd)},
               frac_noisy_by_repair_depth={int(d): float(flipped[rd == d].mean()) for d in np.unique(rd)},
               noise_detection_auroc=float(roc_auc_score(flipped, rd)))
    res = dict(args=dict(dataset="demo_digits", noise=a.noise, trunk="none (raw pixels)", widths="0",
                         max_depth=40, seed=a.seed),
               n_train=len(Xtr), n_test=len(Xte), n_classes=10, feat_dim=64, grow_seconds=sec,
               root_train_acc=levels[0]["train_acc"], root_test_acc=levels[0]["test_acc"],
               tree_train_acc=float((pred == ytr).mean()), tree_test_acc=tree.accuracy(Xte, yte),
               tree_train_acc_vs_clean_labels=float((pred == ytr_c).mean()),
               noisy_samples_memorised=float((pred[flipped] == ytr[flipped]).mean()),
               n_nodes=tree.n_nodes(), n_params=tree.n_params(), max_depth_reached=tree.depth(),
               depth_hist=tree.depth_hist(), levels=levels, width_hist={0: tree.n_nodes()},
               nodes=tree.summary_rows(), **rep)
    save_json(res, os.path.join(a.out, "results.json"))

    prune = []
    for m in (1, 2, 3, 5, 10, 20, 50):
        p = prune_min_pos(tree, m)
        prune.append(dict(min_pos=m, params=p.n_params(), nodes=p.n_nodes(), train_acc=p.accuracy(Xtr, ytr),
                          test_acc=p.accuracy(Xte, yte)))
    save_json(dict(full=dict(params=tree.n_params(), nodes=tree.n_nodes(), train_acc=res["tree_train_acc"],
                             test_acc=res["tree_test_acc"]), prune=prune, sparsify=[], grid=[],
                   width_hist={0: tree.n_nodes()}), os.path.join(a.out, "minimize.json"))

    ds = [l["depth"] for l in levels]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(ds, [l["train_acc"] for l in levels], "o-", label="train")
    ax[0].plot(ds, [l["test_acc"] for l in levels], "s-", label="test")
    ax[0].set_xlabel("tree depth used"); ax[0].set_ylabel("accuracy"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ds, [l["params"] for l in levels], "o-", color="C2"); ax[1].set_yscale("log")
    ax[1].set_xlabel("tree depth used"); ax[1].set_ylabel("parameters"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "accuracy_vs_depth.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot([r["params"] for r in prune], [r["train_acc"] for r in prune], "o-", label="train")
    ax.plot([r["params"] for r in prune], [r["test_acc"] for r in prune], "s-", label="test")
    ax.set_xscale("log"); ax.set_xlabel("parameters"); ax.set_ylabel("accuracy"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "minimize.png"), dpi=150); plt.close(fig)
    print(f"train {res['tree_train_acc']:.4f} test {res['tree_test_acc']:.4f} (root-only test {res['root_test_acc']:.4f}) "
          f"nodes {res['n_nodes']} params {res['n_params']} depth {res['max_depth_reached']} "
          f"memorised noisy labels {res['noisy_samples_memorised']:.3f}  noise-detection AUROC {res['noise_detection_auroc']:.3f}  "
          f"flipped fraction by repair depth {res['frac_noisy_by_repair_depth']}")


if __name__ == "__main__":
    main()
