"""Grow a tree-type network of small neural networks on CIFAR-10 / Cats-vs-Dogs.

python scripts/run_tree.py --dataset cifar10 --trunk scratch_cnn --trunk_epochs 3 --out results/cifar10_main
python scripts/run_tree.py --dataset catsdogs --trunk scratch_cnn --trunk_epochs 3 --img_size 64 --out results/catsdogs_main
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tree_nn.data import get_dataset
from tree_nn.features import build_trunk, extract, standardise
from tree_nn.learners_torch import TorchLearner
from tree_nn.tree import grow_tree, GrowConfig
from tree_nn.utils import set_seed, get_device, ints, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "catsdogs"])
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--img_size", type=int, default=64, help="Cats-vs-Dogs resize / ResNet input size")
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--noise", type=float, default=0.0, help="symmetric train-label noise")
    ap.add_argument("--trunk", default="scratch_cnn", help="resnet18 | random_cnn | scratch_cnn | multi:a+b")
    ap.add_argument("--trunk_epochs", type=int, default=3)
    ap.add_argument("--widths", default="0,16,64,256", help="node widths tried in ascending order (0 = linear)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--min_pos", type=int, default=1)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--max_nodes", type=int, default=20000)
    ap.add_argument("--use_B_multi", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    set_seed(a.seed)
    dev = get_device()
    print("device:", dev, flush=True)
    os.makedirs(a.out, exist_ok=True)

    D = get_dataset(a.dataset, a.data_root, a.n_train, a.noise, a.seed, a.img_size)
    K = D["n_classes"]
    print(f"train {len(D['Xtr'])}  test {len(D['Xte'])}  classes {K}  noisy labels {D['noisy_mask'].sum()}", flush=True)

    t0 = time.time()
    trunk = build_trunk(a.trunk, D["Xtr"], D["ytr"], K, dev, a.trunk_epochs, a.img_size, a.seed)
    Ftr = extract(trunk, D["Xtr"], dev)
    Fte = extract(trunk, D["Xte"], dev)
    Ftr, Fte, mu, sd = standardise(Ftr, Fte)
    t_trunk = time.time() - t0
    print(f"features {Ftr.shape}  ({t_trunk:.0f}s)", flush=True)
    ytr, yte = D["ytr"], D["yte"]

    learner = TorchLearner(dev, ints(a.widths), a.steps, a.lr, seed=a.seed)
    cfg = GrowConfig(a.max_depth, a.min_pos, a.margin, a.max_nodes, a.use_B_multi, verbose=True)
    t0 = time.time()
    tree = grow_tree(Ftr, ytr, K, learner, cfg)
    t_grow = time.time() - t0

    levels = []
    for d in range(tree.depth() + 1):
        levels.append(dict(depth=d, train_acc=tree.accuracy(Ftr, ytr, d), test_acc=tree.accuracy(Fte, yte, d),
                           params=tree.params_up_to_depth(d)))
        print(levels[-1], flush=True)
    pred = tree.predict(Ftr)
    nm = D["noisy_mask"]
    rd = tree.repair_depth(Ftr)
    rep = dict(repair_depth_hist={int(d): int((rd == d).sum()) for d in np.unique(rd)})
    if nm.any():
        from sklearn.metrics import roc_auc_score
        rep["frac_noisy_by_repair_depth"] = {int(d): float(nm[rd == d].mean()) for d in np.unique(rd)}
        rep["noise_detection_auroc"] = float(roc_auc_score(nm, rd))
    res = dict(
        args=vars(a), n_train=len(Ftr), n_test=len(Fte), n_classes=K, feat_dim=int(Ftr.shape[1]),
        trunk_seconds=t_trunk, grow_seconds=t_grow,
        root_train_acc=levels[0]["train_acc"], root_test_acc=levels[0]["test_acc"],
        tree_train_acc=float((pred == ytr).mean()), tree_test_acc=tree.accuracy(Fte, yte),
        tree_train_acc_vs_clean_labels=float((pred == D["ytr_clean"]).mean()),
        noisy_samples_memorised=float((pred[nm] == ytr[nm]).mean()) if nm.any() else None,
        n_nodes=tree.n_nodes(), n_params=tree.n_params(), max_depth_reached=tree.depth(),
        depth_hist=tree.depth_hist(), levels=levels,
        width_hist={int(w): int(sum(1 for n in tree.nodes() if n.fitted.width == w))
                    for w in sorted({n.fitted.width for n in tree.nodes()})},
        nodes=tree.summary_rows(), **rep,
    )
    save_json(res, os.path.join(a.out, "results.json"))
    torch.save(dict(tree=tree, mu=mu, sd=sd, classes=D["classes"]), os.path.join(a.out, "tree.pt"))
    torch.save(trunk.cpu(), os.path.join(a.out, "trunk.pt"))
    np.savez_compressed(os.path.join(a.out, "feats.npz"), Ftr=Ftr.astype(np.float32), Fte=Fte.astype(np.float32),
                        ytr=ytr, yte=yte, ytr_clean=D["ytr_clean"])

    ds = [l["depth"] for l in levels]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(ds, [l["train_acc"] for l in levels], "o-", label="train")
    ax[0].plot(ds, [l["test_acc"] for l in levels], "s-", label="test")
    ax[0].set_xlabel("tree depth used"); ax[0].set_ylabel("accuracy"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ds, [l["params"] for l in levels], "o-", color="C2")
    ax[1].set_yscale("log"); ax[1].set_xlabel("tree depth used"); ax[1].set_ylabel("parameters"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "accuracy_vs_depth.png"), dpi=150)
    print(f"DONE  train {res['tree_train_acc']:.4f}  test {res['tree_test_acc']:.4f}  nodes {res['n_nodes']}  "
          f"params {res['n_params']}  root-only test {res['root_test_acc']:.4f}")


if __name__ == "__main__":
    main()
