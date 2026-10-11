"""Plotting helpers that need only numpy / matplotlib (no PyTorch)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def last(r, key, n=3):
    """median of the last n evaluations: a single final epoch can be a training spike near interpolation"""
    return float(np.median(r[key][-n:]))


def plot_dd(res, out):
    ks = sorted(int(k) for k in res["runs"])
    runs = [res["runs"][str(k)] for k in ks]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    ax[0].plot(ks, [last(r, "train_err") for r in runs], "o-", label="train error (noisy labels)")
    ax[0].plot(ks, [last(r, "test_err") for r in runs], "s-", label="test error")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("ResNet-18 width k"); ax[0].set_ylabel("error")
    ax[0].set_title("model-wise (median of last 3 evals)", fontsize=9); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    P = [r["params"] for r in runs]
    ax[1].plot(P, [min(r["test_err"]) for r in runs], "^-", color="C3", label="best epoch")
    ax[1].plot(P, [last(r, "test_err") for r in runs], "s-", color="C1", label="end of training (median last 3)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("parameters"); ax[1].set_ylabel("test error")
    ax[1].legend(fontsize=7); ax[1].set_title("test error vs size", fontsize=9); ax[1].grid(alpha=.3)
    gap = [last(r, "test_err") - min(r["test_err"]) for r in runs]
    ax[2].plot(ks, gap, "o-", color="C4"); ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("ResNet-18 width k"); ax[2].set_ylabel("end-of-training minus best-epoch test error")
    ax[2].set_title("epoch-wise overfitting penalty", fontsize=9); ax[2].grid(alpha=.3)
    pick = [k for k in ks if k in (2, 4, 8, 16, 64)] or ks[:: max(1, len(ks) // 4)]
    for k in pick:
        r = res["runs"][str(k)]
        ax[3].plot(r["epoch"], r["test_err"], label=f"k={k}")
    ax[3].set_xlabel("epoch"); ax[3].set_ylabel("test error"); ax[3].set_title("epoch-wise", fontsize=9)
    ax[3].legend(fontsize=7); ax[3].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "double_descent.png"), dpi=150); plt.close(fig)
