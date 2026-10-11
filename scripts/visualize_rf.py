"""Receptive-field figures for a grown tree.

python scripts/visualize_rf.py --run results/cifar10_main --dataset cifar10
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tree_nn.data import get_dataset
from tree_nn.rf import visualise
from tree_nn.utils import get_device, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--n_children", type=int, default=8)
    ap.add_argument("--max_depth", type=int, default=3)
    a = ap.parse_args()
    dev = get_device()
    res = load_json(os.path.join(a.run, "results.json"))
    args = res["args"]
    ds = a.dataset or args["dataset"]
    # same seeds => same subset and same noisy labels as the training run
    D = get_dataset(ds, a.data_root, args["n_train"], args["noise"], args["seed"], args["img_size"])
    ck = torch.load(os.path.join(a.run, "tree.pt"), weights_only=False)
    trunk = torch.load(os.path.join(a.run, "trunk.pt"), weights_only=False).to(dev).eval()
    d = np.load(os.path.join(a.run, "feats.npz"))
    Ftr = d["Ftr"].astype(np.float32)
    assert (d["ytr"] == D["ytr"]).all(), "label mismatch: dataset arguments differ from the training run"
    agg = visualise(ck["tree"], trunk, Ftr, D["Xtr"], D["ytr"], ck["mu"], ck["sd"], D["classes"],
                    os.path.join(a.run, "rf"), dev, a.n_children, a.max_depth)
    for k, v in agg.items():
        print(k, v)


if __name__ == "__main__":
    main()
