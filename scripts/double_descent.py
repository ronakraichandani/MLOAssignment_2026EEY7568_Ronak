"""Model-wise and epoch-wise double descent with width-k ResNet-18 + label noise
(the Nakkiran et al. 'Deep Double Descent' recipe: no augmentation, Adam lr 1e-4, noisy labels).

python scripts/double_descent.py --dataset cifar10 --noise 0.15 --n_train 15000 --out results/dd_cifar10
python scripts/double_descent.py --out results/dd_cifar10 --plot_only
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
from tree_nn.models import ResNet18k
from tree_nn.plots import plot_dd
from tree_nn.utils import set_seed, get_device, ints, save_json, load_json

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.247, 0.243, 0.261)


def prep(X, dev):
    x = torch.as_tensor(X, device=dev).permute(0, 3, 1, 2).float().div_(255)
    m = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return ((x - m) / s).half()


@torch.no_grad()
def evaluate(model, X, y, bs=1000):
    model.eval()
    wrong, loss = 0, 0.0
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(X[i:i + bs])
        wrong += (z.argmax(1) != y[i:i + bs]).sum().item()
        loss += Fn.cross_entropy(z.float(), y[i:i + bs], reduction="sum").item()
    return wrong / len(X), loss / len(X)


def run_width(k, Xtr, ytr, Xte, yte, K, epochs, bs, lr, eval_every, seed):
    torch.manual_seed(seed)
    dev = Xtr.device
    model = ResNet18k(k, K).to(dev).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    hist = dict(epoch=[], train_err=[], test_err=[], train_loss=[], test_loss=[])
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = Fn.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if ep % eval_every == 0 or ep == epochs or ep == 1:
            tre, trl = evaluate(model, Xtr, ytr)
            tee, tel = evaluate(model, Xte, yte)
            for key, v in zip(hist, (ep, tre, tee, trl, tel)):
                hist[key].append(v)
            print(f"  k={k:3d} ep {ep:4d}  train err {tre:.4f}  test err {tee:.4f}", flush=True)
    return dict(k=k, params=n_params, **hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "catsdogs"])
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--img_size", type=int, default=32)
    ap.add_argument("--n_train", type=int, default=15000)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--widths", default="1,2,4,6,8,12,16,24,32,48,64")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "dd_results.json")
    if a.plot_only:
        plot_dd(load_json(jpath), a.out)
        return

    set_seed(a.seed)
    dev = get_device()
    D = get_dataset(a.dataset, a.data_root, a.n_train, a.noise, a.seed, a.img_size)
    Xtr, Xte = prep(D["Xtr"], dev), prep(D["Xte"], dev)
    ytr = torch.as_tensor(D["ytr"], device=dev).long()
    yte = torch.as_tensor(D["yte"], device=dev).long()
    res = load_json(jpath) if os.path.exists(jpath) else dict(args=vars(a), runs={})
    for k in ints(a.widths):
        if str(k) in res["runs"]:                 # resume after a Colab disconnect
            continue
        t0 = time.time()
        res["runs"][str(k)] = run_width(k, Xtr, ytr, Xte, yte, D["n_classes"], a.epochs, a.bs, a.lr,
                                        a.eval_every, a.seed)
        res["runs"][str(k)]["seconds"] = time.time() - t0
        save_json(res, jpath)
        plot_dd(res, a.out)
    print("DONE")


if __name__ == "__main__":
    main()
