"""Datasets: CIFAR-10 and Cats-vs-Dogs (Microsoft/Kaggle 25k images), plus label noise."""
import io
import os
import urllib.request
import zipfile

import numpy as np

CATSDOGS_URL = ("https://download.microsoft.com/download/3/E/1/"
                "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_cifar10(root):
    import torchvision
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)
    return (tr.data, np.array(tr.targets), te.data, np.array(te.targets), CIFAR_CLASSES)


def load_catsdogs(root, size=64, n_test=5000, seed=0):
    """25k cat/dog photos squashed to size x size. Fixed split: n_test held out."""
    from PIL import Image
    os.makedirs(root, exist_ok=True)
    cache = os.path.join(root, f"catsdogs_{size}.npz")
    if not os.path.exists(cache):
        zpath = os.path.join(root, "kagglecatsanddogs.zip")
        if not os.path.exists(zpath):
            print("downloading Cats vs Dogs ...", flush=True)
            urllib.request.urlretrieve(CATSDOGS_URL, zpath)
        X, y = [], []
        with zipfile.ZipFile(zpath) as z:
            for name in sorted(z.namelist()):
                if not name.lower().endswith(".jpg"):
                    continue
                if "/Cat/" in name:
                    lab = 0
                elif "/Dog/" in name:
                    lab = 1
                else:
                    continue
                try:
                    im = Image.open(io.BytesIO(z.read(name))).convert("RGB").resize((size, size), Image.BILINEAR)
                except Exception:      # a handful of corrupt files in the original archive
                    continue
                X.append(np.asarray(im, dtype=np.uint8))
                y.append(lab)
        np.savez_compressed(cache, X=np.stack(X), y=np.array(y))
    d = np.load(cache)
    X, y = d["X"], d["y"]
    perm = np.random.RandomState(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    return X[n_test:], y[n_test:], X[:n_test], y[:n_test], ["cat", "dog"]


def add_label_noise(y, p, n_classes, seed=0):
    """Symmetric label noise: a fraction p of labels is moved to a different random class."""
    y = np.array(y).copy()
    if p <= 0:
        return y, np.zeros(len(y), bool)
    rng = np.random.RandomState(seed)
    flip = rng.rand(len(y)) < p
    y[flip] = (y[flip] + rng.randint(1, n_classes, flip.sum())) % n_classes
    return y, flip


def get_dataset(name, root="data", n_train=None, noise=0.0, seed=0, img_size=64):
    """Returns dict(Xtr, ytr (possibly noisy), ytr_clean, Xte, yte, classes, noisy_mask).
    Images are uint8 NHWC."""
    if name == "cifar10":
        Xtr, ytr, Xte, yte, classes = load_cifar10(root)
    elif name == "catsdogs":
        Xtr, ytr, Xte, yte, classes = load_catsdogs(root, size=img_size)
    else:
        raise ValueError(name)
    if n_train is not None and n_train < len(Xtr):
        idx = np.random.RandomState(seed).permutation(len(Xtr))[:n_train]
        Xtr, ytr = Xtr[idx], ytr[idx]
    ynoisy, flipped = add_label_noise(ytr, noise, len(classes), seed + 1)
    return dict(Xtr=Xtr, ytr=ynoisy, ytr_clean=ytr, Xte=Xte, yte=yte, classes=classes,
                noisy_mask=flipped, n_classes=len(classes))
