import json
import os
import random

import numpy as np


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def ints(s):
    return tuple(int(x) for x in s.split(",") if x != "")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(path):
    with open(path) as f:
        return json.load(f)
