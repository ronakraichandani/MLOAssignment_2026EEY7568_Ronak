"""Feature trunks.  The tree's nodes are small networks on top of trunk features.

trunk kinds
-----------
resnet18      ImageNet-pretrained ResNet-18 (external data!), 512-d, shared by all nodes
random_cnn    frozen random 3-layer CNN (no learning at all), 1024-d
scratch_cnn   small CNN trained from scratch on the (possibly noisy) training set for only a
              few epochs, then frozen; 512-d.  `trunk_epochs` controls how much of the
              problem the trunk solves, i.e. how much is left for the tree.
multi:a+b     concatenation of several separately-trained trunks, e.g. multi:scratch_cnn+random_cnn
All trunks take float images in [0,1], NCHW, and handle their own normalisation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)


class Norm(nn.Module):
    def __init__(self, mean=IM_MEAN, std=IM_STD):
        super().__init__()
        self.register_buffer("m", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.m) / self.s


class ResNet18Trunk(nn.Module):
    dim = 512

    def __init__(self, size=64):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        self.m, self.size, self.norm = m, size, Norm()

    def forward(self, x):
        x = Fn.interpolate(self.norm(x), size=self.size, mode="bilinear", align_corners=False)
        return self.m(x)


class RandomCNNTrunk(nn.Module):
    dim = 1024

    def __init__(self, seed=0):
        super().__init__()
        g = torch.manual_seed(seed)
        self.norm = Norm()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten())

    def forward(self, x):
        return self.net(self.norm(x))


class ScratchCNNTrunk(nn.Module):
    dim = 512

    def __init__(self, n_classes):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.norm = Norm()
        self.body = nn.Sequential(blk(3, 64), blk(64, 128), blk(128, 256), nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(), nn.Linear(256, 512), nn.ReLU())
        self.head = nn.Linear(512, n_classes)          # only used while training the trunk

    def forward(self, x):
        return self.body(self.norm(x))


class ConcatTrunk(nn.Module):
    def __init__(self, trunks):
        super().__init__()
        self.trunks = nn.ModuleList(trunks)
        self.dim = sum(t.dim for t in trunks)

    def forward(self, x):
        return torch.cat([t(x) for t in self.trunks], 1)


def to_tensor(X_uint8, device):
    """uint8 NHWC numpy -> float NCHW in [0,1]"""
    return torch.as_tensor(X_uint8, device=device).permute(0, 3, 1, 2).float().div_(255.0)


def train_scratch_trunk(model, X, y, epochs, device, bs=256, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    yt = torch.as_tensor(y, device=device)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, correct = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs].numpy()
            xb = to_tensor(X[idx], device)
            z = model.head(model(xb))
            loss = Fn.cross_entropy(z, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            correct += (z.argmax(1) == yt[idx]).sum().item()
        print(f"  trunk epoch {ep + 1}/{epochs}  loss {tot / n:.3f}  train acc {correct / n:.3f}", flush=True)
    return model


def build_trunk(kind, X, y, n_classes, device, trunk_epochs=3, img_size=64, seed=0):
    if kind.startswith("multi:"):
        parts = [build_trunk(k, X, y, n_classes, device, trunk_epochs, img_size, seed + i)
                 for i, k in enumerate(kind[6:].split("+"))]
        trunk = ConcatTrunk(parts)
    elif kind == "resnet18":
        trunk = ResNet18Trunk(img_size)
    elif kind == "random_cnn":
        trunk = RandomCNNTrunk(seed)
    elif kind == "scratch_cnn":
        trunk = train_scratch_trunk(ScratchCNNTrunk(n_classes), X, y, trunk_epochs, device, seed=seed)
    else:
        raise ValueError(kind)
    trunk.to(device).eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk


@torch.no_grad()
def extract(trunk, X_uint8, device, bs=512):
    out = []
    for i in range(0, len(X_uint8), bs):
        out.append(trunk(to_tensor(X_uint8[i:i + bs], device)).float().cpu())
    return torch.cat(out).numpy()


def standardise(Ftr, Fte):
    mu = Ftr.mean(0, keepdims=True)
    sd = Ftr.std(0, keepdims=True) + 1e-6
    return (Ftr - mu) / sd, (Fte - mu) / sd, mu.astype(np.float32), sd.astype(np.float32)
