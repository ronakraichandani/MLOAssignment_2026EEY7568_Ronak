"""ResNet-18 with variable width k (channels k,2k,4k,8k), as in Nakkiran et al.,
'Deep Double Descent' -- used for the model-wise double-descent experiment."""
import torch.nn as nn
import torch.nn.functional as Fn


class Block(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(o)
        self.sc = nn.Sequential()
        if stride != 1 or i != o:
            self.sc = nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        y = Fn.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return Fn.relu(y + self.sc(x))


class ResNet18k(nn.Module):
    def __init__(self, k=64, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, k, 3, 1, 1, bias=False), nn.BatchNorm2d(k), nn.ReLU())
        cfg, layers, i = [(k, 1), (2 * k, 2), (4 * k, 2), (8 * k, 2)], [], k
        for o, s in cfg:
            layers += [Block(i, o, s), Block(o, o, 1)]
            i = o
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(8 * k, n_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(Fn.adaptive_avg_pool2d(x, 1).flatten(1))
