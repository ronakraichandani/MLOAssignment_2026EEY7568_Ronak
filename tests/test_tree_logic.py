"""Unit tests for the tree-growing logic (no PyTorch needed).

Run:  python tests/test_tree_logic.py
Checks the central claim: when every node interpolates its own masked problem,
the whole tree reaches 100% training accuracy -- even with label noise.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tree_nn.tree import grow_tree, GrowConfig, prune_min_pos
from tree_nn.learners_sk import SkLearner


def make_data(n, d, K, modes, noise, seed):
    rng = np.random.RandomState(seed)
    centers = rng.randn(K * modes, d) * 2.0
    lab = np.repeat(np.arange(K), modes)
    idx = rng.randint(0, K * modes, n)
    X = centers[idx] + rng.randn(n, d)
    y = lab[idx].copy()
    flip = rng.rand(n) < noise
    y[flip] = (y[flip] + rng.randint(1, K, flip.sum())) % K
    return X.astype(np.float64), y


def test_multiclass_noisy(use_B=False):
    X, y = make_data(1500, 12, 4, 3, 0.15, 0)
    tree = grow_tree(X, y, 4, SkLearner(), GrowConfig(max_depth=30, verbose=False, use_B_multi=use_B))
    acc = tree.accuracy(X, y)
    accs = [tree.accuracy(X, y, max_depth=d) for d in range(tree.depth() + 1)]
    print(f"[multi use_B={use_B}] nodes={tree.n_nodes()} depth={tree.depth()} "
          f"params={tree.n_params()} train acc by depth={np.round(accs, 3).tolist()}")
    assert acc == 1.0, acc
    # repairing deeper never hurts on the training set
    assert all(b >= a - 1e-12 for a, b in zip(accs, accs[1:])), accs
    # truncation / pruning stays a valid classifier but loses the guarantee
    p = prune_min_pos(tree, 5)
    assert p.n_params() <= tree.n_params()
    print(f"   pruned(min_pos=5): params={p.n_params()} acc={p.accuracy(X, y):.3f}")


def test_binary_noisy():
    X, y = make_data(1200, 10, 2, 4, 0.2, 1)
    tree = grow_tree(X, y, 2, SkLearner(), GrowConfig(max_depth=40, verbose=False))
    acc = tree.accuracy(X, y)
    print(f"[binary] nodes={tree.n_nodes()} depth={tree.depth()} params={tree.n_params()} acc={acc}")
    assert acc == 1.0, acc


def test_paper_example():
    # Fig. 1 of the paper: x1..x6, hyperplane misclassifies x5 (4,2) and x6 (-2,-1)
    X = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [4, 2], [-2, -1]], float)
    y = np.array([0, 0, 0, 1, 0, 1])           # x4 class 1 (C2), x5 class 0 (C4), x6 class 1 (C3)
    tree = grow_tree(X, y, 2, SkLearner(), GrowConfig(max_depth=10, verbose=False))
    assert tree.accuracy(X, y) == 1.0
    print(f"[paper example] nodes={tree.n_nodes()} acc={tree.accuracy(X, y)}")


if __name__ == "__main__":
    test_paper_example()
    test_multiclass_noisy(False)
    test_multiclass_noisy(True)
    test_binary_noisy()
    print("ALL TESTS PASSED")
