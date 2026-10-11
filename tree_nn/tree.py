"""Tree-type network with small-neural-network nodes (backend-agnostic core).

This module contains ONLY the tree-growing logic, written against numpy and a
tiny "learner" interface, so that it can be unit-tested without PyTorch.

Idea (generalising Jayadeva's "A Tree Type Neural Network")
-----------------------------------------------------------
A node is a small network trained on a (masked) subset of the training set.
Samples it still gets wrong are repaired by *child* nodes:

  binary node  (one logit z, prediction = [z > 0])
      A child : fires (=1) on false negatives, 0 on every other sample with
                true label 0 or false positive; "don't care" on true positives.
      B child : fires on false positives, 0 on true positives / false negatives;
                "don't care" on true negatives.
      output  : z' = z + wA * step(A) - wB * step(B),  wA, wB >= 0

  multiclass node (K logits, prediction = argmax)
      A_k child fires on samples of class k that the node did NOT predict as k
      (false negatives of class k); "don't care" on true positives of class k.
      output  : z'_k = z_k + wA_k * step(A_k)
      (B children are redundant for K >= 2 -- raising the true logit repairs
      every error -- but can be enabled with GrowConfig.use_B_multi.)

The correction weights are set in closed form, just like the weights 5.5 / -5.5
chosen "by inspection" in the paper, but with a guarantee: wA_k is the largest
logit gap that has to be bridged over the false negatives, plus a margin.

Children are themselves binary nodes trained on their own masked problem. If a
child cannot fit its problem perfectly it grows children of its own.  Hence the
recursion ends exactly when every node interpolates its own (masked) set.
Under that condition the whole tree is *provably* 100% accurate on the training
set (see tests/test_tree_logic.py and the docstring of `_grow`).

Learner interface
-----------------
learner.fit(F, t, mask, kind, n_out) -> fitted
    F    : (N, d) float array, all samples
    t    : (N,) int labels (kind == 'multi') or {0,1} targets (kind == 'binary')
    mask : (N,) bool, only samples with mask==True are used for fitting
    kind : 'multi' | 'binary'
    n_out: number of logits (K for multi, 1 for binary)
fitted.logits(F) -> (N, K) or (N,) array of logits
fitted.n_params  -> int
fitted.width     -> int (hidden width, 0 == linear)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


@dataclass
class GrowConfig:
    max_depth: int = 8          # children of a depth-d node live at depth d+1
    min_pos: int = 1            # do not spawn a child for fewer than this many positives
    margin: float = 0.05        # safety margin added to closed-form weights
    max_nodes: int = 100000
    use_B_multi: bool = False   # also spawn B children for multiclass nodes
    verbose: bool = True


class Node:
    def __init__(self, fitted, kind, depth, role, cls, n_mask):
        self.fitted = fitted
        self.kind = kind              # 'multi' | 'binary'
        self.depth = depth
        self.role = role              # 'root' | 'A' | 'B'
        self.cls = cls                # class index this child repairs (None for root)
        self.n_mask = int(n_mask)     # size of the training problem this node solved
        self.n_pos = 0                # number of positives (children) / samples (root)
        self.n_err = 0                # training errors left on its own mask
        self.A, self.B = {}, {}       # child nodes keyed by class (binary node: key 0)
        self.wA, self.wB = {}, {}
        self.pos_idx = None           # indices of its positive training samples (for visualisation)
        self.id = -1

    # --------------------------------------------------------------- forward
    def subtree_max_depth(self):
        if getattr(self, "_md", None) is None:
            self._md = max([self.depth] + [c.subtree_max_depth() for c in self.children()])
        return self._md

    def logits(self, F, max_depth=None):
        """Forward pass.  With max_depth=m a child is used only if its WHOLE sub-tree
        lies at depth <= m.  The closed-form weights assume a child is perfect on its
        own problem; a half-repaired child would fire spuriously with a large weight
        and can lower accuracy.  Suppressing it (output 0) is graceful: it only
        leaves that child's positives uncorrected."""
        z = np.array(self.fitted.logits(F), dtype=np.float64)
        if max_depth is not None and self.role != "root" and self.subtree_max_depth() > max_depth:
            return np.full_like(z, -1.0)
        for k, c in self.A.items():
            a = (c.logits(F, max_depth) > 0).astype(np.float64)
            if self.kind == "multi":
                z[:, k] += self.wA[k] * a
            else:
                z += self.wA[k] * a
        for k, c in self.B.items():
            b = (c.logits(F, max_depth) > 0).astype(np.float64)
            if self.kind == "multi":
                z[:, k] -= self.wB[k] * b
            else:
                z -= self.wB[k] * b
        return z

    def deepest_firing(self, F):
        """Per sample: depth of the deepest repair node on a firing chain (0 = no child fires)."""
        out = np.zeros(len(F), dtype=np.int64)
        for c in self.children():
            fire = c.logits(F) > 0
            out = np.maximum(out, np.where(fire, np.maximum(c.depth, c.deepest_firing(F)), 0))
        return out

    def children(self):
        return list(self.A.values()) + list(self.B.values())

    def own_params(self):
        return int(self.fitted.n_params) + 2 * len(self.children())


class Tree:
    def __init__(self, root, n_classes):
        self.root, self.n_classes = root, n_classes

    def predict(self, F, max_depth=None):
        z = self.root.logits(F, max_depth)
        return z.argmax(1)

    def accuracy(self, F, y, max_depth=None):
        return float((self.predict(F, max_depth) == y).mean())

    def nodes(self):
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children())
        return sorted(out, key=lambda n: n.id)

    def repair_depth(self, F):
        """How deep into the tree each sample had to be repaired. Samples that are hard to fit
        (e.g. mislabelled ones) need deep chains of repair nodes."""
        return self.root.deepest_firing(F)

    def propagate_cls(self):
        """Make every deeper node carry the class of the depth-1 node whose repair chain it belongs to
        (older pickled trees stored a placeholder 0 for grand-children).  Only affects labels."""
        def walk(n, inherited):
            for c in n.children():
                if n.role != "root":
                    c.cls = inherited
                walk(c, c.cls)
        walk(self.root, None)

    def n_nodes(self):
        return len(self.nodes())

    def n_params(self):
        return sum(n.own_params() for n in self.nodes())

    def depth(self):
        return max(n.depth for n in self.nodes())

    def depth_hist(self):
        h = {}
        for n in self.nodes():
            h[n.depth] = h.get(n.depth, 0) + 1
        return dict(sorted(h.items()))

    def params_up_to_depth(self, d):
        return sum(n.own_params() for n in self.nodes() if n.depth <= d)

    def summary_rows(self):
        return [dict(id=n.id, depth=n.depth, role=n.role, cls=n.cls, n_mask=n.n_mask,
                     n_pos=n.n_pos, n_err=n.n_err, width=int(n.fitted.width),
                     params=int(n.fitted.n_params), n_children=len(n.children()))
                for n in self.nodes()]


class _State:
    def __init__(self):
        self.count = 0


def grow_tree(F, y, n_classes, learner, cfg: GrowConfig | None = None) -> Tree:
    cfg = cfg or GrowConfig()
    st = _State()
    mask = np.ones(len(F), dtype=bool)
    root = _grow(F, np.asarray(y), mask, "multi", n_classes, 0, "root", None, learner, cfg, st)
    root.n_pos = len(F)
    return Tree(root, n_classes)


def _grow(F, t, mask, kind, n_out, depth, role, cls, learner, cfg, st):
    """Fit a node on `mask`, then recursively repair its errors.

    Correctness argument.  Let a node have base logits z and let every child be
    perfect on its own mask (induction on the tree height; a node without
    children has no errors on its mask by construction).

    Multiclass: an error sample has true class k and prediction j != k, so it is a
    false negative of class k, hence a *positive* of child A_k, hence A_k = 1 on
    it.  Its logit z_k is raised by wA_k >= (max_m z_m - z_k) + margin, so k wins.
    Every other sample inside the mask has A_m = 0 for m != y (A_m's negatives
    contain every sample whose label is not m) and therefore only its own true
    logit can be raised -- it stays correct.  True positives of class k are
    don't-cares for A_k, but if A_k fires on them it only raises the true logit.
    Binary: analogous with A (raises z on false negatives) and B (lowers z on
    false positives); don't-care sets are chosen so the outputs on don't-cares can
    only push z in the already-correct direction.
    """
    fitted = learner.fit(F, t, mask, kind, n_out)
    z = np.asarray(fitted.logits(F), dtype=np.float64)
    node = Node(fitted, kind, depth, role, cls, mask.sum())
    node.id = st.count
    st.count += 1

    if kind == "multi":
        pred = z.argmax(1)
        wrong = mask & (pred != t)
    else:
        pred = z > 0
        wrong = mask & (((t == 1) & ~pred) | ((t == 0) & pred))
    node.n_err = int(wrong.sum())
    if cfg.verbose and depth <= 1:
        print(f"{'  ' * depth}[node {node.id} {role}{'' if cls is None else cls} d={depth}] "
              f"n={node.n_mask} width={fitted.width} errors={node.n_err}", flush=True)
    if node.n_err == 0 or depth >= cfg.max_depth or st.count >= cfg.max_nodes:
        return node

    if kind == "multi":
        K = n_out
        top1 = z.max(1)
        z2 = z.copy()
        z2[np.arange(len(z)), pred] = -np.inf
        second = z2.max(1)
        for k in range(K):
            tp = mask & (t == k) & (pred == k)
            fn = mask & (t == k) & (pred != k)
            fp = mask & (t != k) & (pred == k)
            if fn.sum() >= cfg.min_pos:
                gap = (top1 - z[:, k])[fn].max()
                child = _grow(F, fn.astype(np.int64), mask & ~tp, "binary", 1, depth + 1,
                              "A", k, learner, cfg, st)
                child.n_pos = int(fn.sum()); child.pos_idx = np.flatnonzero(fn)[:5000]
                node.A[k] = child
                node.wA[k] = float(gap + cfg.margin)
            if cfg.use_B_multi and fp.sum() >= cfg.min_pos:
                gap = (z[:, k] - second)[fp].max()
                child = _grow(F, fp.astype(np.int64), mask & ((t == k) | fp), "binary", 1,
                              depth + 1, "B", k, learner, cfg, st)
                child.n_pos = int(fp.sum()); child.pos_idx = np.flatnonzero(fp)[:5000]
                node.B[k] = child
                node.wB[k] = float(gap + cfg.margin)
    else:
        tp = mask & (t == 1) & pred
        tn = mask & (t == 0) & ~pred
        fn = mask & (t == 1) & ~pred
        fp = mask & (t == 0) & pred
        if fn.sum() >= cfg.min_pos:
            child = _grow(F, fn.astype(np.int64), mask & ~tp, "binary", 1, depth + 1,
                          "A", cls, learner, cfg, st)
            child.n_pos = int(fn.sum()); child.pos_idx = np.flatnonzero(fn)[:5000]
            node.A[0] = child
            node.wA[0] = float((-z[fn]).max() + cfg.margin)
        if fp.sum() >= cfg.min_pos:
            child = _grow(F, fp.astype(np.int64), mask & ~tn, "binary", 1, depth + 1,
                          "B", cls, learner, cfg, st)
            child.n_pos = int(fp.sum()); child.pos_idx = np.flatnonzero(fp)[:5000]
            node.B[0] = child
            node.wB[0] = float(z[fp].max() + cfg.margin)
    return node


# ---------------------------------------------------------------- size reduction
def prune_min_pos(tree: Tree, m: int) -> Tree:
    """Return a copy in which every child with fewer than m positives (and its whole
    subtree) is removed.  Remaining correction weights stay valid, but training
    accuracy is no longer guaranteed to be 100%."""
    t2 = copy.deepcopy(tree)
    for n in t2.nodes():
        for d, w in ((n.A, n.wA), (n.B, n.wB)):
            for k in [k for k, c in d.items() if c.n_pos < m]:
                del d[k]
                del w[k]
    for n in t2.nodes():
        n._md = None
    return t2


def sparsify_tree(tree: Tree, frac: float) -> Tree:
    """Return a copy with `frac` of the smallest-magnitude weights of every node zeroed
    (only for fitted objects that implement .sparsify)."""
    t2 = copy.deepcopy(tree)
    for n in t2.nodes():
        if hasattr(n.fitted, "sparsify"):
            n.fitted.sparsify(frac)
    return t2
