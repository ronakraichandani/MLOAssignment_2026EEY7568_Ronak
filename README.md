# Tree-type networks of small neural networks (CIFAR-10, Cats vs Dogs)

Graded assignment "Tree type network revisited": grow a tree whose nodes are small neural networks, drive the
training accuracy to ~100 %, look for double descent, visualise node receptive fields, and shrink the network.

## Idea in one paragraph
A root node (small MLP / linear layer on frozen trunk features) is trained first. Samples it gets wrong are repaired by
child nodes, exactly as in Jayadeva's note, but generalised to **multiclass** and to **neural-network nodes**:

* child `A_k` fires on class-`k` samples the parent did *not* predict as `k`; it is silent on all samples whose label is
  not `k`; it is **don't-care** on correctly predicted class-`k` samples (they are simply excluded from its training set);
* the parent logit becomes `z'_k = z_k + w_k · step(A_k(x))` with `w_k` set in **closed form** (largest logit gap to bridge + margin);
* binary nodes (all children) additionally use `B` children that lower the logit on false positives (as in the paper);
* a child that cannot interpolate its own masked problem grows children of its own;
* every node tries hidden widths in ascending order (0 = linear, 16, 64, 256) and keeps the **smallest that interpolates**.

If every node interpolates its own problem, the tree is exactly 100 % accurate on the training set
(argument in `tree_nn/tree.py::_grow`, checked by `tests/test_tree_logic.py`, including with 15–20 % label noise).

Extra analyses beyond the brief: *repair depth* (how deep a sample's repair chain goes) as a label-noise detector;
graceful depth truncation; post-hoc pruning / sparsification curves; tree-wise "double descent" sweep.

## Layout
```
tree_nn/tree.py            tree growth, closed-form weights, pruning (numpy only, unit tested)
tree_nn/learners_torch.py  node networks + minimum-width search (PyTorch)
tree_nn/learners_sk.py     scikit-learn node learner (tests / CPU demo)
tree_nn/features.py        trunks: scratch_cnn | random_cnn | resnet18 (ImageNet) | multi:a+b
tree_nn/data.py            CIFAR-10, Cats-vs-Dogs (auto-download), label noise
tree_nn/rf.py              receptive fields: SmoothGrad, activation maximisation, localisation statistics
tree_nn/models.py          width-k ResNet-18 for double descent
scripts/run_tree.py        grow + evaluate a tree          -> results/<run>/results.json, tree.pt, figures
scripts/double_descent.py  model-wise / epoch-wise DD      -> results/dd_*/dd_results.json, figure
scripts/tree_sweep.py      tree-wise DD (node width sweep) -> results/sweep_*/
scripts/minimize.py        prune / sparsify a grown tree   -> results/<run>/minimize.json, figure
scripts/visualize_rf.py    receptive-field figures         -> results/<run>/rf/
scripts/demo_cpu.py        CPU-only demo (sklearn digits), no downloads
make_report.py             builds the PDF from the real result files
notebooks/colab_run.ipynb  runs everything on Colab
interpretation_template.md template for the "interpretations and conclusions" section
```

## What has and has not been verified
* **Verified here (CPU, no PyTorch):** the tree-growing logic — 100 % training accuracy on noisy multiclass and binary synthetic
  data and on noisy digits; monotone depth truncation; pruning; `make_report.py` renders a complete PDF from result files.
* **Not executed yet:** everything that needs PyTorch, a GPU or the datasets (`learners_torch.py`, `features.py`, `data.py`,
  `rf.py`, all of `scripts/` except `demo_cpu.py`). These were syntax-checked and reviewed only. **Run the quick trial first.**
* **No results are included.** All numbers in the PDF come from your own runs. Run times are untested estimates; the
  `--max_nodes`, `--max_depth`, `--steps`, `--n_train` options exist to keep runs within your budget.

## Running on Colab
1. Zip this folder as `tree_nn_project.zip` (top-level folder `tree_nn_project`), open `notebooks/colab_run.ipynb` in Colab with a T4 GPU runtime.
2. Run the cells in order: setup → unit tests → **quick trial** → main runs → minimisation / receptive fields → double descent → report.
3. Look at the figures and tables, write `interpretation.md` (start from `interpretation_template.md`), set `REPO_URL`, run the report cell.

Local equivalent:
```bash
pip install -r requirements.txt
python tests/test_tree_logic.py
python scripts/run_tree.py --dataset cifar10 --trunk scratch_cnn --trunk_epochs 3 --out results/cifar10_clean
python scripts/minimize.py --run results/cifar10_clean
python scripts/visualize_rf.py --run results/cifar10_clean
python scripts/double_descent.py --dataset cifar10 --noise 0.15 --n_train 15000 --out results/dd_cifar10
python scripts/tree_sweep.py --dataset cifar10 --noise 0.15 --n_train 15000 --out results/sweep_cifar10
python make_report.py --repo_url https://github.com/<you>/<repo> --interpretation interpretation.md --out report.pdf
```

## Knobs worth knowing
* `--trunk_epochs`: a stronger trunk leaves less for the tree (smaller tree); a weaker one gives a bigger, more interesting tree. `resnet18` uses ImageNet pre-training (external data) — say so if you report it.
* `--noise 0.15`: label noise makes interpolation non-trivial and is what produces the double-descent peak in `double_descent.py`.
* `--max_nodes` / `--max_depth`: if either cap is hit the tree may stop short of 100 %. `results.json` records the real train accuracy — report it honestly.
* Double descent is an empirical phenomenon; if you do not see a peak with the defaults, try more label noise, a different `--n_train`, more `--epochs`, or a finer `--widths` grid, and report what you saw.

## Publishing to a private GitHub repository
```bash
git init && git add . && git commit -m "Tree-type network of small neural networks"
gh repo create tree-type-nn --private --source=. --push      # or create it on github.com and `git remote add origin ...`
```
(`*.pt`, `*.npz` and `data/` are git-ignored; commit `results/**/results.json` and the PNGs.)
