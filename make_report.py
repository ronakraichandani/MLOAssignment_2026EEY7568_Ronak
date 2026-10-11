"""Build the assignment PDF from the REAL experiment outputs in results/.

python make_report.py --results results --repo_url https://github.com/<you>/<repo> \
       --interpretation interpretation.md --out report.pdf

Every number and figure comes from the JSON/PNG files written by the scripts; nothing is hard-coded.
Sections whose experiments have not been run are marked "not run yet".  Section 'Interpretations
and conclusions' is read from interpretation.md -- write it AFTER looking at your results.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
                                KeepTogether)

SS = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=SS["BodyText"], fontSize=9.5, leading=13, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=SS["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=SS["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=4)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
TODO = ParagraphStyle("todo", parent=BODY, textColor=colors.HexColor("#B00020"))
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=8, leading=10)
W = A4[0] - 4 * cm


def P(t, s=BODY):
    return Paragraph(t, s)


def img(path, width=W, caption=None):
    if not os.path.exists(path):
        return [P(f"[figure not available: {os.path.basename(path)} - experiment not run yet]", TODO)]
    from PIL import Image as PI
    w, h = PI.open(path).size
    out = [Image(path, width=width, height=width * h / w)]
    if caption:
        out.append(P(caption, CAP))
    return out


def table(rows, colw=None, header=True, fs=8):
    t = Table(rows, colWidths=colw, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), fs), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7"))]
    t.setStyle(TableStyle(st))
    return t


def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def arch_figure(path):
    fig, ax = plt.subplots(figsize=(9, 3.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    def box(x, y, w, h, t, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="k", lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)
    def arr(x1, y1, x2, y2, t=None):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1))
        if t:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, t, fontsize=6.5, ha="center")
    box(0.1, 1.5, 1.2, 1, "image x", "#F4F4F4")
    box(1.8, 1.5, 1.6, 1, "frozen trunk(s)\n(shared features)", "#DCE9F7")
    box(4.0, 1.4, 1.8, 1.2, "ROOT node\nsmall MLP / linear\nK logits z", "#FFE9C7")
    box(6.9, 3.0, 1.6, 0.8, "A_k child (binary)\nfires on FN of class k", "#E2F3E2")
    box(6.9, 1.8, 1.6, 0.8, "A_j child ...", "#E2F3E2")
    box(6.9, 0.5, 1.6, 0.8, "(B children: binary\nnodes only)", "#F7E0E0")
    box(8.8, 3.0, 1.1, 0.8, "grand-children\nrepair child errors", "#EFEFEF")
    arr(1.3, 2, 1.8, 2); arr(3.4, 2, 4.0, 2, "features")
    arr(5.8, 2.4, 6.9, 3.3); arr(5.8, 2.0, 6.9, 2.2); arr(5.8, 1.6, 6.9, 0.9)
    arr(8.5, 3.4, 8.8, 3.4)
    ax.text(5.9, 3.7, "z'_k = z_k + w_k * 1[A_k(x) > 0]", fontsize=8, ha="center")
    ax.text(5.0, 0.15, "child nodes see the same features and are trained only on their own masked problem "
            "(don't-care samples excluded)", fontsize=7, ha="center", style="italic")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close(fig)


def md_to_flow(text):
    out, bullets = [], []
    def flush():
        nonlocal bullets
        if bullets:
            out.append(P("<br/>".join("&bull; " + b for b in bullets)))
            bullets = []
    for blk in text.split("\n"):
        s = blk.strip()
        if s.startswith("## "):
            flush(); out.append(P(s[3:], H2))
        elif s.startswith("# "):
            flush(); out.append(P(s[2:], H2))
        elif s.startswith("- "):
            bullets.append(s[2:])
        elif s == "":
            flush()
        else:
            flush(); out.append(P(s))
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--repo_url", default="[PRIVATE GITHUB URL: pass --repo_url]")
    ap.add_argument("--author", default="Ronak Raichandani (2026EEY7568)")
    ap.add_argument("--interpretation", default="interpretation.md")
    ap.add_argument("--out", default="report.pdf")
    a = ap.parse_args()

    story = []
    arch_png = os.path.join(a.results, "architecture.png")
    os.makedirs(a.results, exist_ok=True)
    arch_figure(arch_png)

    story += [P("Tree-Type Networks of Small Neural Networks", ParagraphStyle("t", parent=SS["Title"], fontSize=20)),
              P(f"Graded assignment: tree type network revisited &mdash; CIFAR-10 and Cats vs Dogs<br/>{a.author}<br/>"
                f"Code (private repository): {a.repo_url}", ParagraphStyle("s", parent=BODY, alignment=1)),
              Spacer(1, 8)]

    runs = {}
    for p in sorted(glob.glob(os.path.join(a.results, "*", "results.json"))):
        r = load(p)
        runs[os.path.basename(os.path.dirname(p))] = (os.path.dirname(p), r)

    # ------------------------------------------------------------ 1 architecture
    story += [P("1. Architecture and implementation", H1),
              P("<b>Starting point.</b> In the original tree-type network a single neuron is trained first; the samples it "
                "misclassifies are partitioned into false negatives and false positives, and two child neurons (A and B) "
                "are trained to fire exactly on those samples and to stay silent on the others, with <i>don't-care</i> "
                "outputs on samples that are already safe. The children feed the parent through a positive (A) and a "
                "negative (B) weight, so every training sample ends up correctly classified. If a child's problem is not "
                "linearly separable, it grows children of its own."),
              P("<b>What is new here.</b> (i) Every node is a <i>small neural network</i> (linear, or a one-hidden-layer MLP) "
                "operating on features from a frozen trunk. (ii) The construction is extended to <b>multiclass</b>: the "
                "root has K logits; for each class k the child A<sub>k</sub> fires on the class-k samples the root did "
                "not predict as k (positive set), is silent on all samples whose label is not k, and is don't-care on "
                "true positives of class k. The corrected logit is z'<sub>k</sub> = z<sub>k</sub> + w<sub>k</sub>&middot;"
                "step(A<sub>k</sub>(x)). B children are redundant for K&ge;2 (raising the true logit already repairs every "
                "error), which halves the number of children; they are used for binary (child) nodes. (iii) The "
                "correction weights are set in <b>closed form</b> instead of by inspection: w<sub>k</sub> is the largest "
                "logit gap over the class-k false negatives plus a margin. (iv) Each node searches the "
                "<b>smallest hidden width</b> that interpolates its own masked training set (0 = linear, then 16, 64, 256 by "
                "default), so nodes are only as large as needed."),
              P("<b>Guarantee.</b> If every node interpolates its own masked problem, the whole tree classifies the "
                "training set perfectly: an error sample is a positive of exactly the child that raises its true logit by "
                "more than the gap; samples outside the positive set cannot be disturbed because every child is silent on "
                "samples whose label differs from the class it repairs, and may only raise the true logit on its "
                "don't-care set. Recursion terminates because masked problems shrink. This is verified by "
                "unit tests (tests/test_tree_logic.py) on noisy synthetic and noisy digit data."),
              ] + img(arch_png, W * 0.95, "Figure 1: Tree architecture. Nodes share frozen trunk features; each child solves a "
                                          "masked binary problem and is added to the logit of the class it repairs.")
    story += [P("<b>Features.</b> The task asks how to provide appropriate features. Implemented options (features.py): a "
                "<i>scratch CNN trunk</i> trained for only a few epochs on the (noisy) training labels and then frozen "
                "(so that it solves part of the problem and leaves a well-defined remainder for the tree); a frozen "
                "<i>random CNN</i>; an ImageNet-pretrained <i>ResNet-18</i> (uses external data, so reported separately); and "
                "<i>multi-trunk</i> concatenations of separately trained trunks. All nodes of a tree share the trunk, "
                "which also makes every node differentiable down to the pixels for receptive-field analysis."),
              P("<b>Implementation.</b> Python / PyTorch. tree.py contains the growth logic (backend independent), "
                "learners_torch.py the node networks, rf.py the receptive-field tools; scripts/ holds one script per "
                "experiment. Hyper-parameters of every run are stored in its results.json and listed below.")]

    # ------------------------------------------------------------ 2 results
    story += [P("2. Results: interpolation on CIFAR-10 and Cats vs Dogs", H1)]
    if not runs:
        story += [P("No run found in results/.", TODO)]
    for name, (d, r) in runs.items():
        ar = r["args"]
        story += [P(f"2.{list(runs).index(name) + 1} Run <i>{name}</i> ({ar.get('dataset')})", H2),
                  P(f"trunk: {ar.get('trunk')} (epochs {ar.get('trunk_epochs', '-')}), node widths tried: {ar.get('widths')}, "
                    f"max depth {ar.get('max_depth')}, label noise {ar.get('noise')}, train/test size "
                    f"{r['n_train']}/{r['n_test']}, feature dim {r['feat_dim']}, seed {ar.get('seed')}.")]
        rows = [["", "train acc", "test acc"],
                ["root node only", pct(r["root_train_acc"]), pct(r["root_test_acc"])],
                ["full tree", pct(r["tree_train_acc"]), pct(r["tree_test_acc"])]]
        story += [table(rows, [5 * cm, 3 * cm, 3 * cm]), Spacer(1, 4)]
        extra = [["nodes", r["n_nodes"]], ["parameters", r["n_params"]], ["max depth", r["max_depth_reached"]],
                 ["nodes per depth", r["depth_hist"]], ["node width histogram", r["width_hist"]],
                 ["train accuracy w.r.t. clean labels", pct(r.get("tree_train_acc_vs_clean_labels"))],
                 ["noisy labels memorised", pct(r.get("noisy_samples_memorised"))],
                 ["growth time (s)", f"{r.get('grow_seconds', 0):.0f}"]]
        story += [table(extra, [6 * cm, 9 * cm], header=False), Spacer(1, 4)]
        lv = [["depth used", "train acc", "test acc", "params"]] + [
            [l["depth"], pct(l["train_acc"]), pct(l["test_acc"]), l["params"]] for l in r["levels"]]
        story += [table(lv, [2.5 * cm, 3 * cm, 3 * cm, 3 * cm]), Spacer(1, 4)]
        story += img(os.path.join(d, "accuracy_vs_depth.png"), W * 0.8,
                     "Accuracy and size when only complete sub-trees up to a given depth are used.")
        if r.get("noise_detection_auroc") is not None:
            story += [P(f"<b>Repair depth as a label-noise detector.</b> The depth of the deepest repair node that fires on a "
                        f"training sample separates mislabelled from clean samples with AUROC "
                        f"{r['noise_detection_auroc']:.3f}. Fraction of flipped labels per repair depth: "
                        f"{ {k: round(v, 3) for k, v in r['frac_noisy_by_repair_depth'].items()} }.")]
        mz = load(os.path.join(d, "minimize.json"))
        if mz:
            story += [P("Size minimisation (pruning children that repair fewer than m samples, magnitude sparsification, "
                        "no fine-tuning):", BODY)]
            rows = [["m", "params", "nodes", "train acc", "test acc"]] + [
                [p["min_pos"], p["params"], p["nodes"], pct(p["train_acc"]), pct(p["test_acc"])] for p in mz["prune"]]
            story += [table(rows, [1.5 * cm, 3 * cm, 2.5 * cm, 3 * cm, 3 * cm])]
            if mz.get("sparsify"):
                rows = [["sparsity", "non-zero params", "train acc", "test acc"]] + [
                    [p["sparsity"], p["params"], pct(p["train_acc"]), pct(p["test_acc"])] for p in mz["sparsify"]]
                story += [Spacer(1, 4), table(rows, [2.5 * cm, 3.5 * cm, 3 * cm, 3 * cm])]
            story += img(os.path.join(d, "minimize.png"), W * 0.95)
        story += [Spacer(1, 6)]

    # ------------------------------------------------------------ 3 double descent
    story += [P("3. Overfitting and double descent", H1)]
    found = False
    for p in sorted(glob.glob(os.path.join(a.results, "*", "dd_results.json"))):
        found = True
        r = load(p); ar = r["args"]
        story += [P(f"Model-wise / epoch-wise: ResNet-18 of width k on {ar['dataset']} ({ar['n_train']} training images, "
                    f"{int(100 * ar['noise'])}% label noise, Adam lr {ar['lr']}, {ar['epochs']} epochs, no augmentation)."),
                  ]
        import statistics
        lm = lambda v, key: statistics.median(v[key][-3:])
        rows = [["k", "params", "train err (median last 3 evals)", "test err (median last 3 evals)", "best-epoch test err"]] + [
            [k, v["params"], f"{lm(v, 'train_err'):.3f}", f"{lm(v, 'test_err'):.3f}", f"{min(v['test_err']):.3f}"]
            for k, v in sorted(r["runs"].items(), key=lambda kv: int(kv[0]))]
        story += [table(rows, [1.5 * cm, 2.5 * cm, 4 * cm, 4 * cm, 3 * cm])]
        story += img(os.path.join(os.path.dirname(p), "double_descent.png"), W,
                     "Figure: model-wise (left, middle) and epoch-wise (right) test error.")
    for p in sorted(glob.glob(os.path.join(a.results, "*", "sweep_results.json"))):
        found = True
        r = load(p); ar = r["args"]
        story += [P(f"Tree-wise sweep on {ar['dataset']} features ({ar['trunk']} trunk, {ar['n_train']} images, "
                    f"{int(100 * ar['noise'])}% label noise): every node has the same fixed hidden width W, trees are grown to "
                    f"interpolation; a single MLP of width W on the same features is the reference.")]
        rows = [["W", "nodes", "params", "train acc", "test acc", "root-only test acc"]] + [
            [t["width"], t["n_nodes"], t["params"], pct(t["train_acc"]), pct(t["test_acc"]), pct(t["root_test_acc"])]
            for t in r["tree"]]
        story += [table(rows, [1.5 * cm, 2 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 3.5 * cm])]
        story += img(os.path.join(os.path.dirname(p), "tree_sweep.png"), W)
    if not found:
        story += [P("Double-descent experiments not run yet.", TODO)]

    # ------------------------------------------------------------ 4 receptive fields
    story += [P("4. Receptive fields of tree nodes", H1),
              P("For each node, the gradient receptive field (SmoothGrad |d logit / d pixel| averaged over the node's positive "
                "training samples), an activation-maximisation image and the top-activating positives are shown. "
                "Localisation statistics are averaged by depth.")]
    found = False
    for name, (d, r) in runs.items():
        sj = load(os.path.join(d, "rf", "rf_summary.json"))
        if not sj:
            continue
        found = True
        rows = [["level", "nodes", "frac. of pixels for 50% of |grad|", "centre offset", "normalised entropy"]] + [
            [k, v["n"], f"{v['frac_pixels_for_50pct']:.3f}", f"{v['centre_offset']:.3f}", f"{v['norm_entropy']:.3f}"]
            for k, v in sorted(sj["by_depth"].items(), key=lambda kv: (kv[0] != "root", kv[0]))]
        story += [P(f"Run {name}", H2), table(rows, [2.5 * cm, 1.8 * cm, 5.2 * cm, 2.6 * cm, 3 * cm])]
        story += img(os.path.join(d, "rf", "rf_vs_depth.png"), W * 0.8)
        for pg in sj["pages"][:2]:
            story += img(os.path.join(d, "rf", os.path.basename(pg)), W,
                         "Per node: top-activating positives | gradient RF | activation maximisation.")
    if not found:
        story += [P("Receptive-field analysis not run yet.", TODO)]

    # ------------------------------------------------------------ 5 interpretations
    story += [PageBreak(), P("5. Interpretations and conclusions", H1)]
    if os.path.exists(a.interpretation):
        story += md_to_flow(open(a.interpretation).read())
    else:
        print(f"WARNING: {a.interpretation} not found - the interpretation section is a placeholder.")
        story += [P("[Interpretation not written yet. Write interpretation.md after inspecting the tables and figures above; "
                    "see interpretation_template.md for the questions to answer.]", TODO)]

    # ------------------------------------------------------------ 6 future
    story += [P("6. Future directions", H1)]
    fut = [
        "<b>Joint training.</b> The trunk is frozen. Fine-tuning the trunk together with the tree (hard step replaced by a "
        "straight-through estimator for the child outputs) could let the features adapt to the repair structure.",
        "<b>Calibrated, not closed-form, weights.</b> Closed-form weights assume a child is perfect. A child that fires wrongly on "
        "a test sample gets a large weight; learning the weights with a margin/hinge objective (eqs. 6&ndash;9 of the paper) or "
        "using soft gates should improve generalisation.",
        "<b>Tree as a label-noise detector.</b> Repair depth flags mislabelled training samples (section 2). Using it to "
        "discard or relabel samples, then regrowing, is a natural data-cleaning loop.",
        "<b>Smaller multiclass trees.</b> Share child networks across classes (a single multi-output repair network per node), "
        "use error-correcting output codes, or merge children with similar receptive fields.",
        "<b>Compression with fine-tuning.</b> Iterative magnitude pruning or L0 regularisation of node weights with "
        "re-optimisation instead of post-hoc pruning; node-width search with a global parameter budget.",
        "<b>Per-node features and feature selection.</b> Give each subtree its own trunk or an L1/zero-norm selected subset of "
        "features (open question 5 of the original note).",
        "<b>Theory.</b> Bound the number of nodes by margin / separability of the masked problems, relate depth to the "
        "complexity of the label noise, and explain the tree-wise test-error curve with the same tools as double descent.",
    ]
    story += [P("&bull; " + f) for f in fut]

    # ------------------------------------------------------------ 7 reproducibility
    story += [P("7. Reproducibility", H1),
              P(f"Private repository: {a.repo_url}. Scripts: run_tree.py (growth, interpolation), double_descent.py, "
                f"tree_sweep.py, minimize.py, visualize_rf.py, make_report.py (this PDF). The Colab notebook "
                f"notebooks/colab_run.ipynb runs all experiments in order. Seeds and arguments are stored in each "
                f"results.json."),
              P("References: [1] Martinelli et al., A pyramidal delayed perceptron, IEEE Trans. Circuits Syst. 1990. "
                "[2] Weston et al., Use of the zero-norm with linear models and kernel methods, JMLR 2003. "
                "[3] Jayadeva, Deb, Chandra, Binary classification by SVM based tree type neural network, IJCNN 2002. "
                "[4] Nakkiran et al., Deep double descent, ICLR 2020. [5] Jayadeva, A tree type neural network (course note).",
                CAP)]
    SimpleDocTemplate(a.out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=1.8 * cm, title="Tree-type networks of small neural networks",
                      author=a.author).build(story)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
