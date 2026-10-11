# Copy this file to interpretation.md and replace every bracketed prompt with what YOUR results show.
# make_report.py renders interpretation.md as section 5 of the PDF. Delete these comment lines.

## Interpolation
[Did the tree reach 100% (or near-100%) training accuracy on CIFAR-10 and Cats vs Dogs? With and without label noise? If a run stopped short, which limit stopped it (max_depth, max_nodes, min_pos, node training steps)?]
[How much did the root node alone solve, and how much did the tree add? How do nodes per depth and width histograms look?]

## Generalisation and double descent
[Model-wise curve (double_descent.py): where is the interpolation threshold (train error reaches 0), is there a test-error peak there, and does test error fall again for wider models? Say clearly if you did NOT see a peak, and what noise level / epochs / size you used.]
[Epoch-wise curve: does test error rise and fall again for the mid-sized widths?]
[Tree-wise sweep (tree_sweep.py): how do test error and #parameters/#nodes change with node width W? Compare with the single wide MLP.]
[Why does the tree generalise worse than the root node alone when labels are noisy? (Compare root-only and full-tree test accuracy.)]

## Receptive fields
[What do root-class nodes look at (object, background, colour)? Do repair nodes (depth >= 1) look at different, more local or more idiosyncratic regions? Quote the localisation statistics table. Do repair nodes for noisy samples look like anything meaningful?]

## Size minimisation
[How small can the tree get before train/test accuracy drops? Compare min-width search vs fixed width, pruning small children, and sparsification. Which lever gives the best test accuracy per parameter?]

## Repair depth as a noise detector
[AUROC and the flipped-label fraction per repair depth from the table in section 2.]

## Conclusions
[3-5 sentences: what you learned, what surprised you, limitations (single seed? limited training subset? frozen trunk?).]
