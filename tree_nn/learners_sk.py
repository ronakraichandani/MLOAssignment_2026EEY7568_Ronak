"""scikit-learn node learners.  Used for unit tests and quick CPU experiments."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class SkFitted:
    def __init__(self, model, kind, n_out, width, n_params, const=None):
        self.model, self.kind, self.n_out = model, kind, n_out
        self.width, self.n_params, self.const = width, n_params, const

    def logits(self, F):
        if self.const is not None:               # degenerate: only one class in the mask
            c = np.full((len(F),), self.const) if self.kind == "binary" else None
            if self.kind == "binary":
                return c
            out = np.full((len(F), self.n_out), -10.0)
            out[:, self.const] = 10.0
            return out
        if isinstance(self.model, LogisticRegression):
            z = self.model.decision_function(F)
        else:
            p = np.clip(self.model.predict_proba(F), 1e-9, 1 - 1e-9)
            z = np.log(p[:, 1] / p[:, 0]) if self.kind == "binary" else np.log(p)
        if self.kind == "multi":
            full = np.full((len(F), self.n_out), -10.0)
            full[:, self.model.classes_] = z if z.ndim == 2 else np.stack([-z, z], 1)
            return full
        return z


class SkLearner:
    def __init__(self, widths=(0,), C=1e3, max_iter=300, seed=0):
        self.widths, self.C, self.max_iter, self.seed = widths, C, max_iter, seed

    def fit(self, F, t, mask, kind, n_out):
        X, y = F[mask], t[mask]
        if len(np.unique(y)) == 1:               # nothing to separate
            c = int(y[0]) if kind == "multi" else (1.0 if y[0] == 1 else -1.0)
            return SkFitted(None, kind, n_out, 0, 0, const=c)
        best, best_acc = None, -1
        for w in self.widths:
            if w == 0:
                m = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                       class_weight="balanced" if kind == "binary" else None)
                n_par = (X.shape[1] + 1) * (1 if kind == "binary" else n_out)
            else:
                m = MLPClassifier((w,), max_iter=self.max_iter, random_state=self.seed)
                n_par = X.shape[1] * w + w + w * (1 if kind == "binary" else n_out) + n_out
            m.fit(X, y)
            acc = float((m.predict(X) == y).mean())
            cand = SkFitted(m, kind, n_out, w, n_par)
            if acc > best_acc:
                best, best_acc = cand, acc
            if acc >= 1.0:
                break
        return best
