"""Evaluation helpers: cross-validation and bootstrap confidence intervals."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_head():
    """Linear softmax head on top of frozen embeddings (scaler fitted per fit call)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def compute_metrics(y_true, probabilities, class_names):
    """Overall and per-class metrics from class probabilities."""
    y_pred = probabilities.argmax(axis=1)
    labels = np.arange(len(class_names))
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels),
        "macro_auc": roc_auc_score(
            y_true, probabilities, multi_class="ovr", labels=labels
        ),
    }
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for name, recall in zip(class_names, recalls):
        metrics[f"recall_{name}"] = recall
    return metrics


def cross_validate(X, y, class_names, n_splits=5, n_repeats=3, seed=42):
    """Repeated stratified CV of the linear head; returns one row per fold."""
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    rows = []
    for train_idx, val_idx in splitter.split(X, y):
        head = make_head().fit(X[train_idx], y[train_idx])
        probabilities = head.predict_proba(X[val_idx])
        rows.append(compute_metrics(y[val_idx], probabilities, class_names))
    return pd.DataFrame(rows)


def bootstrap_ci(y_true, probabilities, class_names, n_boot=2000, alpha=0.05, seed=42):
    """Point estimate and percentile CI, resampling images within each class."""
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y_true == c) for c in range(len(class_names))]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(i, size=len(i)) for i in class_indices])
        draws.append(compute_metrics(y_true[idx], probabilities[idx], class_names))
    draws = pd.DataFrame(draws)
    point = compute_metrics(y_true, probabilities, class_names)
    return pd.DataFrame(
        {
            "estimate": pd.Series(point),
            "ci_low": draws.quantile(alpha / 2),
            "ci_high": draws.quantile(1 - alpha / 2),
        }
    )


def format_ci(table):
    """Render a bootstrap table as 'estimate [low, high]' strings."""
    return table.apply(
        lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1
    )
