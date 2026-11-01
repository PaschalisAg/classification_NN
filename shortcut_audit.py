"""Cluster inspection helpers: acquisition-style features, clustering and plots."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from scipy import ndimage
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from evaluation import make_head

MARKERS = ["o", "s", "^", "D", "P", "X"]


def acquisition_features(path, black_threshold=5, corner_fraction=0.15, n_bins=16):
    """Style statistics of the raw file that carry no lung-specific information."""
    with Image.open(path) as image:
        width, height = image.size
        gray = np.asarray(image.convert("L"), dtype=np.float32)

    black = gray < black_threshold
    # Fraction of each border strip (5% of the side) that is pure black.
    strip_h, strip_w = max(1, height // 20), max(1, width // 20)
    borders = [
        black[:strip_h].mean(), black[-strip_h:].mean(),
        black[:, :strip_w].mean(), black[:, -strip_w:].mean(),
    ]

    ch, cw = int(height * corner_fraction), int(width * corner_fraction)
    corners = [gray[:ch, :cw], gray[:ch, -cw:], gray[-ch:, :cw], gray[-ch:, -cw:]]
    corner_means = [c.mean() for c in corners]
    corner_stds = [c.std() for c in corners]

    gx, gy = ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0)
    edge_magnitude = np.hypot(gx, gy)
    histogram = np.histogram(gray, bins=n_bins, range=(0, 255), density=True)[0]

    return np.concatenate([
        [width, height, gray.mean(), gray.std(), black.mean()],
        borders, corner_means, corner_stds,
        [edge_magnitude.mean(), edge_magnitude.std()],
        histogram,
    ]).astype(np.float32)


def acquisition_feature_names(n_bins=16):
    return (
        ["width", "height", "mean", "std", "black_fraction"]
        + [f"border_{side}" for side in ("top", "bottom", "left", "right")]
        + [f"corner_mean_{c}" for c in ("tl", "tr", "bl", "br")]
        + [f"corner_std_{c}" for c in ("tl", "tr", "bl", "br")]
        + ["edge_mean", "edge_std"]
        + [f"hist_{i}" for i in range(n_bins)]
    )


def out_of_fold_predictions(X, y, n_splits=5, seed=42):
    """Predictions from heads that never saw the image, so errors are visible."""
    predictions = np.zeros(len(y), dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y):
        head = make_head().fit(X[train_idx], y[train_idx])
        predictions[test_idx] = head.predict(X[test_idx])
    return predictions


def build_cluster_space(embeddings, style, style_weight=1.0, n_components=20, seed=42):
    """Standardise each block, give blocks equal total variance, then PCA-reduce."""
    blocks = []
    for block, weight in ((embeddings, 1.0), (style, style_weight)):
        scaled = StandardScaler().fit_transform(block)
        blocks.append(scaled * weight / np.sqrt(block.shape[1]))
    joint = np.hstack(blocks)
    n_components = min(n_components, joint.shape[0] - 1, joint.shape[1])
    return PCA(n_components=n_components, random_state=seed).fit_transform(joint)


def run_clusterings(space, n_agglomerative=6, min_cluster_size=8):
    """Agglomerative (Ward, fixed k) and HDBSCAN (density, may mark noise as -1)."""
    return {
        "Agglomerative": AgglomerativeClustering(
            n_clusters=n_agglomerative, linkage="ward"
        ).fit_predict(space),
        "HDBSCAN": HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(space),
    }


def cluster_summary(cluster_labels, y_true, y_pred, class_names):
    """Per-cluster size, class mix and error rate."""
    table = pd.DataFrame({
        "cluster": cluster_labels,
        "true": [class_names[i] for i in y_true],
        "error": y_pred != y_true,
    })
    mix = pd.crosstab(table["cluster"], table["true"]).reindex(
        columns=class_names, fill_value=0
    )
    mix["n"] = mix.sum(axis=1)
    mix["purity"] = mix[class_names].max(axis=1) / mix["n"]
    mix["error_rate"] = table.groupby("cluster")["error"].mean()
    return mix.round(3)


def plot_clusters(coords, y_true, y_pred, class_names, cluster_sets, title_prefix,
                  true_colors=None, cluster_ids=None):
    """Colour = real class, marker = predicted class; one panel per clustering."""
    true_colors = true_colors or ["tab:red", "tab:blue", "tab:green"]
    fig, axes = plt.subplots(1, len(cluster_sets), figsize=(8 * len(cluster_sets), 7),
                             squeeze=False)
    for ax, (name, labels) in zip(axes[0], cluster_sets.items()):
        for pred_class, marker in enumerate(MARKERS[: len(class_names)]):
            for true_class, color in enumerate(true_colors[: len(class_names)]):
                mask = (y_pred == pred_class) & (y_true == true_class)
                if mask.any():
                    ax.scatter(
                        coords[mask, 0], coords[mask, 1], c=color, marker=marker,
                        s=70, edgecolors="black", linewidths=0.6, alpha=0.85,
                    )
        for cluster in np.unique(labels):
            members = labels == cluster
            cx, cy = np.median(coords[members], axis=0)
            label = "noise" if cluster == -1 else f"C{cluster}"
            ax.annotate(label, (cx, cy), fontsize=12, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
        ax.set_title(f"{title_prefix}: {name} ({len(set(labels) - {-1})} clusters"
                     + (f", {int((labels == -1).sum())} noise)" if -1 in labels else ")"))
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [Line2D([], [], marker="s", ls="", color=c, label=f"true: {n}")
               for n, c in zip(class_names, true_colors)]
    handles += [Line2D([], [], marker=m, ls="", color="gray", markeredgecolor="black",
                       label=f"pred: {n}")
                for n, m in zip(class_names, MARKERS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def embed_2d(space, seed=42, perplexity=20):
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=seed).fit_transform(space)


def agreement(cluster_sets):
    names = list(cluster_sets)
    return adjusted_rand_score(cluster_sets[names[0]], cluster_sets[names[1]])
