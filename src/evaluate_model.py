# evaluate_model.py
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(
    model,
    dataset,
    classes: list[str],
    output_dir: str | Path = "./output/eda",
    prefix: str = "model",
    dpi: int = 500,
) -> pd.DataFrame:
    """Evaluate a Keras model and plot/save a confusion matrix.

    Parameters
    ----------
    model:
        Trained Keras model.
    dataset:
        tf.data.Dataset yielding (images, labels) batches. Must not be shuffled.
    classes:
        Ordered list of class names matching the integer label IDs.
    output_dir:
        Directory to save the confusion matrix plot. Created if it does not exist.
    prefix:
        Filename prefix, e.g. "baseline" → "baseline_confusion_matrix.webp".
    dpi:
        Resolution of the saved image.

    Returns
    -------
    pd.DataFrame
        Per-class precision, recall, f1-score and support.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_labels, predictions = _get_predictions(model, dataset)
    class_metrics = _build_classification_report(true_labels, predictions, classes)
    _plot_confusion_matrix(
        true_labels=true_labels,
        predictions=predictions,
        classes=classes,
        title=f"{prefix.capitalize()} Model: Validation Confusion Matrix",
        save_path=output_dir / f"{prefix}_confusion_matrix.webp",
        dpi=dpi,
    )
    plt.show()

    return class_metrics


# --- private helpers ---

def _get_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    true_labels = np.concatenate([labels.numpy() for _, labels in dataset])
    probabilities = model.predict(dataset, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    return true_labels, predictions


def _build_classification_report(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    classes: list[str],
) -> pd.DataFrame:
    report = classification_report(
        true_labels,
        predictions,
        labels=np.arange(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    return (
        pd.DataFrame(report)
        .T.loc[classes, ["precision", "recall", "f1-score", "support"]]
    )


def _plot_confusion_matrix(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    classes: list[str],
    title: str,
    save_path: Path,
    dpi: int,
) -> None:
    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=np.arange(len(classes)),
    )
    fig, ax = plt.subplots()
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)


# --- usage ---
if __name__ == "__main__":
    import tensorflow as tf

    classes = ["Covid", "Normal class", "Pneumonia"]
    baseline_model = tf.keras.models.load_model("models/baseline/baseline_v1.keras")

    class_metrics = evaluate_model(
        model=baseline_model,
        dataset=val_ds,
        classes=classes,
        prefix="baseline",
    )
    display(class_metrics)