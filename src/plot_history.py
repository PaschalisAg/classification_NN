# plot_history.py
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(
    history,
    output_dir: str | Path = "./output/eda",
    prefix: str = "model",
    dpi: int = 500,
) -> None:
    """Plot and save training/validation accuracy and loss curves.

    Parameters
    ----------
    history:
        Keras History object returned by model.fit().
    output_dir:
        Directory to save the plots. Created if it does not exist.
    prefix:
        Filename prefix, e.g. "baseline" → "baseline_accuracy.webp".
    dpi:
        Resolution of the saved images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.history["accuracy"]) + 1)

    plot_metric(
        epochs=epochs,
        train=history.history["accuracy"],
        val=history.history["val_accuracy"],
        title="Training and validation accuracy",
        ylabel="Accuracy",
        save_path=output_dir / f"{prefix}_accuracy.webp",
        dpi=dpi,
    )

    plot_metric(
        epochs=epochs,
        train=history.history["loss"],
        val=history.history["val_loss"],
        title="Training and validation loss",
        ylabel="Loss",
        save_path=output_dir / f"{prefix}_loss.webp",
        dpi=dpi,
    )

    plt.show()


def plot_metric(epochs, train, val, title, ylabel, save_path, dpi):
    fig, ax = plt.subplots()
    ax.plot(epochs, train, "bo", label=f"Training {ylabel.lower()}")
    ax.plot(epochs, val,   "r",  label=f"Validation {ylabel.lower()}")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    # plt.close(fig)  # free memory before the next plot


# --- usage ---
if __name__ == "__main__":
    plot_training_history(history, prefix="baseline")