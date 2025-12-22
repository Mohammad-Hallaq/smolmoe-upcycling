# src/moe_llm/metrics.py
from typing import Dict, Sequence, Optional
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import detach_metrics


def plot_metrics(
    metrics: Dict[str, Sequence[float]],
    x_vals: Sequence[float] | None = None,
    suptitle: str = "Training Metrics",
    save_dir: Optional[str | Path] = None,
    filename: str = "metrics.png",
    show: bool = True,
) -> None:
    """
    Plot a collection of metrics as line plots on a single row.

    Parameters
    ----------
    metrics : dict
        Mapping from metric name to a sequence of values.
    x_vals : sequence, optional
        Shared x-axis values; defaults to 1..len(metric).
    suptitle : str
        Figure-level title.
    save_dir : str or Path, optional
        Directory where the plot will be saved. If None, the plot is not saved.
    filename : str
        Name of the saved figure file.
    show : bool
        Whether to display the plot interactively (plt.show()).
    """
    metrics = detach_metrics(metrics)

    keys = list(metrics.keys())
    n = len(keys)
    if n == 0:
        return

    length = len(next(iter(metrics.values())))
    if x_vals is None:
        x_vals = list(range(1, length + 1))

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), constrained_layout=True)
    if n == 1:
        axes = [axes]

    palette = plt.cm.tab10.colors

    for i, (ax, key_str) in enumerate(zip(axes, keys)):
        y_vals = metrics[key_str]
        ax.plot(x_vals, y_vals, marker="o", color=palette[i % len(palette)])
        ax.set_title(key_str)
        ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle)
    fig.supxlabel("Steps")

    # --- Save to disk if requested ---
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        fig.savefig(save_path, dpi=150)
        print(f"[plot_metrics] Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
