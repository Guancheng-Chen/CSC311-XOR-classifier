"""画图：decision boundary 的单张快照，以及把快照合成 GIF。"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

GRID_MIN, GRID_MAX, GRID_N = -0.5, 1.5, 200


def plot_boundary(model, X, Y, step, loss, path):
    """Save the model's output over the input plane, with the y = 0.5 boundary."""
    axis = np.linspace(GRID_MIN, GRID_MAX, GRID_N)
    xx, yy = np.meshgrid(axis, axis)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        zz = model(grid).reshape(xx.shape).numpy()
        preds = (model(X) > 0.5).float()
    n_correct = int((preds == Y).sum())

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=80)
    ax.contourf(xx, yy, zz, levels=np.linspace(-0.5, 1.5, 21), cmap="RdBu_r",
                extend="both", alpha=0.85)
    if zz.min() < 0.5 < zz.max():
        ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)
    for label, marker, color in ((0, "o", "tab:blue"), (1, "^", "tab:red")):
        mask = Y.squeeze() == label
        ax.scatter(X[mask, 0], X[mask, 1], s=160, marker=marker, c=color,
                   edgecolors="white", linewidths=2, label=f"class {label}", zorder=3)
    ax.set_xlim(GRID_MIN, GRID_MAX)
    ax.set_ylim(GRID_MIN, GRID_MAX)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.set_title(f"step {step}   loss {loss:.4f}   correct {n_correct}/{len(X)}")
    ax.legend(loc="upper center", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_gif(frame_paths, path, duration_ms=60, hold_last_ms=2000):
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    durations = [duration_ms] * (len(frames) - 1) + [hold_last_ms]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)
