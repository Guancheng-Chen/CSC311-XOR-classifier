"""Train a 2-2-1 ReLU network on XOR and snapshot the decision boundary at every update."""

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

GRID_MIN, GRID_MAX, GRID_N = -0.5, 1.5, 200


class XORNet(nn.Module):
    """h = ReLU(W1 x + b1), y = W2 h + b2 (no output activation)."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))


def plot_boundary(model, step, loss, path):
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
    ax.set_title(f"step {step}   loss {loss:.4f}   correct {n_correct}/4")
    ax.legend(loc="upper center", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_gif(frame_paths, path, duration_ms=60, hold_last_ms=2000):
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    durations = [duration_ms] * (len(frames) - 1) + [hold_last_ms]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)


def train(seed, lr, max_steps, tol, image_dir):
    torch.manual_seed(seed)
    model = XORNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if image_dir is not None:
        shutil.rmtree(image_dir, ignore_errors=True)
        (image_dir / "steps").mkdir(parents=True)

    frame_paths = []

    def snapshot(step, loss):
        if image_dir is None:
            return
        frame_path = image_dir / "steps" / f"step_{step:04d}.png"
        plot_boundary(model, step, loss, frame_path)
        frame_paths.append(frame_path)

    loss = loss_fn(model(X), Y)
    snapshot(0, loss.item())
    step = 0
    while step < max_steps and loss.item() > tol:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        loss = loss_fn(model(X), Y)
        snapshot(step, loss.item())

    if image_dir is not None:
        shutil.copy(frame_paths[-1], image_dir / "final.png")
        make_gif(frame_paths, image_dir / "training.gif")
    return model, step, loss.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="stop once the MSE loss drops below this value")
    parser.add_argument("--image-dir", type=Path, default=Path(__file__).parent / "image")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    model, step, loss = train(args.seed, args.lr, args.max_steps, args.tol,
                              None if args.no_images else args.image_dir)

    torch.set_printoptions(precision=4, sci_mode=False)
    print(f"seed {args.seed}: stopped after {step} updates, loss {loss:.6f}")
    print("W1 =", model.hidden.weight.data)
    print("b1 =", model.hidden.bias.data)
    print("W2 =", model.output.weight.data)
    print("b2 =", model.output.bias.data)
    with torch.no_grad():
        hidden = torch.relu(model.hidden(X))
        out = model(X)
    for x, h, o, y in zip(X, hidden, out, Y):
        print(f"x = {x.tolist()}  h = {[round(v, 4) for v in h.tolist()]}  "
              f"y = {o.item():.4f}  target = {int(y.item())}")


if __name__ == "__main__":
    main()
