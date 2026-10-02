"""Train a 2-2-1 ReLU network on XOR and snapshot the decision boundary at every update."""

import argparse
import shutil
from pathlib import Path

import torch

from data import make_xor_data
from model import XORNet
from plot import make_gif, plot_boundary
from train import train


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="stop once the MSE loss drops below this value")
    parser.add_argument("--image-dir", type=Path, default=Path(__file__).parent / "image")
    parser.add_argument("--images", action="store_true",
                        help="save a decision boundary snapshot at every update "
                             "(clears and regenerates --image-dir)")
    return parser.parse_args()


def report(model, X, Y, seed, step, loss):
    torch.set_printoptions(precision=4, sci_mode=False)
    print(f"seed {seed}: stopped after {step} updates, loss {loss:.6f}")
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


def main():
    args = parse_args()

    # 1. 数据
    X, Y = make_xor_data()

    # 2. 模型（随机初始化，seed 只决定随机数，不预设权重）
    torch.manual_seed(args.seed)
    model = XORNet()

    # 3. 画图（默认关闭，加 --images 才开）：每次参数更新后存一张 decision boundary
    frame_paths = []
    on_step = None
    if args.images:
        shutil.rmtree(args.image_dir, ignore_errors=True)
        (args.image_dir / "steps").mkdir(parents=True)

        def on_step(step, loss):
            frame_path = args.image_dir / "steps" / f"step_{step:04d}.png"
            plot_boundary(model, X, Y, step, loss, frame_path)
            frame_paths.append(frame_path)

    # 4. 训练
    step, loss = train(model, X, Y, lr=args.lr, max_steps=args.max_steps,
                       tol=args.tol, on_step=on_step)

    # 5. 收尾：最终图、GIF、打印结果
    if frame_paths:
        shutil.copy(frame_paths[-1], args.image_dir / "final.png")
        make_gif(frame_paths, args.image_dir / "training.gif")
    report(model, X, Y, args.seed, step, loss)


if __name__ == "__main__":
    main()
