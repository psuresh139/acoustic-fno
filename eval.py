"""
Evaluation and visualisation script.

Three things this does:

1. In-distribution error  — measure relative L2 on the held-out test set at
   training resolution (64×64).

2. Zero-shot super-resolution — regenerate a test sample at 256×256 using the
   FDTD solver, feed it to the *unchanged* trained model, and compare.  No
   fine-tuning, no retraining.  The FNO predicts correctly because the Fourier
   transform is resolution-agnostic.

3. Wall-swap demo  — take one room geometry, move/add an obstacle, run the FNO
   and the FDTD solver side by side.  Shows that the model generalises to
   unseen geometries instantly while the solver has to re-integrate from scratch.

All figures are saved to figures/ as PNGs.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from generate_data import make_geometry, make_source_map, make_alpha_map, fdtd_2d
from model import FNO2D


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str) -> FNO2D:
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg    = ckpt["config"]
    model  = FNO2D(C_in=2, C_out=1, d_model=cfg["d_model"],
                   n_layers=cfg["n_layers"],
                   k_max_h=cfg["k_max"], k_max_w=cfg["k_max"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    diff  = pred - target
    denom = np.linalg.norm(target)
    return float(np.linalg.norm(diff) / max(denom, 1e-8))


def pressure_clim(field: np.ndarray) -> float:
    """Symmetric colour limit at 99th percentile of absolute value."""
    return float(np.percentile(np.abs(field), 99))


def save_comparison(
    title: str,
    fields: list[tuple[str, np.ndarray]],   # (label, 2-D array)
    mask: np.ndarray | None = None,
    out: str = "figures/comparison.png",
) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    vmax = max(pressure_clim(f) for _, f in fields)

    for ax, (label, field) in zip(axes, fields):
        im = ax.imshow(field, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       origin="lower", interpolation="bilinear")
        if mask is not None:
            ax.contour(mask, levels=[0.5], colors="k", linewidths=0.8)
        ax.set_title(label, fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─── evaluation routines ──────────────────────────────────────────────────────

def eval_indistribution(model, data_path: str, device: str, n_show: int = 3) -> None:
    print("\n── In-distribution evaluation ──")
    data    = torch.load(data_path, weights_only=True)
    inputs  = data["inputs"][-100:]    # last 100 samples as test set
    targets = data["targets"][-100:]

    errors = []
    with torch.no_grad():
        for i in range(len(inputs)):
            x = inputs[i:i+1].to(device)
            y = targets[i:i+1].to(device)
            p = model(x, mask=x[:, 0:1])
            errors.append(relative_l2(p[0, 0].cpu().numpy(),
                                      y[0, 0].cpu().numpy()))

    print(f"  Mean relative L2: {np.mean(errors):.4f}  "
          f"(std {np.std(errors):.4f}  max {np.max(errors):.4f})")

    # plot a few examples
    with torch.no_grad():
        for i in range(n_show):
            x = inputs[i:i+1].to(device)
            y = targets[i:i+1]
            p = model(x, mask=x[:, 0:1]).cpu()
            mask = x[0, 0].cpu().numpy()
            save_comparison(
                title=f"In-distribution sample {i}  (rel L2 = {errors[i]:.3f})",
                fields=[
                    ("FDTD ground truth", y[0, 0].numpy()),
                    ("FNO prediction",    p[0, 0].numpy()),
                    ("Residual",          p[0, 0].numpy() - y[0, 0].numpy()),
                ],
                mask=mask,
                out=f"figures/indist_sample{i}.png",
            )


def eval_superresolution(
    model,
    ckpt_path: str,
    device: str,
    H_train: int     = 64,
    H_super: int     = 256,
    n_steps: int     = 200,
    max_obstacles: int = 0,
    seed: int        = 99,
) -> None:
    """
    Generate one sample at both 64×64 and 256×256 using FDTD, then compare
    FNO predictions at both resolutions.  The model is unchanged between runs.
    max_obstacles should match what the model was trained on.
    """
    print(f"\n── Zero-shot super-resolution ({H_train}² → {H_super}²) ──")
    rng = np.random.default_rng(seed)

    for H, label in [(H_train, "train_res"), (H_super, "super_res")]:
        # Scale steps so physical time T = n_steps * dt is constant across resolutions.
        # dt ∝ dx ∝ 1/H, so to hold T fixed we need steps ∝ H.
        scaled_steps = int(n_steps * H / H_train)
        n_obs       = int(rng.integers(0, max_obstacles + 1)) if max_obstacles > 0 else 0
        mask        = make_geometry(H, H, n_obstacles=n_obs, rng=np.random.default_rng(seed))
        source, _   = make_source_map(H, H, mask, rng=np.random.default_rng(seed))
        pressure    = fdtd_2d(mask, source, n_steps=scaled_steps)

        x = torch.tensor(np.stack([mask, source])[None]).to(device)  # (1, 2, H, H)
        with torch.no_grad():
            pred = model(x, mask=x[:, 0:1])[0, 0].cpu().numpy()

        err = relative_l2(pred, pressure)
        print(f"  {H}×{H}  relative L2 = {err:.4f}")

        save_comparison(
            title=f"Super-resolution demo — {H}×{H}  steps={scaled_steps}  (rel L2 = {err:.3f})",
            fields=[
                ("FDTD ground truth", pressure),
                ("FNO prediction",    pred),
                ("Residual",          pred - pressure),
            ],
            mask=mask,
            out=f"figures/superres_{label}.png",
        )


def eval_geometry_swap(
    model,
    device: str,
    H: int           = 64,
    n_steps: int     = 200,
    max_obstacles: int = 0,
    seed: int        = 7,
) -> None:
    """
    Keep the source position fixed, swap in a different obstacle count, compare
    FNO versus FDTD on each geometry — without any retraining.
    Obstacle counts tested: 0 up to max_obstacles+1 (so Stage 1 just shows 0).
    """
    print("\n── Geometry-swap demo ──")
    obs_range = list(range(max_obstacles + 2))   # e.g. [0] for Stage 1, [0,1,2,3] for Stage 2

    for i, n_obs in enumerate(obs_range):
        mask   = make_geometry(H, H, n_obstacles=n_obs, rng=np.random.default_rng(seed + i))
        source, _ = make_source_map(H, H, mask, rng=np.random.default_rng(seed))
        gt     = fdtd_2d(mask, source, n_steps=n_steps)

        x = torch.tensor(np.stack([mask, source])[None]).to(device)
        with torch.no_grad():
            pred = model(x, mask=x[:, 0:1])[0, 0].cpu().numpy()

        err = relative_l2(pred, gt)
        print(f"  {n_obs} obstacle(s)  relative L2 = {err:.4f}")

        save_comparison(
            title=f"Geometry swap — {n_obs} obstacle(s)  (rel L2 = {err:.3f})",
            fields=[
                ("FDTD ground truth", gt),
                ("FNO prediction",    pred),
                ("Residual",          pred - gt),
            ],
            mask=mask,
            out=f"figures/geom_swap_obs{n_obs}.png",
        )


def eval_alpha_sweep(
    model,
    device: str,
    H: int          = 128,
    n_steps: int    = 400,
    max_alpha: float = 4.0,
    n_obs: int      = 1,
    seed: int       = 42,
) -> None:
    """
    Fix geometry and source, sweep α from 0 (fully reflective) to max_alpha
    (strongly absorptive).  Shows the FNO predicting the continuous transition
    from a live reverberant room to a dead anechoic room — the Stage 3 headline result.
    """
    print(f"\n── Alpha sweep (α = 0 → {max_alpha}, {n_obs} obstacle(s)) ──")
    rng  = np.random.default_rng(seed)
    mask = make_geometry(H, H, n_obstacles=n_obs, rng=rng)
    source, _ = make_source_map(H, H, mask, rng=rng)

    alphas = np.linspace(0, max_alpha, 6)
    n      = len(alphas)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))

    for col, alpha in enumerate(alphas):
        gt        = fdtd_2d(mask, source, n_steps=n_steps, alpha=alpha)
        alpha_map = make_alpha_map(H, H, alpha, max_alpha=max_alpha)

        x = torch.tensor(
            np.stack([mask, source, alpha_map])[None], dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            pred = model(x, mask=x[:, 0:1])[0, 0].cpu().numpy()

        err  = relative_l2(pred, gt)
        vmax = pressure_clim(gt)

        for row, (field, label) in enumerate([
            (gt,   f"FDTD  α={alpha:.1f}"),
            (pred, f"FNO   err={err:.2f}"),
        ]):
            ax = axes[row, col]
            ax.imshow(field, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      origin="lower", interpolation="bilinear")
            ax.contour(mask, levels=[0.5], colors="k", linewidths=0.6)
            ax.set_title(label, fontsize=9)
            ax.axis("off")

    fig.suptitle("Alpha sweep: reflective → absorptive", fontsize=13)
    plt.tight_layout()
    out = "figures/alpha_sweep.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",          type=str,   required=True)
    parser.add_argument("--data",          type=str,   required=True)
    parser.add_argument("--device",        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_steps",       type=int,   default=400)
    parser.add_argument("--H_train",       type=int,   default=128)
    parser.add_argument("--H_super",       type=int,   default=256)
    parser.add_argument("--max_obstacles", type=int,   default=0)
    parser.add_argument("--max_alpha",     type=float, default=0.0,
                        help="Max alpha used in training. >0 enables alpha sweep demo.")
    args = parser.parse_args()

    model = load_model(args.ckpt, args.device)
    print(f"Loaded model from {args.ckpt}  (device={args.device})")

    eval_indistribution(model, args.data, args.device)
    eval_superresolution(model, args.ckpt, args.device,
                         H_train=args.H_train, H_super=args.H_super,
                         n_steps=args.n_steps, max_obstacles=args.max_obstacles)
    eval_geometry_swap(model, args.device, H=args.H_train,
                       n_steps=args.n_steps, max_obstacles=args.max_obstacles)

    if args.max_alpha > 0:
        eval_alpha_sweep(model, args.device, H=args.H_train,
                         n_steps=args.n_steps, max_alpha=args.max_alpha,
                         n_obs=min(1, args.max_obstacles))

    print("\nAll figures saved to figures/")
