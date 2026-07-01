"""
FDTD solver for the 2D acoustic wave equation.

PDE:  d²p/dt² = c² (d²p/dx² + d²p/dy²)

Discretised with a second-order central-difference stencil (the classic
leap-frog FDTD update):

  p^{n+1}[i,j] = 2p^n[i,j] - p^{n-1}[i,j]
                 + λ²(p^n[i+1,j] + p^n[i-1,j] + p^n[i,j+1] + p^n[i,j-1] - 4p^n[i,j])

where λ = c·dt/dx  (Courant number; stable iff λ ≤ 1/√2 in 2D).

Boundary conditions
───────────────────
Outer walls support a Robin (impedance) BC:  ∂p/∂n + α·p = 0

Discretising the normal derivative with a one-sided difference gives:

  p_wall = p_neighbour / (1 + α·dx)

Special cases:
  α = 0          →  p_wall = p_neighbour   (Neumann, ∂p/∂n = 0, perfectly reflective)
  α → ∞          →  p_wall → 0             (Dirichlet, pressure-release, fully absorptive)

α is the wall impedance parameter (units 1/length with dx=1).
Physically meaningful range: α ∈ [0, 4].
  α·dx = 0   → reflection coefficient R = 1   (hard wall)
  α·dx = 1   → R = 0.5
  α·dx = 4   → R = 0.2
  α·dx → ∞  → R → 0   (anechoic)

Interior rectangular obstacles always use Dirichlet (rigid scatterers).

Dataset layout (each sample):
  inputs  shape (3, H, W) — channel 0: geometry mask (1=free, 0=wall/obstacle)
                          — channel 1: source map (Gaussian at source position)
                          — channel 2: alpha map (α value at outer wall cells, 0 elsewhere)
  target  shape (1, H, W) — pressure field at step T_STEPS
"""

import numpy as np
import torch
from pathlib import Path
import argparse


# ─── solver ───────────────────────────────────────────────────────────────────

def make_geometry(
    H: int,
    W: int,
    n_obstacles: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Binary mask: 1 = free space, 0 = wall.

    The outer boundary is always a hard wall (one-cell-thick border of 0s).
    n_obstacles rectangular blocks are placed randomly in the interior.
    Obstacles cannot overlap the source placement margin (8 cells from boundary).
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = np.ones((H, W), dtype=np.float32)
    mask[0, :] = mask[-1, :] = 0.0   # top/bottom walls
    mask[:, 0] = mask[:, -1] = 0.0   # left/right walls

    for _ in range(n_obstacles):
        # obstacle size: between 10% and 30% of domain in each dimension
        oh = rng.integers(max(2, H // 10), max(3, H // 3))
        ow = rng.integers(max(2, W // 10), max(3, W // 3))
        # placement: fully inside the domain (at least 2 cells from outer wall)
        r0 = rng.integers(2, H - oh - 2)
        c0 = rng.integers(2, W - ow - 2)
        mask[r0 : r0 + oh, c0 : c0 + ow] = 0.0

    return mask


def make_source_map(
    H: int,
    W: int,
    mask: np.ndarray,
    margin: int = 8,
    sigma: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Place a Gaussian pressure pulse at a random free-space position.
    Returns (source_map, (row, col)).
    """
    if rng is None:
        rng = np.random.default_rng()

    # candidate positions: interior cells that are free
    free = np.argwhere(mask[margin:-margin, margin:-margin] == 1.0)
    idx = rng.integers(len(free))
    sr, sc = free[idx] + margin   # shift back to full-domain coordinates

    y, x = np.mgrid[0:H, 0:W]
    source_map = np.exp(-((y - sr) ** 2 + (x - sc) ** 2) / (2 * sigma ** 2))
    source_map *= mask            # zero out any bleed into walls
    return source_map.astype(np.float32), (int(sr), int(sc))


def make_alpha_map(
    H: int,
    W: int,
    alpha: float,
    max_alpha: float = 4.0,
) -> np.ndarray:
    """
    A map with the normalised α value on outer wall cells, 0 elsewhere.
    Normalised to [0, 1] so the FNO sees a consistent input scale.
    Interior obstacles are not encoded here — they're already in the mask.
    """
    alpha_map = np.zeros((H, W), dtype=np.float32)
    alpha_norm = alpha / max_alpha
    alpha_map[0, :]  = alpha_norm   # top wall
    alpha_map[-1, :] = alpha_norm   # bottom wall
    alpha_map[:, 0]  = alpha_norm   # left wall
    alpha_map[:, -1] = alpha_norm   # right wall
    return alpha_map


def fdtd_2d(
    mask: np.ndarray,
    source_map: np.ndarray,
    n_steps: int,
    dx: float = 1.0,
    c: float = 1.0,
    alpha: float = 0.0,
) -> np.ndarray:
    """
    Run FDTD for n_steps and return the pressure field at the final step.

    alpha : Robin BC parameter on outer walls (0 = Neumann/reflective,
            large = Dirichlet/absorptive). Interior obstacles always Dirichlet.
    """
    dt   = 0.4 * dx / c
    lam2 = (c * dt / dx) ** 2
    denom = 1.0 + alpha * dx       # Robin BC denominator: p_wall = p_nbr / denom

    p_prev = np.zeros_like(mask)
    p_curr = source_map.copy()

    for _ in range(n_steps):
        lap = (
            p_curr[2:, 1:-1] + p_curr[:-2, 1:-1]
            + p_curr[1:-1, 2:] + p_curr[1:-1, :-2]
            - 4.0 * p_curr[1:-1, 1:-1]
        )
        p_next = np.zeros_like(p_curr)
        p_next[1:-1, 1:-1] = 2.0 * p_curr[1:-1, 1:-1] - p_prev[1:-1, 1:-1] + lam2 * lap

        # Dirichlet on interior obstacles (mask zeros them out)
        p_next *= mask

        # Robin BC on the four outer walls (overwrite the zeros left by mask)
        p_next[0, :]  = p_next[1, :]  / denom
        p_next[-1, :] = p_next[-2, :] / denom
        p_next[:, 0]  = p_next[:, 1]  / denom
        p_next[:, -1] = p_next[:, -2] / denom

        p_prev, p_curr = p_curr, p_next

    return p_curr.astype(np.float32)


# ─── dataset generation ───────────────────────────────────────────────────────

def generate_dataset(
    n_samples: int,
    H: int = 64,
    W: int = 64,
    n_steps: int = 200,
    max_obstacles: int = 0,
    max_alpha: float = 0.0,    # 0 = Dirichlet-only (Stages 1&2), >0 = Robin BC (Stage 3)
    seed: int = 42,
    out_path: str = "data/dataset.pt",
) -> None:
    """
    Generate n_samples (input, target) pairs and save as a .pt file.

    max_alpha > 0 enables Stage 3: each sample draws α ~ Uniform(0, max_alpha),
    encoding the wall absorption from perfectly reflective (α=0) to strongly
    absorptive (α=max_alpha).  Recommended max_alpha=4.0.
    """
    rng = np.random.default_rng(seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    inputs  = torch.zeros(n_samples, 3, H, W)   # [mask, source, alpha_map]
    targets = torch.zeros(n_samples, 1, H, W)

    for i in range(n_samples):
        n_obs  = int(rng.integers(0, max_obstacles + 1)) if max_obstacles > 0 else 0
        alpha  = float(rng.uniform(0, max_alpha)) if max_alpha > 0 else 0.0

        mask       = make_geometry(H, W, n_obstacles=n_obs, rng=rng)
        source_map, _ = make_source_map(H, W, mask, rng=rng)
        alpha_map  = make_alpha_map(H, W, alpha, max_alpha=max(max_alpha, 1e-6))
        pressure   = fdtd_2d(mask, source_map, n_steps=n_steps, alpha=alpha)

        inputs[i, 0]  = torch.from_numpy(mask)
        inputs[i, 1]  = torch.from_numpy(source_map)
        inputs[i, 2]  = torch.from_numpy(alpha_map)
        targets[i, 0] = torch.from_numpy(pressure)

        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  {i+1}/{n_samples} samples generated")

    torch.save({"inputs": inputs, "targets": targets}, out_path)
    print(f"Saved {n_samples} samples → {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",     type=int, default=1000)
    parser.add_argument("--H",             type=int, default=64)
    parser.add_argument("--W",             type=int, default=64)
    parser.add_argument("--n_steps",       type=int, default=200)
    parser.add_argument("--max_obstacles", type=int,   default=0)
    parser.add_argument("--max_alpha",     type=float, default=0.0,
                        help="Max wall absorption (0=Dirichlet only, 4.0=Stage 3)")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--out",           type=str,   default="data/dataset.pt")
    args = parser.parse_args()

    print(f"Generating {args.n_samples} samples  ({args.H}×{args.W}, {args.n_steps} steps, "
          f"max_obstacles={args.max_obstacles}, max_alpha={args.max_alpha}) ...")
    generate_dataset(
        n_samples=args.n_samples,
        H=args.H, W=args.W,
        n_steps=args.n_steps,
        max_obstacles=args.max_obstacles,
        max_alpha=args.max_alpha,
        seed=args.seed,
        out_path=args.out,
    )
