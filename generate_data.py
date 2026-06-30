"""
FDTD solver for the 2D acoustic wave equation.

PDE:  d²p/dt² = c² (d²p/dx² + d²p/dy²)

Discretised with a second-order central-difference stencil (the classic
leap-frog FDTD update):

  p^{n+1}[i,j] = 2p^n[i,j] - p^{n-1}[i,j]
                 + λ²(p^n[i+1,j] + p^n[i-1,j] + p^n[i,j+1] + p^n[i,j-1] - 4p^n[i,j])

where λ = c·dt/dx  (Courant number; stable iff λ ≤ 1/√2 in 2D).

Geometry is encoded as a binary mask (1 = free space, 0 = wall/obstacle).
Dirichlet BC p=0 at walls is enforced by zeroing masked cells each step.
Interior rectangular obstacles use the same mechanism, so the transition
from a clean rectangular room to a room-with-obstacles costs exactly one
extra multiply.

Dataset layout (each sample):
  inputs  shape (2, H, W) — channel 0: geometry mask
                          — channel 1: source map (Gaussian at source position)
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


def fdtd_2d(
    mask: np.ndarray,
    source_map: np.ndarray,
    n_steps: int,
    dx: float = 1.0,
    c: float = 1.0,
) -> np.ndarray:
    """
    Run FDTD for n_steps and return the pressure field at the final step.

    Initial condition: p^0 = source_map, p^{-1} = 0  (sharp pulse).
    Courant number λ = c·dt/dx is set to 0.4 (well below 1/√2 ≈ 0.707).
    """
    dt = 0.4 * dx / c
    lam2 = (c * dt / dx) ** 2   # λ²

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
        p_next *= mask            # enforce Dirichlet BC on walls + obstacles
        p_prev, p_curr = p_curr, p_next

    return p_curr.astype(np.float32)


# ─── dataset generation ───────────────────────────────────────────────────────

def generate_dataset(
    n_samples: int,
    H: int = 64,
    W: int = 64,
    n_steps: int = 200,
    max_obstacles: int = 0,   # 0 = fixed-geometry (Phase 1)
    seed: int = 42,
    out_path: str = "data/dataset.pt",
) -> None:
    """
    Generate n_samples (input, target) pairs and save as a .pt file.

    With max_obstacles=0 every sample is the same rectangular room; only the
    source position varies.  Set max_obstacles>0 for variable-geometry mode.
    """
    rng = np.random.default_rng(seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    inputs  = torch.zeros(n_samples, 2, H, W)   # [mask, source]
    targets = torch.zeros(n_samples, 1, H, W)   # [pressure at T]

    for i in range(n_samples):
        n_obs = int(rng.integers(0, max_obstacles + 1)) if max_obstacles > 0 else 0
        mask       = make_geometry(H, W, n_obstacles=n_obs, rng=rng)
        source_map, _ = make_source_map(H, W, mask, rng=rng)
        pressure   = fdtd_2d(mask, source_map, n_steps=n_steps)

        inputs[i, 0]  = torch.from_numpy(mask)
        inputs[i, 1]  = torch.from_numpy(source_map)
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
    parser.add_argument("--max_obstacles", type=int, default=0,
                        help="0=fixed room, >0=variable geometry")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--out",           type=str, default="data/dataset.pt")
    args = parser.parse_args()

    print(f"Generating {args.n_samples} samples  ({args.H}×{args.W}, {args.n_steps} steps, "
          f"max_obstacles={args.max_obstacles}) ...")
    generate_dataset(
        n_samples=args.n_samples,
        H=args.H, W=args.W,
        n_steps=args.n_steps,
        max_obstacles=args.max_obstacles,
        seed=args.seed,
        out_path=args.out,
    )
