"""
Training script for AcousticFNO.

Loss: relative L2 loss  ‖u_pred - u_true‖₂ / ‖u_true‖₂
      averaged over the batch.  This normalises for samples where the
      wavefield amplitude varies (different source positions, geometries).

Usage
─────
# Phase 1 — fixed geometry, vary source position
python generate_data.py --n_samples 1000 --max_obstacles 0 --out data/fixed.pt
python train.py --data data/fixed.pt --out checkpoints/fixed.pt

# Phase 2 — variable geometry
python generate_data.py --n_samples 2000 --max_obstacles 3 --out data/variable.pt
python train.py --data data/variable.pt --out checkpoints/variable.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import FNO2D


# ─── loss ─────────────────────────────────────────────────────────────────────

def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean relative L2 loss over the batch."""
    diff  = (pred - target).flatten(1)           # (B, H*W)
    denom = target.flatten(1).norm(dim=1)        # (B,)
    denom = denom.clamp(min=1e-8)
    return (diff.norm(dim=1) / denom).mean()


# ─── training loop ────────────────────────────────────────────────────────────

def train(
    data_path: str,
    out_path: str,
    d_model: int   = 32,
    n_layers: int  = 4,
    k_max: int     = 16,
    epochs: int    = 100,
    batch_size: int = 32,
    lr: float      = 1e-3,
    train_frac: float = 0.9,
    device: str    = "cpu",
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # ── data ──
    data   = torch.load(data_path, weights_only=True)
    inputs  = data["inputs"]    # (N, 2, H, W)
    targets = data["targets"]   # (N, 1, H, W)
    dataset = TensorDataset(inputs, targets)

    n_train = int(len(dataset) * train_frac)
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"Dataset: {len(dataset)} samples  ({n_train} train / {n_val} val)")

    # ── model ──
    model = FNO2D(C_in=2, C_out=1, d_model=d_model,
                  n_layers=n_layers, k_max_h=k_max, k_max_w=k_max).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask    = x_batch[:, 0:1]          # geometry mask channel

            pred = model(x_batch, mask=mask)
            loss = relative_l2(pred, y_batch)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(x_batch)

        train_loss /= n_train

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                mask    = x_batch[:, 0:1]
                pred    = model(x_batch, mask=mask)
                val_loss += relative_l2(pred, y_batch).item() * len(x_batch)
        val_loss /= n_val

        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(),
                        "config": {"d_model": d_model, "n_layers": n_layers,
                                   "k_max": k_max}},
                       out_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"best_val={best_val:.4f}")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoint saved → {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str, required=True)
    parser.add_argument("--out",        type=str, default="checkpoints/model.pt")
    parser.add_argument("--d_model",    type=int, default=32)
    parser.add_argument("--n_layers",   type=int, default=4)
    parser.add_argument("--k_max",      type=int, default=16)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    train(
        data_path=args.data,
        out_path=args.out,
        d_model=args.d_model,
        n_layers=args.n_layers,
        k_max=args.k_max,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
