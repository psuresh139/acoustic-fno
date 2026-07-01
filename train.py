"""
Training script for AcousticFNO.

Loss: relative L2 loss  ‖u_pred - u_true‖₂ / ‖u_true‖₂
      averaged over the batch.

Multi-resolution training
─────────────────────────
Pass --data_lo and --data_hi to train on two resolutions simultaneously.
Each epoch alternates: odd epochs train on the low-res dataset, even epochs
on the high-res dataset.  Validation uses the high-res dataset only (that is
the harder, more informative signal).  Because the FNO's Fourier layers are
resolution-agnostic, the same weights process both grids — the model learns
that the wave operator is consistent across scales, which dramatically improves
zero-shot super-resolution at inference time.

Usage
─────
# Single-resolution (original behaviour)
python train.py --data data/variable_128.pt --out checkpoints/model.pt

# Multi-resolution
python generate_data.py --n_samples 2000 --H 64  --W 64  --n_steps 200 \
    --max_obstacles 3 --out data/variable_64.pt
python generate_data.py --n_samples 2000 --H 128 --W 128 --n_steps 400 \
    --max_obstacles 3 --out data/variable_128.pt
python train.py --data_lo data/variable_64.pt --data_hi data/variable_128.pt \
    --out checkpoints/multires.pt --k_max 32 --epochs 150
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import FNO2D


# ─── loss ─────────────────────────────────────────────────────────────────────

def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff  = (pred - target).flatten(1)
    denom = target.flatten(1).norm(dim=1).clamp(min=1e-8)
    return (diff.norm(dim=1) / denom).mean()


# ─── dataset helpers ──────────────────────────────────────────────────────────

def make_loaders(
    data_path: str,
    batch_size: int,
    train_frac: float,
) -> tuple[DataLoader, DataLoader, int, int]:
    data    = torch.load(data_path, weights_only=True)
    dataset = TensorDataset(data["inputs"], data["targets"])
    n_train = int(len(dataset) * train_frac)
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        n_train,
        n_val,
    )


def run_epoch(model, loader, n, device, optimiser=None):
    """One train or val pass. optimiser=None → eval mode."""
    training = optimiser is not None
    model.train(training)
    total = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, mask=x[:, 0:1])
            loss = relative_l2(pred, y)
            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total += loss.item() * len(x)
    return total / n


# ─── training loop ────────────────────────────────────────────────────────────

def train(
    data_path: str | None,
    data_lo_path: str | None,
    data_hi_path: str | None,
    out_path: str,
    d_model: int    = 32,
    n_layers: int   = 4,
    k_max: int      = 32,
    epochs: int     = 150,
    batch_size: int = 32,
    lr: float       = 1e-3,
    train_frac: float = 0.9,
    device: str     = "cpu",
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    multires = (data_lo_path is not None and data_hi_path is not None)

    if multires:
        lo_tr, lo_val, n_lo_tr, n_lo_val = make_loaders(data_lo_path, batch_size, train_frac)
        hi_tr, hi_val, n_hi_tr, n_hi_val = make_loaders(data_hi_path, batch_size, train_frac)
        print(f"Multi-resolution training:")
        print(f"  low-res  {n_lo_tr} train / {n_lo_val} val")
        print(f"  high-res {n_hi_tr} train / {n_hi_val} val")
    else:
        tr, val, n_tr, n_val = make_loaders(data_path, batch_size, train_frac)
        print(f"Single-resolution: {n_tr} train / {n_val} val")

    model = FNO2D(C_in=2, C_out=1, d_model=d_model,
                  n_layers=n_layers, k_max_h=k_max, k_max_w=k_max).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    best_val  = float("inf")

    for epoch in range(1, epochs + 1):
        if multires:
            # Alternate resolutions each epoch so the model sees both scales
            if epoch % 2 == 1:
                tr_loss = run_epoch(model, lo_tr, n_lo_tr, device, optimiser)
            else:
                tr_loss = run_epoch(model, hi_tr, n_hi_tr, device, optimiser)
            # Validate on high-res — the harder, more informative signal
            val_loss = run_epoch(model, hi_val, n_hi_val, device)
        else:
            tr_loss  = run_epoch(model, tr,  n_tr,  device, optimiser)
            val_loss = run_epoch(model, val, n_val, device)

        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(),
                        "config": {"C_in": 3, "d_model": d_model,
                                   "n_layers": n_layers, "k_max": k_max}},
                       out_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs}  "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"best_val={best_val:.4f}")

    print(f"\nDone. Best val loss: {best_val:.4f}  →  {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # single-resolution mode
    parser.add_argument("--data",       type=str, default=None)
    # multi-resolution mode
    parser.add_argument("--data_lo",    type=str, default=None,
                        help="Low-resolution dataset (e.g. 64×64)")
    parser.add_argument("--data_hi",    type=str, default=None,
                        help="High-resolution dataset (e.g. 128×128)")
    parser.add_argument("--out",        type=str, default="checkpoints/model.pt")
    parser.add_argument("--d_model",    type=int, default=32)
    parser.add_argument("--n_layers",   type=int, default=4)
    parser.add_argument("--k_max",      type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.data is None and (args.data_lo is None or args.data_hi is None):
        parser.error("Provide either --data (single-res) or both --data_lo and --data_hi (multi-res)")

    print(f"Device: {args.device}")
    train(
        data_path=args.data,
        data_lo_path=args.data_lo,
        data_hi_path=args.data_hi,
        out_path=args.out,
        d_model=args.d_model,
        n_layers=args.n_layers,
        k_max=args.k_max,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
