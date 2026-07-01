"""
Fourier Neural Operator (FNO) for 2D scalar fields.

Reference: Li et al. 2020  "Fourier Neural Operator for Parametric PDEs"
           https://arxiv.org/abs/2010.08895

Architecture overview
─────────────────────
                     ┌──────────────────────────────────────────┐
  a(x) ──► Lift ──► │  FourierLayer × n_layers                 │ ──► Project ──► u(x)
  (C_in)   (d)      │                                           │      (d → C_out)
                     │  For each layer:                         │
                     │    y = σ( W·v  +  K(v) )                │
                     │        ↑           ↑                     │
                     │    pointwise   spectral conv             │
                     └──────────────────────────────────────────┘

Spectral convolution K(v):
  1.  v̂ = rfft2(v)                   — real FFT, shape (..., H, W//2+1)
  2.  Truncate to k_max modes in each frequency axis
  3.  Multiply by learned complex weight tensor R  (shape d × d × k_max × k_max)
  4.  Pad back to original frequency grid
  5.  irfft2 → spatial domain

Key property: because the operator is parameterised in frequency space and
rfft2/irfft2 are resolution-agnostic, a model trained at 64×64 can be
evaluated at any larger resolution without retraining  (zero-shot super-resolution).

Geometry / boundary handling
────────────────────────────
The geometry mask and source map are concatenated as input channels.  The
network therefore "sees" the room shape and obstacle layout directly.  No
special boundary-condition layer is needed for the Dirichlet case because the
mask channel carries that information, and the output can be post-multiplied
by the mask to enforce p=0 at walls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── spectral convolution ─────────────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    """
    Learns a linear map in the truncated Fourier domain.

    Parameters
    ----------
    in_channels, out_channels : channel widths (= d_model in the paper)
    k_max_h, k_max_w          : number of Fourier modes to keep per axis.
                                  Higher → more expressive, more parameters.
                                  Rule of thumb: k_max ≈ H // 4 for a H-grid.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k_max_h: int = 16,
        k_max_w: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.k_max_h = k_max_h
        self.k_max_w = k_max_w

        # Complex weights stored as two real tensors (real, imag parts).
        # Shape: (out_ch, in_ch, k_max_h, k_max_w)
        scale = 1.0 / (in_channels * out_channels)
        self.weights_re = nn.Parameter(
            scale * torch.randn(out_channels, in_channels, k_max_h, k_max_w)
        )
        self.weights_im = nn.Parameter(
            scale * torch.randn(out_channels, in_channels, k_max_h, k_max_w)
        )

    @property
    def weights(self) -> torch.Tensor:
        return torch.complex(self.weights_re, self.weights_im)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C_in, H, W)
        returns (B, C_out, H, W)
        """
        B, C, H, W = x.shape

        # 1. Real FFT2 — rfft2 returns shape (B, C, H, W//2+1)
        x_hat = torch.fft.rfft2(x, norm="ortho")

        # 2. Allocate output in frequency domain
        out_hat = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # 3. Multiply low-frequency block by learned weights.
        #    einsum "bixy,oixy->boxy" = batched matrix multiply over (x,y) modes.
        kh = min(self.k_max_h, H)
        kw = min(self.k_max_w, W // 2 + 1)
        out_hat[:, :, :kh, :kw] = torch.einsum(
            "bixy,oixy->boxy",
            x_hat[:, :, :kh, :kw],
            self.weights[:, :, :kh, :kw],
        )

        # 4. Inverse FFT2 back to spatial domain
        return torch.fft.irfft2(out_hat, s=(H, W), norm="ortho")


# ─── Fourier layer ────────────────────────────────────────────────────────────

class FourierLayer(nn.Module):
    """
    Single FNO layer:  y = σ( W·x  +  K(x) )

    W is a pointwise (1×1) convolution — the "residual bypass" that lets the
    network learn local corrections on top of the global spectral operator.
    """

    def __init__(
        self,
        channels: int,
        k_max_h: int = 16,
        k_max_w: int = 16,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, k_max_h, k_max_w)
        self.bypass   = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm     = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


# ─── full FNO ─────────────────────────────────────────────────────────────────

class FNO2D(nn.Module):
    """
    Fourier Neural Operator for 2D scalar-field prediction.

    Input  : (B, C_in, H, W)   — stacked channels (mask, source, ...)
    Output : (B, C_out, H, W)  — predicted field (pressure)

    Parameters
    ----------
    C_in, C_out    : input / output channel counts
    d_model        : latent channel width (called 'd' or 'width' in the paper)
    n_layers       : number of Fourier layers
    k_max_h/w      : Fourier mode truncation per axis
    """

    def __init__(
        self,
        C_in: int    = 3,
        C_out: int   = 1,
        d_model: int = 32,
        n_layers: int = 4,
        k_max_h: int  = 16,
        k_max_w: int  = 16,
    ) -> None:
        super().__init__()

        # Lifting: map C_in channels → d_model latent channels
        self.lift = nn.Conv2d(C_in, d_model, kernel_size=1)

        # Fourier processing blocks
        self.fourier_layers = nn.ModuleList([
            FourierLayer(d_model, k_max_h, k_max_w)
            for _ in range(n_layers)
        ])

        # Projection: d_model → 128 → C_out  (two-layer MLP per point)
        self.project = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, C_out, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x    : (B, C_in, H, W)
        mask : (B, 1, H, W) optional geometry mask — if provided, output is
               zeroed at wall/obstacle cells to hard-enforce Dirichlet BC.
        """
        x = self.lift(x)

        for layer in self.fourier_layers:
            x = layer(x)

        x = self.project(x)

        if mask is not None:
            x = x * mask

        return x


# ─── quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    model = FNO2D(C_in=2, C_out=1, d_model=32, n_layers=4, k_max_h=16, k_max_w=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FNO2D  parameters: {n_params:,}")

    # Training resolution
    x_train = torch.randn(4, 2, 64, 64)
    y_train = model(x_train)
    print(f"Train  input {tuple(x_train.shape)} → output {tuple(y_train.shape)}")

    # Zero-shot super-resolution: same weights, larger grid
    x_super = torch.randn(4, 2, 256, 256)
    y_super = model(x_super)
    print(f"Super  input {tuple(x_super.shape)} → output {tuple(y_super.shape)}")
