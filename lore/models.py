r"""Reward-basis function classes :math:`R_\phi : \mathbb{R}^D \to \mathbb{R}^K`.

All modules take an embedding **difference** ``Delta = e(x, y_c) - e(x, y_r)`` and
return ``K`` basis rewards.  Because every training row is oriented
chosen-minus-rejected (label always 1), a basis must be **odd**,
``R(-Delta) = -R(Delta)``, otherwise it can reach zero loss with a constant output
(the "orientation shortcut", see ``NaiveMLPBasis`` and Section 5.3 of the report).

* Linear maps without bias are odd automatically (this is the paper's Example 1).
* Nonlinear maps are made odd by antisymmetrisation ``0.5 * (f(x) - f(-x))``.

Architectures used in the function-class ablation
-------------------------------------------------
=========================  ==========================================  ==================
name                       description                                 trainable (D=4096)
=========================  ==========================================  ==================
``pca5``                   fixed top-5 PCA directions, no training     0
``pca512_linear``          fixed PCA-512 projection + linear 512->K    2,560
``random512_linear``       fixed random orthogonal 512 + linear        2,560
``linear``                 LoRe: linear 4096->K, no bias (paper)       20,480
``mlp8`` / ``mlp64``       odd 2-layer GELU MLP, hidden 8 / 64         32,821 / 262,533
``residual16``             linear + odd rank-16 GELU residual          86,096
``fixed_anchor_residual``  frozen Skywork head + 4 learned directions  16,384
=========================  ==========================================  ==================
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBasis(nn.Module):
    """LoRe basis ``R(Delta) = A Delta`` with ``A in R^{K x D}`` (paper Eq. 8/21).

    Initialised N(0, 1) like the official implementation.
    """

    def __init__(self, in_dim: int, K: int):
        super().__init__()
        self.V = nn.Parameter(torch.randn(in_dim, K))

    def forward(self, x):
        return x @ self.V


class FixedProjectionBasis(nn.Module):
    """Frozen projection ``P in R^{K x D}`` (e.g. top-K PCA directions). No trainable
    parameters: only the user weights are learned, which makes the problem convex."""

    def __init__(self, projection: torch.Tensor):
        super().__init__()
        self.register_buffer("P", projection.float())  # [K, D]

    def forward(self, x):
        return x @ self.P.T


class ProjectedLinearBasis(nn.Module):
    """Frozen projection to ``m`` dims (PCA or random orthogonal) followed by a
    trainable bias-free linear map ``m -> K``."""

    def __init__(self, projection: torch.Tensor, K: int):
        super().__init__()
        self.register_buffer("P", projection.float())  # [m, D]
        self.linear = nn.Linear(projection.shape[0], K, bias=False)

    def forward(self, x):
        return self.linear(x @ self.P.T)


class OddMLPBasis(nn.Module):
    """Two-layer GELU MLP made odd: ``0.5 * (f(x) - f(-x))``."""

    def __init__(self, in_dim: int, K: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, K))

    def forward(self, x):
        return 0.5 * (self.net(x) - self.net(-x))


class OddResidualBasis(nn.Module):
    """Linear LoRe basis plus an odd low-rank nonlinear residual branch."""

    def __init__(self, in_dim: int, K: int, rank: int = 16):
        super().__init__()
        self.base = nn.Linear(in_dim, K, bias=False)
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, K, bias=False)

    def _branch(self, x):
        return self.up(F.gelu(self.down(x)))

    def forward(self, x):
        return self.base(x) + 0.5 * (self._branch(x) - self._branch(-x))


class FixedAnchorResidualBasis(nn.Module):
    """Reference reward + learned residual directions.

    Column 0 is the frozen pre-trained reward head ``a`` (Skywork ``score.weight``)
    with coefficient fixed to 1; the remaining ``K-1`` directions are learned, kept
    orthogonal to ``a`` and (by default) rescaled to ``||a||``.  Users mix only the
    residual directions, so the model is "everyone shares the reference reward,
    and personalisation is a correction on top of it".
    """

    has_anchor = True

    def __init__(self, anchor: torch.Tensor, n_residual: int = 4, fix_norm: bool = True):
        super().__init__()
        anchor = torch.as_tensor(anchor, dtype=torch.float32).flatten()
        if anchor.norm() == 0:
            raise ValueError("Anchor vector must be non-zero.")
        self.register_buffer("anchor", anchor)
        self.register_buffer("anchor_unit", anchor / anchor.norm())
        self.residual = nn.Parameter(torch.empty(n_residual, anchor.numel()))
        nn.init.orthogonal_(self.residual)
        self.fix_norm = fix_norm
        self.user_dim = n_residual

    def directions(self) -> torch.Tensor:
        r = self.residual
        r = r - (r @ self.anchor_unit).unsqueeze(1) * self.anchor_unit
        if self.fix_norm:
            r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12) * self.anchor.norm()
        return r

    def forward(self, x):
        return torch.cat([(x @ self.anchor).unsqueeze(1), x @ self.directions().T], dim=1)


class NaiveMLPBasis(nn.Module):
    """**Deliberately wrong** baseline: an ordinary MLP (with biases) on differences.

    It is *not* odd, so with every training label equal to 1 it can output a
    positive constant and reach ~100% "accuracy" on held-out chosen-minus-rejected
    pairs while carrying no preference information.  Used only to demonstrate the
    orientation shortcut we hit in our first ablation pass.
    """

    def __init__(self, in_dim: int, K: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, K))

    def forward(self, x):
        return self.net(x)


ARCHITECTURES = [
    "pca5",
    "pca512_linear",
    "random512_linear",
    "linear",
    "mlp8",
    "mlp64",
    "residual16",
    "fixed_anchor_residual",
]

DISPLAY_NAMES = {
    "pca5": "PCA-5",
    "pca512_linear": "PCA-512 + Linear",
    "random512_linear": "Random-512 + Linear",
    "linear": "Original Linear (LoRe)",
    "mlp8": "MLP-8 (odd)",
    "mlp64": "MLP-64 (odd)",
    "residual16": "Residual-16 (odd)",
    "fixed_anchor_residual": "Fixed-Anchor-Residual",
    "naive_mlp8": "MLP-8 (naive, not odd)",
}


def build_basis(
    name: str,
    in_dim: int = 4096,
    K: int = 5,
    projections: Optional[dict] = None,
    anchor: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Factory for every basis used in the project.

    ``projections`` must contain ``pca5`` ``[K, D]``, ``pca512`` ``[512, D]`` and
    ``random512`` ``[512, D]`` for the projected architectures (see
    :mod:`lore.projections`).  ``anchor`` is the frozen reward head for
    ``fixed_anchor_residual``.
    """
    if name == "pca5":
        return FixedProjectionBasis(projections["pca5"][:K])
    if name == "pca512_linear":
        return ProjectedLinearBasis(projections["pca512"], K)
    if name == "random512_linear":
        return ProjectedLinearBasis(projections["random512"], K)
    if name == "linear":
        return LinearBasis(in_dim, K)
    if name == "mlp8":
        return OddMLPBasis(in_dim, K, 8)
    if name == "mlp64":
        return OddMLPBasis(in_dim, K, 64)
    if name == "residual16":
        return OddResidualBasis(in_dim, K, 16)
    if name == "fixed_anchor_residual":
        if anchor is None:
            raise ValueError("fixed_anchor_residual needs the reward-head anchor vector")
        return FixedAnchorResidualBasis(anchor, n_residual=K - 1)
    if name == "naive_mlp8":
        return NaiveMLPBasis(in_dim, K, 8)
    raise ValueError(f"Unknown architecture {name!r}. Choose from {ARCHITECTURES}.")


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_frozen(module: nn.Module) -> int:
    return sum(b.numel() for n, b in module.named_buffers() if n in {"P", "anchor"})
