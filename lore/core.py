r"""Core LoRe training and evaluation.

Notation (paper Eq. 7, 10, 11)
------------------------------
For an embedding difference :math:`\Delta = e(x, y_c) - e(x, y_r)` and a reward basis
:math:`R_\phi`, user :math:`i` prefers :math:`y_c` with probability

.. math::  \sigma\!\left( w_i^\top R_\phi(\Delta) / \tau \right), \qquad w_i \in \Delta^{B-1}.

* ``fit_joint``   solves Eq. (10): learn :math:`\phi` and the seen-user weights jointly.
* ``fit_fewshot`` solves Eq. (11): freeze :math:`\phi`, learn weights for unseen users.

Implementation details that follow the official LoRe code (and our notebooks):

* user weights are parameterised as logits and mapped to the simplex with ``softmax``;
* the margin is divided by a temperature ``logit_scale = 100`` (not stated in the paper);
* the loss is the *sum over users of the per-user mean* logistic loss, so every user
  counts equally regardless of how many comparisons they contributed;
* optimisation is full-batch Adam (lr 0.5 joint, 0.1 few-shot).

Every input row is oriented as *chosen minus rejected*, so the label is always 1.
This is only safe if the basis is an **odd** function (see ``lore.models``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import seed_everything


@dataclass
class LoReConfig:
    K: int = 5                 # number of basis reward functions (rank B in the paper)
    joint_steps: int = 1000
    joint_lr: float = 0.5
    fewshot_steps: int = 500
    fewshot_lr: float = 0.1
    logit_scale: float = 100.0
    seed: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Scoring and loss
# --------------------------------------------------------------------------- #
def bt_scores(
    basis_out: torch.Tensor,
    user_logits: torch.Tensor,
    user_idx: torch.Tensor,
    logit_scale: float = 100.0,
    anchor: bool = False,
) -> torch.Tensor:
    """Personalised Bradley-Terry margins ``w_i^T R(Delta) / tau`` for every row.

    If ``anchor`` is True the first basis column is a fixed reference reward with
    coefficient 1 and the user simplex weights act on the remaining columns
    (used by :class:`lore.models.FixedAnchorResidualBasis`).
    """
    w = F.softmax(user_logits, dim=1)[user_idx]
    if anchor:
        s = basis_out[:, 0] + (basis_out[:, 1:] * w).sum(dim=1)
    else:
        s = (basis_out * w).sum(dim=1)
    return s / logit_scale


def user_balanced_bt_loss(scores: torch.Tensor, user_idx: torch.Tensor, n_users: int) -> torch.Tensor:
    """Sum over users of the mean logistic loss ``log(1 + exp(-s))`` (paper Eq. 10)."""
    losses = F.softplus(-scores)
    sums = torch.zeros(n_users, device=scores.device).index_add_(0, user_idx, losses)
    counts = torch.bincount(user_idx, minlength=n_users).float()
    active = counts > 0
    return (sums[active] / counts[active]).sum()


def _user_dim(basis: nn.Module, K: int) -> int:
    return getattr(basis, "user_dim", K)


def _is_anchor(basis: nn.Module) -> bool:
    return bool(getattr(basis, "has_anchor", False))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def fit_joint(
    basis: nn.Module,
    X: torch.Tensor,
    user_idx: torch.Tensor,
    n_users: int,
    cfg: LoReConfig,
    device: torch.device | str = "cpu",
    log_every: int = 0,
) -> Dict:
    """Jointly learn the shared basis and seen-user weights (paper Eq. 10)."""
    seed_everything(cfg.seed)
    device = torch.device(device)
    basis = basis.to(device)
    X, user_idx = X.to(device).float(), user_idx.to(device)

    W = nn.Parameter(torch.randn(n_users, _user_dim(basis, cfg.K), device=device))
    params = [p for p in basis.parameters() if p.requires_grad] + [W]
    opt = torch.optim.Adam(params, lr=cfg.joint_lr)
    anchor = _is_anchor(basis)

    history: List[float] = []
    basis.train()
    for step in range(cfg.joint_steps):
        opt.zero_grad(set_to_none=True)
        s = bt_scores(basis(X), W, user_idx, cfg.logit_scale, anchor)
        loss = user_balanced_bt_loss(s, user_idx, n_users)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite joint loss at step {step + 1}")
        loss.backward()
        opt.step()
        history.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"  joint {step + 1:5d}/{cfg.joint_steps}  loss={loss.item():.4f}")

    basis.eval()
    for p in basis.parameters():
        p.requires_grad_(False)
    return {"basis": basis, "W_logits": W.detach(), "loss_history": history}


def fit_fewshot(
    basis: nn.Module,
    X: torch.Tensor,
    user_idx: torch.Tensor,
    n_users: int,
    cfg: LoReConfig,
    init_logits: Optional[torch.Tensor] = None,
    device: torch.device | str = "cpu",
    steps: Optional[int] = None,
    eval_fn=None,
    eval_at: tuple = (),
) -> Dict:
    """Freeze the basis and learn unseen-user weights (paper Eq. 11).

    ``init_logits`` overrides LoRe's default N(0, I) logit initialisation; this is
    the hook used by the initialisation ablation (``lore.init``).
    ``eval_fn(W_logits) -> dict`` is called at every step listed in ``eval_at``
    (step 0 = before any update).
    """
    device = torch.device(device)
    steps = cfg.fewshot_steps if steps is None else steps
    basis = basis.to(device).eval()
    X, user_idx = X.to(device).float(), user_idx.to(device)

    with torch.no_grad():  # the basis is frozen: compute its output once
        Z = basis(X)

    if init_logits is None:
        g = torch.Generator(device="cpu").manual_seed(cfg.seed + 1000)
        init_logits = torch.randn(n_users, _user_dim(basis, cfg.K), generator=g)
    W = nn.Parameter(init_logits.clone().float().to(device))
    opt = torch.optim.Adam([W], lr=cfg.fewshot_lr)
    anchor = _is_anchor(basis)

    trace, history = [], []
    for step in range(steps + 1):
        if eval_fn is not None and step in eval_at:
            trace.append({"updates": step, **eval_fn(W.detach())})
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss = user_balanced_bt_loss(bt_scores(Z, W, user_idx, cfg.logit_scale, anchor), user_idx, n_users)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return {"W_logits": W.detach(), "loss_history": history, "trace": trace}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_users(
    basis: nn.Module,
    X: torch.Tensor,
    user_idx: torch.Tensor,
    W_logits: torch.Tensor,
    logit_scale: float = 100.0,
    batch_size: int = 65536,
    symmetric: bool = False,
) -> Dict:
    """Pairwise accuracy ``1[w_i^T R(Delta) > 0]`` per user (paper Eq. 15).

    With ``symmetric=True`` every pair is also scored in the swapped orientation
    (``-Delta``, label 0) and the two accuracies are averaged.  For a correct
    (odd) basis this equals the standard accuracy; a model that exploits the
    "label is always 1" shortcut collapses to 50%.
    """
    device = W_logits.device
    basis = basis.to(device).eval()
    anchor = _is_anchor(basis)
    n_users = W_logits.shape[0]
    correct = torch.zeros(n_users, device=device)
    counts = torch.bincount(user_idx.to(device), minlength=n_users).float()

    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size].to(device).float()
        ub = user_idx[start:start + batch_size].to(device)
        s = bt_scores(basis(xb), W_logits, ub, logit_scale, anchor)
        hit = (s > 0).float()
        if symmetric:
            s_neg = bt_scores(basis(-xb), W_logits, ub, logit_scale, anchor)
            hit = 0.5 * (hit + (s_neg < 0).float())
        correct.index_add_(0, ub, hit)

    per_user = (correct / counts.clamp_min(1)).cpu().numpy()
    active = counts.cpu().numpy() > 0
    per_user = per_user[active]
    return {
        "mean_accuracy": float(per_user.mean()),          # macro average over users
        "user_std": float(per_user.std()),
        "micro_accuracy": float(correct.sum().item() / counts.sum().item()),
        "per_user_accuracy": per_user,
        "num_users": int(active.sum()),
        "num_comparisons": int(counts.sum().item()),
    }


def evaluate_with_weights(Z: torch.Tensor, weights: torch.Tensor) -> np.ndarray:
    """Fast path when features are already projected: ``Z`` is ``[users, n, K]``,
    ``weights`` is ``[users, K]`` on the simplex.  Returns per-user accuracy."""
    margins = (Z * weights.unsqueeze(1)).sum(-1)
    return (margins > 0).float().mean(1).cpu().numpy()
