"""LoRe: Low-Rank Reward Modeling for few-shot LLM personalization.

A compact, readable re-implementation of the core of
Bose et al. (2025), "LoRe: Personalizing LLMs via Low-Rank Reward Modeling"
(arXiv:2504.14439), together with the ablations studied in our course project.

Modules
-------
core          Bradley-Terry losses, joint basis training, few-shot adaptation, evaluation.
models        Reward-basis function classes (linear, projected, odd MLP, residual, fixed anchor).
data          PersonalLLM synthetic users, Reddit TL;DR worker splits, pair construction.
projections   PCA and random orthogonal projections of embedding differences.
init          Initialization strategies for unseen-user preference weights.
embeddings    Frozen Skywork reward-model embedding extraction (GPU, optional).
synthetic     Small CPU-only synthetic benchmark used for tests and sanity checks.
"""

from .core import (
    LoReConfig,
    bt_scores,
    evaluate_users,
    fit_fewshot,
    fit_joint,
    user_balanced_bt_loss,
)
from .models import ARCHITECTURES, build_basis, count_trainable

__all__ = [
    "LoReConfig",
    "bt_scores",
    "evaluate_users",
    "fit_fewshot",
    "fit_joint",
    "user_balanced_bt_loss",
    "ARCHITECTURES",
    "build_basis",
    "count_trainable",
]

__version__ = "1.0.0"
