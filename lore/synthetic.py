r"""A small CPU-only synthetic benchmark with known low-rank structure.

Generative model
----------------
* response embeddings ``e ~ N(0, I_D)`` (``n_prompts x n_responses x D``);
* a true basis ``A* in R^{K* x D}`` and users ``w_i ~ Dir(alpha)``;
* user reward ``r_i(e) = w_i^T A* e`` and preference labels drawn from the
  Bradley-Terry model with temperature ``beta``.

Pairs are stored as ``Delta = e_chosen - e_rejected`` (label 1), mirroring the real
pipeline, so the same code paths (and the same pitfalls) apply.  Used by the unit
tests and by ``scripts/demo_synthetic.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class SyntheticConfig:
    D: int = 128
    K_true: int = 4
    n_prompts: int = 4000
    n_responses: int = 4
    n_seen: int = 200
    n_unseen: int = 100
    n_train: int = 40
    n_fewshot: int = 10
    n_test: int = 300
    alpha: float = 0.1
    beta: float = 1.0
    seed: int = 0


def _pairs_for_user(rng, E, A, w, n, beta):
    n_prompts, n_resp, _ = E.shape
    p = rng.choice(n_prompts, size=n, replace=False)
    i, j = np.stack([rng.choice(n_resp, 2, replace=False) for _ in range(n)]).T
    ei, ej = E[p, i], E[p, j]
    margin = (ei - ej) @ A.T @ w
    prob = 1.0 / (1.0 + np.exp(-margin / beta))
    first_wins = rng.random(n) < prob
    delta = np.where(first_wins[:, None], ei - ej, ej - ei)
    return torch.tensor(delta, dtype=torch.float32)


def make_synthetic(cfg: SyntheticConfig = SyntheticConfig()) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    E = rng.standard_normal((cfg.n_prompts, cfg.n_responses, cfg.D))
    A = rng.standard_normal((cfg.K_true, cfg.D)) / np.sqrt(cfg.D) * 6.0
    Ws = rng.dirichlet(np.full(cfg.K_true, cfg.alpha), size=cfg.n_seen)
    Wu = rng.dirichlet(np.full(cfg.K_true, cfg.alpha), size=cfg.n_unseen)

    seen_train = [_pairs_for_user(rng, E, A, w, cfg.n_train, cfg.beta) for w in Ws]
    seen_test = [_pairs_for_user(rng, E, A, w, cfg.n_test, cfg.beta) for w in Ws]
    unseen_few = [_pairs_for_user(rng, E, A, w, cfg.n_fewshot, cfg.beta) for w in Wu]
    unseen_test = [_pairs_for_user(rng, E, A, w, cfg.n_test, cfg.beta) for w in Wu]
    return {
        "A_true": torch.tensor(A, dtype=torch.float32),
        "W_seen_true": Ws, "W_unseen_true": Wu,
        "seen_train": seen_train, "seen_test": seen_test,
        "unseen_fewshot": unseen_few, "unseen_test": unseen_test,
    }
