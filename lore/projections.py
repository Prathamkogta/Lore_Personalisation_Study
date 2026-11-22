"""Frozen projections of embedding differences used in the representation ablation."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def fit_pca(train_deltas: np.ndarray | torch.Tensor, n_components: int = 512, seed: int = 42) -> Dict[str, torch.Tensor]:
    """Randomised PCA on seen-user training differences only (no test leakage).

    Returns ``pca512`` (components x D), ``pca5`` (first 5 rows) and the explained
    variance ratio.  Note: PCA directions have an arbitrary sign; with simplex user
    weights a fixed PCA basis cannot flip a direction (see report, Prop. 2).
    """
    from sklearn.decomposition import PCA

    X = train_deltas.numpy() if isinstance(train_deltas, torch.Tensor) else np.asarray(train_deltas)
    n_components = min(n_components, *X.shape)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(X)
    comps = torch.tensor(pca.components_, dtype=torch.float32)
    return {
        "pca512": comps,
        "pca5": comps[:5].clone(),
        "explained_variance_ratio": torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32),
    }


def random_orthogonal(dim: int, m: int = 512, seed: int = 42) -> torch.Tensor:
    """Data-independent random projection with orthonormal rows, ``[m, dim]``."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim, m, generator=g), mode="reduced")
    return Q.T.contiguous()
