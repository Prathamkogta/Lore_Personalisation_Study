r"""Initialisation strategies for unseen-user preference weights.

LoRe initialises the few-shot logits as ``N(0, I)``.  Because the few-shot problem
(Eq. 11) is convex in ``w`` for a frozen basis, every start converges to the same
optimum; the initialisation only matters when the update budget is small.  We test
whether the *seen population* gives a better starting point:

``random_logits``      LoRe default, softmax(N(0, I))
``uniform``            w = 1/K
``population_mean``    mean of the learned seen-user weights
``nearest_seen_user``  seen-user weight vector with the lowest BT loss on the new user's support set
``cluster_centroid``   K-means (k=5) centroid of seen-user weights with the lowest support loss

All strategies operate on support features already projected through the frozen
basis: ``Z = Delta @ V`` with shape ``[users, n_support, K]``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

METHODS = ["random_logits", "uniform", "population_mean", "nearest_seen_user", "cluster_centroid"]


def normalize(w: torch.Tensor) -> torch.Tensor:
    w = w.clamp_min(1e-8)
    return w / w.sum(-1, keepdim=True)


def to_logits(w: torch.Tensor) -> torch.Tensor:
    """Centred logits whose softmax equals ``w``."""
    z = torch.log(normalize(w))
    return z - z.mean(-1, keepdim=True)


def support_loss(Z: torch.Tensor, candidates: torch.Tensor, logit_scale: float = 100.0) -> torch.Tensor:
    """Mean BT loss of every candidate weight vector on every user's support set.

    Z: [users, n, K], candidates: [C, K]  ->  [users, C]
    """
    return F.softplus(-(Z @ candidates.T) / logit_scale).mean(1)


def select_by_support(Z: torch.Tensor, candidates: torch.Tensor, batch: int = 64, logit_scale: float = 100.0):
    """Pick, for each user, the candidate with minimum support loss (a 1-NN in loss space)."""
    chosen, idx = [], []
    for s in range(0, len(Z), batch):
        L = support_loss(Z[s:s + batch].float(), candidates, logit_scale)
        i = L.argmin(1)
        idx.append(i)
        chosen.append(candidates[i])
    return torch.cat(chosen), torch.cat(idx)


def kmeans_centroids(W_seen: torch.Tensor, k: int = 5, seed: int = 42) -> torch.Tensor:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(W_seen.cpu().numpy())
    return normalize(torch.tensor(km.cluster_centers_, dtype=torch.float32))


def initial_weights(
    method: str,
    Z_support: torch.Tensor,
    W_seen: torch.Tensor,
    seed: int,
    n_clusters: int = 5,
) -> torch.Tensor:
    """Simplex weights ``[users, K]`` for one strategy."""
    n_users, _, K = Z_support.shape
    if method == "random_logits":
        g = torch.Generator().manual_seed(900_000 + seed)
        return torch.softmax(torch.randn(n_users, K, generator=g), dim=1)
    if method == "uniform":
        return torch.full((n_users, K), 1.0 / K)
    if method == "population_mean":
        return normalize(W_seen.mean(0, keepdim=True)).expand(n_users, -1).clone()
    if method == "nearest_seen_user":
        return select_by_support(Z_support, W_seen)[0]
    if method == "cluster_centroid":
        return select_by_support(Z_support, kmeans_centroids(W_seen, n_clusters))[0]
    raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")


def adapt(
    Z_support: torch.Tensor,
    Z_query: torch.Tensor,
    w0: torch.Tensor,
    steps: int = 50,
    lr: float = 0.1,
    eval_at=(0, 5, 20, 50),
    logit_scale: float = 100.0,
) -> list:
    """Few-shot Adam on softmax logits starting from ``w0``; query accuracy is logged
    at the requested update counts."""
    theta = torch.nn.Parameter(torch.log(w0.clamp_min(1e-8)).clone())
    opt = torch.optim.Adam([theta], lr=lr)
    out = []
    for step in range(steps + 1):
        if step in eval_at:
            with torch.no_grad():
                w = torch.softmax(theta, 1)
                acc = ((Z_query * w.unsqueeze(1)).sum(-1) > 0).float().mean(1)
                sl = F.softplus(-(Z_support * w.unsqueeze(1)).sum(-1) / logit_scale).mean()
            out.append({"updates": step, "query_accuracy": acc.mean().item(),
                        "query_accuracy_user_std": acc.std(unbiased=False).item(),
                        "support_loss": sl.item()})
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        w = torch.softmax(theta, 1)
        loss = F.softplus(-(Z_support * w.unsqueeze(1)).sum(-1) / logit_scale).mean(1).sum()
        loss.backward()
        opt.step()
    return out
