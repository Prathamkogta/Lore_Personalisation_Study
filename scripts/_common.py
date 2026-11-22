"""Shared loaders that turn cached embeddings into a uniform benchmark interface.

Expected layout of ``--data-dir`` (written by ``scripts/extract_embeddings.py``)::

    data/
      personalllm/train.safetensors      [75216, 4096] bf16  (9402 prompts x 8 responses)
      personalllm/test.safetensors       [8000, 4096]  bf16
      personalllm/reward_scores.npz      train_scores [9402,8,10], test_scores [1000,8,10]
      reddit/tldr_embeddings_train.pkl   {worker: [record, ...]}
      reddit/tldr_embeddings_val.pkl
      reddit/user_split.json
      skywork_reward_head.pt             {"score_weight": [4096]}
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore.core import bt_scores  # noqa: E402
from lore.data import (  # noqa: E402
    best_worst_pairs,
    build_personalllm_users,
    personalllm_deltas,
    reddit_records_to_deltas,
    subsample_per_user,
)
from lore.utils import pack_users  # noqa: E402

Chunks = Callable[[], Iterator[Tuple[torch.Tensor, torch.Tensor]]]


@dataclass
class Benchmark:
    name: str
    n_seen: int
    n_unseen: int
    seen_train: Tuple[torch.Tensor, torch.Tensor]   # X [N, D], user_idx [N]
    unseen_fewshot: Tuple[torch.Tensor, torch.Tensor]
    seen_eval: Chunks                                # yields (X, user_idx) chunks
    unseen_eval: Chunks


# --------------------------------------------------------------------------- #
def load_personalllm(data_dir: Path, alpha: float, seed: int, n_seen=1000, n_unseen=500, users_per_chunk=8) -> Benchmark:
    from safetensors.torch import load_file

    d = Path(data_dir) / "personalllm"
    train_emb = load_file(str(d / "train.safetensors"))["embeddings"].reshape(-1, 8, 4096)
    test_emb = load_file(str(d / "test.safetensors"))["embeddings"].reshape(-1, 8, 4096)
    sc = np.load(d / "reward_scores.npz")
    tr_scores, te_scores = sc["train_scores"], sc["test_scores"]

    pop = build_personalllm_users(tr_scores, alpha, seed, n_seen, n_unseen)

    def train_split(W, prompts):
        c, r = best_worst_pairs(tr_scores, W, prompts)
        X = personalllm_deltas(train_emb, prompts, c, r)  # [U, n, D]
        u = torch.arange(len(W)).repeat_interleave(X.shape[1])
        return X.reshape(-1, X.shape[-1]), u

    def eval_chunks(W):
        all_prompts = np.arange(len(te_scores))

        def gen():
            for s in range(0, len(W), users_per_chunk):
                Wb = W[s:s + users_per_chunk]
                prompts = np.tile(all_prompts, (len(Wb), 1))
                c, r = best_worst_pairs(te_scores, Wb, prompts)
                X = personalllm_deltas(test_emb, prompts, c, r)
                u = torch.arange(s, s + len(Wb)).repeat_interleave(X.shape[1])
                yield X.reshape(-1, X.shape[-1]), u
        return gen

    bm = Benchmark(
        name=f"PersonalLLM(alpha={alpha})",
        n_seen=n_seen, n_unseen=n_unseen,
        seen_train=train_split(pop["W_seen_true"], pop["seen_prompts"]),
        unseen_fewshot=train_split(pop["W_unseen_true"], pop["fewshot_prompts"]),
        seen_eval=eval_chunks(pop["W_seen_true"]),
        unseen_eval=eval_chunks(pop["W_unseen_true"]),
    )
    bm.population = pop
    bm.train_emb, bm.test_emb, bm.train_scores, bm.test_scores = train_emb, test_emb, tr_scores, te_scores
    return bm


def load_reddit(data_dir: Path, n_seen_examples: int = 150, seed: int = 42) -> Benchmark:
    d = Path(data_dir) / "reddit"
    with open(d / "tldr_embeddings_train.pkl", "rb") as f:
        train = reddit_records_to_deltas(pickle.load(f))
    with open(d / "tldr_embeddings_val.pkl", "rb") as f:
        val = reddit_records_to_deltas(pickle.load(f))
    split = json.load(open(d / "user_split.json"))
    seen, unseen = sorted(split["seen_workers"]), sorted(split["unseen_workers"])

    seen_sub = subsample_per_user(train, seen, n_seen_examples, seed)
    Xs, us, _ = pack_users(seen_sub, seen)
    Xf, uf, _ = pack_users(train, unseen)
    Xvs, uvs, _ = pack_users(val, seen)
    Xvu, uvu, _ = pack_users(val, unseen)

    bm = Benchmark(
        name=f"RedditTLDR(n_seen={n_seen_examples})",
        n_seen=len(seen), n_unseen=len(unseen),
        seen_train=(Xs, us), unseen_fewshot=(Xf, uf),
        seen_eval=lambda: iter([(Xvs, uvs)]),
        unseen_eval=lambda: iter([(Xvu, uvu)]),
    )
    bm.seen_workers, bm.unseen_workers = seen, unseen
    return bm


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_chunks(basis, W_logits, chunks: Chunks, logit_scale=100.0, symmetric=False, device="cpu"):
    """Macro (per-user) accuracy over a lazily generated evaluation set."""
    n_users = W_logits.shape[0]
    correct = torch.zeros(n_users)
    count = torch.zeros(n_users)
    anchor = bool(getattr(basis, "has_anchor", False))
    W = W_logits.to(device)
    for X, u in chunks():
        X, ud = X.to(device).float(), u.to(device)
        hit = (bt_scores(basis(X), W, ud, logit_scale, anchor) > 0).float()
        if symmetric:
            hit = 0.5 * (hit + (bt_scores(basis(-X), W, ud, logit_scale, anchor) < 0).float())
        correct.index_add_(0, u, hit.cpu())
        count.index_add_(0, u, torch.ones(len(u)))
    acc = (correct / count.clamp_min(1)).numpy()
    return {"mean_accuracy": float(acc.mean()), "user_std": float(acc.std()), "per_user_accuracy": acc}


@torch.no_grad()
def reference_accuracy(anchor: torch.Tensor, chunks: Chunks, n_users: int) -> float:
    """Accuracy of the frozen Skywork reward ``score.weight @ e`` (no personalisation)."""
    correct, count = torch.zeros(n_users), torch.zeros(n_users)
    for X, u in chunks():
        hit = ((X.float() @ anchor) > 0).float()
        correct.index_add_(0, u, hit)
        count.index_add_(0, u, torch.ones(len(u)))
    return float((correct / count.clamp_min(1)).mean())
