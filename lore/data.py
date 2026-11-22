"""Benchmark construction: PersonalLLM synthetic users and Reddit TL;DR worker splits.

PersonalLLM (Zollo et al., 2024)
    10,402 prompts x 8 responses, each scored by 10 reward models.  A synthetic user
    is ``w ~ Dirichlet(alpha * 1_10)``; their reward is ``w^T R(x, y)``.  For every
    sampled prompt the best- and worst-scoring responses form the (chosen, rejected)
    pair.  Seen users get ``0.005 * 9402 = 47`` training prompts, unseen users
    ``0.001 * 9402 = 9`` few-shot prompts; all 1,000 test prompts are used for evaluation.

Reddit TL;DR (Stiennon et al., 2020, ``openai/summarize_from_feedback``)
    Workers present in both the train and validation splits are shuffled with seed 0
    and split in half (seen / unseen).  Seen workers contribute up to 150 training
    comparisons, unseen workers up to 50 few-shot comparisons.  A worker with a single
    annotation is dropped, leaving 20 seen and 19 unseen workers.  Validation is capped
    at 300 comparisons per worker (seed 42) to keep embedding extraction affordable.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

PERSONALLLM_REWARD_MODELS = [
    "gemma_2b",
    "gemma_7b",
    "mistral_raft",
    "llama3_sfairx",
    "oasst_deberta_v3",
    "beaver_7b",
    "oasst_pythia_7b",
    "oasst_pythia_1b",
    "mistral_ray",
    "mistral_weqweasdas",
]
N_RESPONSES = 8


# --------------------------------------------------------------------------- #
# PersonalLLM
# --------------------------------------------------------------------------- #
def personalllm_reward_scores(split) -> np.ndarray:
    """``[n_prompts, 8, 10]`` reward-model scores from a HF ``datasets`` split."""
    scores = np.empty((len(split), N_RESPONSES, len(PERSONALLLM_REWARD_MODELS)), dtype=np.float32)
    for r in range(N_RESPONSES):
        for m, name in enumerate(PERSONALLLM_REWARD_MODELS):
            scores[:, r, m] = np.asarray(split[f"response_{r + 1}_{name}"], dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite reward scores")
    return scores


def sample_dirichlet_users(n_users: int, alpha: float, rng: np.random.Generator, n_models: int = 10) -> np.ndarray:
    """Synthetic users ``w ~ Dir(alpha)``. Small alpha -> users close to one reward model."""
    return rng.dirichlet(np.full(n_models, alpha), size=n_users)


def best_worst_pairs(scores: np.ndarray, user_weights: np.ndarray, prompt_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Chosen/rejected response indices for each (user, prompt).

    scores       : [n_prompts, 8, n_models]
    user_weights : [n_users, n_models]
    prompt_idx   : [n_users, n_examples]
    returns      : chosen [n_users, n_examples], rejected [n_users, n_examples]
    """
    util = np.einsum("unrm,um->unr", scores[prompt_idx], user_weights, optimize=True)
    return util.argmax(-1), util.argmin(-1)


def sample_prompts(n_users: int, n_examples: int, n_prompts: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([rng.choice(n_prompts, size=n_examples, replace=False) for _ in range(n_users)])


def personalllm_deltas(
    embeddings: torch.Tensor,
    prompt_idx: np.ndarray,
    chosen: np.ndarray,
    rejected: np.ndarray,
) -> torch.Tensor:
    """Embedding differences ``[n_users, n_examples, D]`` (float32).

    ``embeddings`` is ``[n_prompts, 8, D]`` (bf16 is fine; the subtraction is done in
    the stored dtype and then cast, exactly as in our notebooks)."""
    p = torch.as_tensor(prompt_idx, dtype=torch.long)
    c = torch.as_tensor(chosen, dtype=torch.long)
    r = torch.as_tensor(rejected, dtype=torch.long)
    return (embeddings[p, c] - embeddings[p, r]).float()


def build_personalllm_users(
    train_scores: np.ndarray,
    alpha: float,
    seed: int,
    n_seen: int = 1000,
    n_unseen: int = 500,
    seen_frac: float = 0.005,
    fewshot_frac: float = 0.001,
) -> Dict[str, np.ndarray]:
    """Sample the synthetic population and the training prompts for one run.

    The RNG call order matches notebook 02 (seen weights, unseen weights, seen
    prompts, few-shot prompts) so results are bit-for-bit reproducible.
    """
    rng = np.random.default_rng(seed)
    n_prompts = train_scores.shape[0]
    W_seen = sample_dirichlet_users(n_seen, alpha, rng, train_scores.shape[-1])
    W_unseen = sample_dirichlet_users(n_unseen, alpha, rng, train_scores.shape[-1])
    n_seen_pairs = int(seen_frac * n_prompts)      # 47
    n_fewshot_pairs = int(fewshot_frac * n_prompts)  # 9
    seen_prompts = sample_prompts(n_seen, n_seen_pairs, n_prompts, rng)
    fewshot_prompts = sample_prompts(n_unseen, n_fewshot_pairs, n_prompts, rng)
    return {
        "W_seen_true": W_seen,
        "W_unseen_true": W_unseen,
        "seen_prompts": seen_prompts,
        "fewshot_prompts": fewshot_prompts,
    }


# --------------------------------------------------------------------------- #
# Reddit TL;DR
# --------------------------------------------------------------------------- #
def reddit_worker_split(
    train_workers: Sequence[str],
    val_workers: Sequence[str],
    seed: int = 0,
    n_seen_train: int = 150,
    n_fewshot: int = 50,
    val_cap: int = 300,
    val_seed: int = 42,
    min_train: int = 2,
) -> Dict:
    """Reproduce the worker split / sampling used in notebook 01.

    ``train_workers`` and ``val_workers`` are the per-row worker IDs of the HF
    ``train`` and ``validation`` splits.  Returns row indices for every role.
    """
    def index(workers):
        d = defaultdict(list)
        for i, w in enumerate(workers):
            d[str(w)].append(i)
        return dict(d)

    tr, va = index(train_workers), index(val_workers)
    # Released LoRe script: drop the two least-represented workers in each split.
    tr_kept = dict(sorted(tr.items(), key=lambda kv: len(kv[1]))[2:])
    va_kept = dict(sorted(va.items(), key=lambda kv: len(kv[1]))[2:])

    common = sorted(set(tr) & set(va))
    rng = random.Random(seed)
    rng.shuffle(common)
    seen, unseen = common[: len(common) // 2], common[len(common) // 2:]

    seen_idx, fewshot_idx = {}, {}
    for w, rows in tr_kept.items():
        if w in seen:
            k = min(len(rows), n_seen_train)
            seen_idx[w] = [rows[p] for p in rng.sample(range(len(rows)), k)]
        elif w in unseen:
            k = min(len(rows), n_fewshot)
            fewshot_idx[w] = [rows[p] for p in rng.sample(range(len(rows)), k)]

    # Workers with too little data (one unseen worker has a single annotation).
    unseen = [w for w in unseen if w in fewshot_idx and len(tr.get(w, [])) >= min_train]
    seen = [w for w in seen if w in seen_idx]

    vrng = random.Random(val_seed)
    val_idx = {}
    for w, rows in va_kept.items():
        if w in seen or w in unseen:
            val_idx[w] = sorted(vrng.sample(rows, min(val_cap, len(rows))))

    return {
        "seed": seed,
        "seen_workers": seen,
        "unseen_workers": unseen,
        "seen_train_indices": seen_idx,
        "unseen_fewshot_indices": {w: fewshot_idx[w] for w in unseen},
        "val_eval_indices": val_idx,
    }


def reddit_records_to_deltas(records_by_worker: Dict[str, List[dict]]) -> Dict[str, torch.Tensor]:
    """Convert the pickled ``{worker: [record, ...]}`` produced by
    ``scripts/extract_embeddings.py`` into ``{worker: [n, D]}`` difference tensors."""
    out = {}
    for w, records in records_by_worker.items():
        win = torch.stack([torch.as_tensor(r["embeddings"]["winning"][0]).float().flatten() for r in records])
        lose = torch.stack([torch.as_tensor(r["embeddings"]["losing"][0]).float().flatten() for r in records])
        out[w] = win - lose
    return out


def subsample_per_user(deltas: Dict[str, torch.Tensor], users: Sequence[str], n: int, seed: int) -> Dict[str, torch.Tensor]:
    """Nested per-user subsample (the n=100 set is a subset of the n=150 set for a seed)."""
    out = {}
    for j, u in enumerate(users):
        X = deltas[u]
        perm = np.random.default_rng(seed + 1000 * j).permutation(len(X))
        out[u] = X[perm[: min(n, len(X))]].clone()
    return out
