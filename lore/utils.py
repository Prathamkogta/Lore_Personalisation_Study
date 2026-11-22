"""Small shared helpers: seeding, device selection, JSON I/O, packing per-user tensors."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_json(path: str | os.PathLike):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def pack_users(
    per_user: Dict[str, torch.Tensor] | Sequence[torch.Tensor],
    order: Iterable | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, List]:
    """Concatenate per-user ``[n_i, d]`` tensors into one ``[sum n_i, d]`` tensor.

    Returns
    -------
    X        : concatenated features
    user_idx : LongTensor with the (0-based) user index of every row
    order    : the user keys in the order used
    """
    if isinstance(per_user, dict):
        order = list(order) if order is not None else list(per_user.keys())
        tensors = [per_user[u] for u in order]
    else:
        tensors = list(per_user)
        order = list(range(len(tensors)))

    if not tensors:
        raise ValueError("No users supplied.")

    xs, idx = [], []
    for i, t in enumerate(tensors):
        if t.ndim != 2 or len(t) == 0:
            raise ValueError(f"User {order[i]!r}: expected non-empty [n, d] tensor, got {tuple(t.shape)}")
        xs.append(t.float())
        idx.append(torch.full((len(t),), i, dtype=torch.long))
    return torch.cat(xs), torch.cat(idx), order


def mean_ci95(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and half-width of a two-sided 95% Student-t interval."""
    from scipy.stats import t

    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), float("nan")
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(t.ppf(0.975, len(v) - 1) * se)
