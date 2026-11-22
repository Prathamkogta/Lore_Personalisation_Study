#!/usr/bin/env python
"""Experiment 3: how should a new user's weights be initialised?

Re-uses the PersonalLLM (alpha=0.1) LoRe bases from Experiment 1 (one per seed),
projects support and query pairs through the frozen basis, and compares five
initialisations under a budget of 0 / 5 / 20 / 50 Adam updates (lr 0.1).

Support sets are nested: the original 9 few-shot prompts are always the first 9,
extended with fresh prompts up to 150; N in {5, 15, 50, 150}.  The query set is
all 1,000 test prompts.

    python scripts/run_init_ablation.py --data-dir data --support 5 15 50 150
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import load_personalllm
from lore import LoReConfig, build_basis, fit_fewshot, fit_joint
from lore.data import best_worst_pairs, personalllm_deltas
from lore.init import METHODS, adapt, initial_weights
from lore.utils import get_device


def projected(bm, W_true, prompts, V, scores, emb, batch=8):
    out = []
    for s in range(0, len(W_true), batch):
        c, r = best_worst_pairs(scores, W_true[s:s + batch], prompts[s:s + batch])
        D = personalllm_deltas(emb, prompts[s:s + batch], c, r)
        out.append(D @ V)
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--support", type=int, nargs="+", default=[5, 15, 50, 150])
    ap.add_argument("--out", default="results/init_ablation_rerun.csv")
    args = ap.parse_args()
    device = get_device()
    rows = []

    for seed in args.seeds:
        bm = load_personalllm(Path(args.data_dir), 0.1, seed)
        cfg = LoReConfig(K=5, joint_steps=2000, fewshot_steps=1000, seed=seed)
        Xs, us = bm.seen_train
        joint = fit_joint(build_basis("linear", 4096, 5), Xs, us, bm.n_seen, cfg, device)
        V = joint["basis"].V.detach().cpu()
        W_seen = torch.softmax(joint["W_logits"].cpu(), 1)

        pop = bm.population
        n_prompts = bm.train_scores.shape[0]
        support_prompts = np.empty((bm.n_unseen, max(args.support)), dtype=np.int64)
        for i, orig in enumerate(pop["fewshot_prompts"]):
            rng = np.random.default_rng(10_000_000 + seed * 1000 + i)
            extra = rng.choice(np.setdiff1d(np.arange(n_prompts), orig), max(args.support) - len(orig), replace=False)
            support_prompts[i] = np.concatenate([orig, extra])
        Zs_all = projected(bm, pop["W_unseen_true"], support_prompts, V, bm.train_scores, bm.train_emb)
        query_prompts = np.tile(np.arange(len(bm.test_scores)), (bm.n_unseen, 1))
        Zq = projected(bm, pop["W_unseen_true"], query_prompts, V, bm.test_scores, bm.test_emb)

        for n in args.support:
            Zs = Zs_all[:, :n]
            for m in METHODS:
                w0 = initial_weights(m, Zs, W_seen, seed)
                for r in adapt(Zs, Zq, w0, steps=50, lr=0.1, eval_at=(0, 5, 20, 50)):
                    rows.append({"seed": seed, "support_size": n, "initialization": m, **r})
            print(f"seed {seed} N={n} done")
        pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
