#!/usr/bin/env python
"""Experiment 1: reproduce LoRe (linear basis, K=5) on PersonalLLM and Reddit TL;DR.

Examples
--------
    # PersonalLLM, two diversity levels, five seeds (joint 2000 / few-shot 1000 steps)
    python scripts/run_reproduction.py --benchmark personalllm --alphas 0.1 0.001

    # Reddit TL;DR, 100 and 150 seen-user examples (joint 1000 / few-shot 500 steps)
    python scripts/run_reproduction.py --benchmark reddit --n-seen 100 150

Writes one row per run to ``results/reproduction_<benchmark>.csv``.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from _common import evaluate_chunks, load_personalllm, load_reddit
from lore import LoReConfig, build_basis, fit_fewshot, fit_joint
from lore.utils import get_device


def run_one(bm, cfg, device):
    t0 = time.time()
    Xs, us = bm.seen_train
    joint = fit_joint(build_basis("linear", Xs.shape[1], cfg.K), Xs, us, bm.n_seen, cfg, device)
    Xf, uf = bm.unseen_fewshot
    few = fit_fewshot(joint["basis"], Xf, uf, bm.n_unseen, cfg, device=device)
    seen = evaluate_chunks(joint["basis"], joint["W_logits"], bm.seen_eval, cfg.logit_scale, device=device)
    unseen = evaluate_chunks(joint["basis"], few["W_logits"], bm.unseen_eval, cfg.logit_scale, device=device)
    return {
        "seen_acc": 100 * seen["mean_accuracy"],
        "unseen_acc": 100 * unseen["mean_accuracy"],
        "overall_acc": 50 * (seen["mean_accuracy"] + unseen["mean_accuracy"]),
        "seen_user_sd": 100 * seen["user_std"],
        "unseen_user_sd": 100 * unseen["user_std"],
        "joint_final_loss": joint["loss_history"][-1],
        "fewshot_final_loss": few["loss_history"][-1],
        "runtime_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["personalllm", "reddit"], required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.001])
    ap.add_argument("--n-seen", type=int, nargs="+", default=[100, 150])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    device = get_device()
    rows = []
    if args.benchmark == "personalllm":
        for alpha in args.alphas:
            for seed in args.seeds:
                cfg = LoReConfig(K=args.K, joint_steps=2000, fewshot_steps=1000, seed=seed)
                bm = load_personalllm(Path(args.data_dir), alpha, seed)
                r = {"alpha": alpha, "seed": seed, **run_one(bm, cfg, device)}
                print(r)
                rows.append(r)
                del bm
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        for n in args.n_seen:
            for seed in args.seeds:
                cfg = LoReConfig(K=args.K, joint_steps=1000, fewshot_steps=500, seed=seed)
                bm = load_reddit(Path(args.data_dir), n, seed)
                r = {"n_seen_examples": n, "seed": seed, **run_one(bm, cfg, device)}
                print(r)
                rows.append(r)

    out = Path(args.out_dir) / f"reproduction_{args.benchmark}_rerun.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved", out)


if __name__ == "__main__":
    main()
