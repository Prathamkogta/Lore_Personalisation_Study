#!/usr/bin/env python
"""Experiment 1: change the representation / function class of the reward basis.

Every architecture is trained with the *same* LoRe recipe (K=5, full-batch Adam,
lr 0.5 joint for 1000 steps, lr 0.1 few-shot for 500 steps, tau=100) on the same
splits, for five seeds.  PersonalLLM uses alpha=0.1 and seed 42 for the synthetic
population (as in notebook 03); Reddit uses 150 seen-user examples.

We report both the standard accuracy and the *symmetric* accuracy (each test pair
also scored in swapped order).  The two agree for odd bases; a gap means the model
is exploiting the "label is always 1" shortcut.  Add ``--include-naive`` to train
the non-odd MLP that triggered this in our first pass.

    python scripts/run_architecture_ablation.py --benchmarks personalllm reddit
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from _common import evaluate_chunks, load_personalllm, load_reddit, reference_accuracy
from lore import LoReConfig, build_basis, count_trainable, fit_fewshot, fit_joint
from lore.models import ARCHITECTURES, count_frozen
from lore.projections import fit_pca, random_orthogonal
from lore.utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["personalllm", "reddit"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--architectures", nargs="+", default=ARCHITECTURES)
    ap.add_argument("--include-naive", action="store_true")
    ap.add_argument("--out", default="results/architecture_ablation_rerun.csv")
    args = ap.parse_args()

    device = get_device()
    anchor = torch.load(Path(args.data_dir) / "skywork_reward_head.pt")["score_weight"].float()
    archs = list(args.architectures) + (["naive_mlp8"] if args.include_naive else [])
    rows = []

    for bname in args.benchmarks:
        bm = load_personalllm(Path(args.data_dir), 0.1, 42) if bname == "personalllm" else load_reddit(Path(args.data_dir), 150, 42)
        Xs, us = bm.seen_train
        proj = fit_pca(Xs, 512)                      # fit on seen-user training data only
        proj["random512"] = random_orthogonal(Xs.shape[1], 512)
        print(f"{bm.name}: PCA-5 explains {proj['explained_variance_ratio'][:5].sum():.3f} of variance")

        ref = {"seen": reference_accuracy(anchor, bm.seen_eval, bm.n_seen),
               "unseen": reference_accuracy(anchor, bm.unseen_eval, bm.n_unseen)}
        rows.append({"benchmark": bname, "architecture": "skywork_reference", "seed": -1,
                     "seen_eval": 100 * ref["seen"], "unseen_eval": 100 * ref["unseen"]})

        for arch in archs:
            for seed in args.seeds:
                cfg = LoReConfig(K=5, joint_steps=1000, fewshot_steps=500, seed=seed)
                basis = build_basis(arch, Xs.shape[1], 5, proj, anchor)
                n_train, n_frozen = count_trainable(basis), count_frozen(basis)
                joint = fit_joint(basis, Xs, us, bm.n_seen, cfg, device)
                Xf, uf = bm.unseen_fewshot
                few = fit_fewshot(joint["basis"], Xf, uf, bm.n_unseen, cfg, device=device)
                row = {"benchmark": bname, "architecture": arch, "seed": seed,
                       "trainable_shared": n_train, "frozen_params": n_frozen}
                for split, chunks, W in [("seen_eval", bm.seen_eval, joint["W_logits"]),
                                         ("unseen_eval", bm.unseen_eval, few["W_logits"])]:
                    row[split] = 100 * evaluate_chunks(joint["basis"], W, chunks, device=device)["mean_accuracy"]
                    row[split + "_symmetric"] = 100 * evaluate_chunks(
                        joint["basis"], W, chunks, symmetric=True, device=device)["mean_accuracy"]
                for split, (X, u), W, n in [("seen_train", bm.seen_train, joint["W_logits"], bm.n_seen),
                                            ("unseen_fewshot", bm.unseen_fewshot, few["W_logits"], bm.n_unseen)]:
                    row[split] = 100 * evaluate_chunks(joint["basis"], W, lambda X=X, u=u: iter([(X, u)]),
                                                       device=device)["mean_accuracy"]
                print(row)
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.out, index=False)

    print("Saved", args.out)


if __name__ == "__main__":
    main()
