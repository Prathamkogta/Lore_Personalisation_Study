#!/usr/bin/env python
"""CPU-only sanity check on synthetic low-rank users (no downloads, ~1 minute).

Reproduces, in miniature, the orientation shortcut from the function-class ablation:
a non-odd network trained on chosen-minus-rejected differences scores ~100% on the
usual metric, but ~50% once every test pair is also shown in swapped order.

    python scripts/demo_synthetic.py --out results/synthetic_shortcut.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore import LoReConfig, build_basis, evaluate_users, fit_fewshot, fit_joint  # noqa: E402
from lore.synthetic import SyntheticConfig, make_synthetic  # noqa: E402
from lore.utils import pack_users  # noqa: E402

ARCHS = {"linear": 0.05, "mlp8": 0.01, "naive_mlp8": 0.01}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/synthetic_shortcut.csv")
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        data = make_synthetic(SyntheticConfig(D=128, seed=seed))
        Xs, us, _ = pack_users(data["seen_train"])
        Xt, ut, _ = pack_users(data["seen_test"])
        Xf, uf, _ = pack_users(data["unseen_fewshot"])
        Xq, uq, _ = pack_users(data["unseen_test"])
        for arch, lr in ARCHS.items():
            cfg = LoReConfig(K=4, joint_steps=400, joint_lr=lr, fewshot_steps=300,
                             fewshot_lr=0.1, logit_scale=1.0, seed=seed)
            res = fit_joint(build_basis(arch, 128, 4), Xs, us, len(data["seen_train"]), cfg)
            few = fit_fewshot(res["basis"], Xf, uf, len(data["unseen_fewshot"]), cfg)
            row = {"seed": seed, "architecture": arch}
            for split, X, u, W in [("seen", Xt, ut, res["W_logits"]), ("unseen", Xq, uq, few["W_logits"])]:
                for sym in (False, True):
                    acc = evaluate_users(res["basis"], X, u, W, 1.0, symmetric=sym)["mean_accuracy"]
                    row[f"{split}_{'symmetric' if sym else 'standard'}"] = 100 * acc
            rows.append(row)
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\nMean over seeds:\n", df.groupby("architecture").mean(numeric_only=True).drop(columns="seed").round(2))


if __name__ == "__main__":
    main()
