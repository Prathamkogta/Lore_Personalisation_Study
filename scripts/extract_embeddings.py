#!/usr/bin/env python
"""Extract frozen Skywork-Reward-Llama-3.1-8B embeddings (GPU, ~15 GB VRAM in bf16).

    python scripts/extract_embeddings.py --dataset personalllm --out data
    python scripts/extract_embeddings.py --dataset reddit      --out data
    python scripts/extract_embeddings.py --dataset reward-head --out data

PersonalLLM: 9,402 + 1,000 prompts x 8 responses -> train/test .safetensors and the
10 reward-model scores.  Reddit: worker split (see ``lore.data.reddit_worker_split``),
then embeddings for the selected 3,865 training and 10,275 validation comparisons.
Extraction is resumable (chunked, atomic writes).  On an A100 the whole thing takes
roughly 2-3 hours.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore.data import personalllm_reward_scores, reddit_worker_split  # noqa: E402
from lore.embeddings import SkyworkEmbedder, extract_in_chunks, reward_head  # noqa: E402


def personalllm(out: Path, batch: int):
    from datasets import load_dataset
    from safetensors.torch import save_file

    ds = load_dataset("namkoong-lab/PersonalLLM")
    emb = SkyworkEmbedder()
    d = out / "personalllm"
    d.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        data = ds[split]

        def seqs(a, b, data=data):
            return [emb.tokenize(data[i // 8]["prompt"], data[i // 8][f"response_{i % 8 + 1}"]) for i in range(a, b)]

        E = extract_in_chunks(len(data) * 8, seqs, emb, d / "chunks" / split, batch_size=batch)
        save_file({"embeddings": E.contiguous()}, str(d / f"{split}.safetensors"))
        print(split, tuple(E.shape))
    np.savez_compressed(d / "reward_scores.npz",
                        train_scores=personalllm_reward_scores(ds["train"]),
                        test_scores=personalllm_reward_scores(ds["test"]))


def reddit(out: Path, batch: int):
    from datasets import load_dataset

    ds = load_dataset("openai/summarize_from_feedback", "comparisons")
    split = reddit_worker_split(ds["train"]["worker"], ds["validation"]["worker"])
    d = out / "reddit"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(split, open(d / "user_split.json", "w"), indent=2)
    emb = SkyworkEmbedder()

    def build(hf, indices, name):
        rows = sorted({int(i) for v in indices.values() for i in v})

        def seqs(a, b):
            out_ = []
            for idx in rows[a:b]:
                r = hf[idx]
                c = int(r["choice"])
                out_ += [emb.tokenize(r["info"]["post"], r["summaries"][c]["text"]),
                         emb.tokenize(r["info"]["post"], r["summaries"][1 - c]["text"])]
            return out_

        # two sequences per comparison (winner, loser), kept interleaved
        E = extract_in_chunks(len(rows), seqs, emb, d / "chunks" / name, batch_size=batch, seqs_per_item=2)
        E = E.reshape(len(rows), 2, -1)
        by_worker = {}
        for j, idx in enumerate(rows):
            r = hf[idx]
            by_worker.setdefault(str(r["worker"]), []).append(
                {"embeddings": {"winning": [E[j, 0].clone()], "losing": [E[j, 1].clone()]}})
        by_worker = dict(sorted(by_worker.items(), key=lambda kv: len(kv[1])))
        with open(d / f"tldr_embeddings_{name}.pkl", "wb") as f:
            pickle.dump(by_worker, f, protocol=pickle.HIGHEST_PROTOCOL)

    build(ds["train"], {**split["seen_train_indices"], **split["unseen_fewshot_indices"]}, "train")
    build(ds["validation"], split["val_eval_indices"], "val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["personalllm", "reddit", "reward-head"], required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    if args.dataset == "personalllm":
        personalllm(out, args.batch_size)
    elif args.dataset == "reddit":
        reddit(out, args.batch_size)
    else:
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"score_weight": reward_head()}, out / "skywork_reward_head.pt")
        print("Saved", out / "skywork_reward_head.pt")


if __name__ == "__main__":
    main()
