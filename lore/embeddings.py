"""Frozen reward-model embeddings ``e(x, y) in R^4096``.

Following the paper, ``e`` is the final-token last hidden state of
``Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`` for the chat-formatted conversation
``[user: x, assistant: y]``.  The model's scalar reward is ``score.weight @ e``, so
the reference model is itself a linear function of the same embedding (we use this
for the "Skywork baseline" and the fixed-anchor basis).

Requires a GPU and ``transformers``; everything downstream only needs the cached
``.safetensors`` / ``.pkl`` files this module writes.

Note on batching: right-padded batched inference in bf16 differs from single-sequence
inference by a relative L2 error of ~1e-2 (cosine similarity 0.99995).  We checked
this in notebook 01 and accepted it; it is far below the effects we study.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Callable, List, Sequence

import torch

MODEL_NAME = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
MODEL_REVISION = "67e958fb067e52fb28aaa8799f3bbd70dbe29923"
HIDDEN = 4096


class SkyworkEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, device_map: str = "auto", dtype=torch.bfloat16):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map, attn_implementation="sdpa"
        ).eval()
        self.device = next(self.model.parameters()).device
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        assert self.model.config.hidden_size == HIDDEN

    def tokenize(self, prompt: str, response: str) -> torch.Tensor:
        msgs = [{"role": "user", "content": str(prompt or "")},
                {"role": "assistant", "content": str(response or "")}]
        return self.tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt")[0]

    @torch.inference_mode()
    def embed_sequences(self, seqs: Sequence[torch.Tensor], batch_size: int = 8) -> torch.Tensor:
        """Last-valid-token hidden states, length-sorted to minimise padding."""
        out = torch.empty((len(seqs), HIDDEN), dtype=torch.bfloat16)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        for s in range(0, len(order), batch_size):
            pos = order[s:s + batch_size]
            batch = [seqs[i] for i in pos]
            L = max(len(x) for x in batch)
            ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, x in enumerate(batch):
                ids[j, : len(x)] = x
                mask[j, : len(x)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            h = self.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            last = mask.sum(1) - 1
            vec = h[torch.arange(len(batch), device=self.device), last].to(torch.bfloat16).cpu()
            if not torch.isfinite(vec.float()).all():
                raise ValueError("Non-finite embedding")
            out[torch.tensor(pos)] = vec
        return out


def extract_in_chunks(
    n_items: int,
    make_sequences: Callable[[int, int], List[torch.Tensor]],
    embedder: SkyworkEmbedder,
    out_dir: str | os.PathLike,
    chunk: int = 512,
    batch_size: int = 8,
    seqs_per_item: int = 1,
) -> torch.Tensor:
    """Resumable extraction: each chunk is written atomically and skipped if complete.

    ``make_sequences(start, end)`` must return ``seqs_per_item * (end - start)``
    tokenised sequences (e.g. 2 per Reddit comparison: winner then loser)."""
    from safetensors.torch import load_file, save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for cid, start in enumerate(range(0, n_items, chunk)):
        end = min(start + chunk, n_items)
        path = out_dir / f"chunk_{cid:05d}.safetensors"
        if path.exists():
            try:
                emb = load_file(str(path))["embeddings"]
                if emb.shape[0] == (end - start) * seqs_per_item:
                    parts.append(emb)
                    continue
            except Exception:
                pass
        seqs = make_sequences(start, end)
        emb = embedder.embed_sequences(seqs, batch_size)
        tmp = path.with_suffix(".tmp")
        save_file({"embeddings": emb.contiguous()}, str(tmp))
        os.replace(tmp, path)
        parts.append(emb)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(parts)


def reward_head(model_name: str = MODEL_NAME, revision: str = MODEL_REVISION) -> torch.Tensor:
    """Download only the last checkpoint shard and return ``score.weight`` (shape [4096])."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(model_name, "model-00004-of-00004.safetensors", revision=revision)
    with safe_open(path, framework="pt", device="cpu") as f:
        w = f.get_tensor("score.weight").float()
    assert w.shape == (1, HIDDEN)
    return w.squeeze(0).contiguous()
