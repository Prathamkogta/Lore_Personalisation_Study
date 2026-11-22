# Recorded results

Every number in the report, copied from the outputs of the notebooks in `../notebooks/`.
All accuracies are in %, macro-averaged over users.

| File | Content | Source |
|---|---|---|
| `reproduction_personalllm.csv` | Reproduction, per seed, α ∈ {0.1, 0.001} | notebook 02, cell 3 |
| `reproduction_reddit.csv` | Reproduction, per seed, 100 / 150 seen examples | notebook 02, cell 9 |
| `paper_reported.csv` | LoRe and baselines as reported in the paper (Tables 1–3) | paper |
| `architecture_ablation_summary.csv` | Exp. 1, mean / seed-SD and parameter counts, incl. frozen Skywork | notebook 03, cell 20 |
| `architecture_ablation_per_seed.csv` | Exp. 1, per seed, all four splits; includes first-pass (naive) MLPs | notebook 03, cells 15 and 19 |
| `shortcut_first_pass.csv` | 100 % results of the non-odd first pass | notebook 03, cell 15 |
| `synthetic_shortcut.csv` | standard vs symmetric accuracy on synthetic users | `scripts/demo_synthetic.py` |
| `init_ablation_summary.csv` | Exp. 2, query accuracy by support size × init × updates | notebook 04, cells 13, 16, 17 |
| `init_paired_vs_random.csv` | Exp. 2, paired improvement over random logits with 95 % t-CI | notebook 04, cell 18 |
| `init_centroid0_vs_adapted.csv` | Exp. 2, zero-update centroid vs adapted baselines (N = 5) | notebook 04, cell 19 |

Notes

* In `init_ablation_summary.csv`, rows with `derived=True` (random logits at N = 50 and 150) are computed
  as `uniform − (uniform − random)` from the paired table. `seed_sd` is missing where the notebook did
  not record it.
* The Fixed-Anchor-Residual per-seed runs were reused from an earlier cell, so only their summary
  (mean ± SD) is available.
