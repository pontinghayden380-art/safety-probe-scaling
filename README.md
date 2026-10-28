# Mini scaling law for safety probes

Linear probes for **jailbreak vs. benign** detection on small open models, with learning curves (error vs. training-set size), a model-size trend, out-of-distribution (OOD) tests, and comparisons to lexical and embedding baselines. Everything runs in one notebook: `mini_scaling_law_safety_probes.ipynb` (designed for Google Colab).

## What is in the notebook

| Section | What it does |
|---|---|
| 1-2 | Data: deduplicated `jackhhao/jailbreak-classification`, split **grouped by prompt template**; OOD sets (in-the-wild jailbreaks, deepset prompt-injection) filtered for exact / template / near-duplicate overlap |
| 3-4 | Activations from Pythia (70m, 160m, 410m, 1b), GPT-2 small, Qwen2.5-0.5B; token-level truncation (128 tokens); cached to disk |
| 5-8 | Learning curves, Hill/power-law fits, full-budget comparison with paired-bootstrap CIs, Pythia size trend |
| 9 | Non-adversarial perturbation test (appended text, lowercase/no-punctuation) with Bonferroni-corrected CIs |
| 10 | Split robustness (5 grouped splits) |
| 11 | Power-law fits, language check, attack-vs-benign shift split, stronger baselines (char n-gram TF-IDF, MiniLM), mean-pooled probes |
| 12 | Mean-pooled readout as the main analysis, with CIs |
| 13 | Benign-coverage test; layer sweep with the layer chosen on in-distribution validation only |

Baselines: TF-IDF on full text, TF-IDF on the same truncated text the models see, char n-gram TF-IDF, and a MiniLM-embedding probe.

## How to run (Colab)

1. Upload the notebook to Colab and choose a **GPU runtime**.
2. Run all cells top to bottom. The first cell after the config mounts Google Drive and stores cached features and checkpoints in `MyDrive/safety_probe_cache`, so a Colab timeout does not lose progress. If Drive cannot be mounted it falls back to `/content` (lost on timeout).
3. Order matters: Section 11d defines `strong` (needed by 12b) and Section 11e creates the mean-pooled features (needed by Section 12 and 13a). 13b must run before 13c.
4. To shorten a run, set `N_SEEDS = 5` and `SPLIT_SEEDS = [0, 1, 2]` in the config cell (wider CIs).
5. Delete `curves_ckpt.pkl` / `curves_mean_ckpt.pkl` in the cache folder to force recomputation.

Everything is seeded (`SEED = 0`) and the split and bootstrap indices are deterministic. Sections 1-11 were first run twice in separate Colab sessions and reproduced the same numbers, and the complete notebook (Sections 1-13) was then run end to end in a fresh session with the key numbers matching those reported here. Python package versions are not pinned, and the datasets are downloaded live from the Hugging Face Hub, so a later run could differ if a dataset or library changes (the OOD cache key includes a hash of the sampled prompts, so a changed dataset produces new features rather than silently reusing old ones). Run `pip freeze` in your Colab session to record the exact environment.

## Main findings (Sections 1-12)

These come from one dataset (about 1,000 training prompts), one primary split, and four Pythia sizes. Treat them as a small, careful replication, not general conclusions.

- **Readout matters more than model size.** With last-token features, probes are significantly worse than TF-IDF in-distribution (AUROC 0.94-0.998 vs 0.998). With **mean-pooled** features, last-layer probes of the larger models (pythia-410m/1b, GPT-2, Qwen) match TF-IDF, char n-gram TF-IDF and MiniLM (AUROC 0.998-0.999 vs 0.997-0.998) and need far less data (error at 64 training examples about 0.01-0.03).
- **Robustness.** Mean-pooled probes lose at most 0.009 AUROC under the appended-text / lowercase perturbations. Last-token mid-layer probes lose up to 0.15 (Qwen, filler suffix). These are non-adversarial perturbations only.
- **Size trend.** In-distribution error is already near zero (0.002-0.014) for mean-pooled probes, so a model-size trend is not reliably identifiable (alpha across 5 splits: 0.115 +/- 0.087 mid layer, 0.170 +/- 0.089 last layer).
- **OOD, in-the-wild jailbreaks.** At best parity with the baselines (best probes AUROC about 0.76; MiniLM 0.786). Error stays around 0.23-0.25 regardless of training size. The attack-vs-benign decomposition suggests most of the drop comes from realistic benign prompts being scored as attacks, not from unseen attack styles (the decomposition mixes datasets, so treat it as directional).
- **OOD, deepset prompt injection.** Mid-layer mean-pooled probes beat all baselines for every model (for example Qwen mid 0.947 vs best baseline 0.788). This is a task shift (injection, not only jailbreak); the advantage persists on the English-only subset.

## Final checks (Section 13)

- **Benign-coverage test (13a).** Adding in-the-wild prompts to training lowers held-out in-the-wild error (group-disjoint held-out half, n=411). For mean-pooled probes, error goes from 0.24-0.31 with no extra prompts to 0.18-0.26 with 25 extra benign prompts and 0.15-0.19 with 200 benign + 200 attacks (TF-IDF: 0.29 to 0.19). Most of the gain arrives with the first 25-50 benign prompts, consistent with benign-prompt shift being a large part of the OOD gap, but error is still about 0.15 with in-domain data from both classes, so coverage does not explain all of it. In-distribution error stays within about 0.01 of its starting value. Extra in-the-wild data does not help the deepset set, and last-layer probes get worse there (Qwen last: 0.17 to 0.26 at 200 benign prompts); mid-layer probes stay roughly flat. These numbers are means over 3 seeds without confidence intervals.
- **Layer chosen on in-distribution validation only (13c).** Grouped CV on the training split selects mid-depth layers (relative depth 0.46-0.58) for pythia-160m/410m/1b and GPT-2, but late layers for pythia-70m (5 of 6) and Qwen (23 of 24), because in-distribution AUROC differs by only about 0.003 between layers (saturated). With these selected layers the deepset AUROC (0.835-0.926) still exceeds every baseline (best baseline 0.788), but Qwen's high mid-layer value (0.955) is not recovered (0.889). On the in-the-wild set the selected layers stay at roughly baseline level (0.72-0.77; MiniLM 0.786). So the OOD advantage of mid-depth layers cannot be reliably predicted from in-distribution validation here; treat it as exploratory. 13c uses a fast probe (C=0.01) and reports no confidence intervals.

## Limitations

- The mean-pooled readout and mid layer were examined after seeing OOD results, so the OOD comparisons are **exploratory**. Section 13c selects the layer using in-distribution validation only; it recovers the mid-depth choice for most models but not for Qwen, and the readout choice was still made after looking at OOD.
- Template grouping is a heuristic (first 8 words); near-duplicates may remain.
- Small datasets, base models only, and a single in-distribution source.
- Perturbation tests are not adversarial. No optimized attack against the probe is evaluated.
- The language check on the deepset set is a stopword heuristic.

## Data sources (check licenses and terms before redistributing anything)

- `jackhhao/jailbreak-classification` (Hugging Face)
- `TrustAIRLab/in-the-wild-jailbreak-prompts` (Hugging Face; configs `jailbreak_2023_12_25`, `regular_2023_12_25`)
- `deepset/prompt-injections` (Hugging Face)
- Models: EleutherAI Pythia, GPT-2, Qwen2.5-0.5B, `sentence-transformers/all-MiniLM-L6-v2`

The notebook downloads these at run time; no data or model weights are included in this repository.
