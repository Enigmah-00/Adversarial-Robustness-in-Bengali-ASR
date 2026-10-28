# Data

Model checkpoints (~60 GB) and per-utterance result files are hosted on Kaggle,
not in this repository. This directory holds one result file as a schema
reference and a script to fetch the rest.

## What is here

| file | |
|---|---|
| `results/eps50_rung100_milestone-3500_T5_n100.csv` | one complete result file, committed so the schema is readable without a Kaggle account |
| `fetch_results.py` | downloads the remaining result files from Kaggle |

## Result file schema

One row per attacked utterance, 100 rows per condition.

| column | meaning |
|---|---|
| `utt_id` | utterance identifier; **join on this, never on row position** — files are duration-sorted |
| `rung` | training rung in hours (1, 5, 25, 100) |
| `edit_type` | `T5` full retarget, `T1` negation deletion |
| `checkpoint` | `final` or `milestone-N` |
| `duration` | clip length in seconds |
| `source_text` | the true transcript |
| `target_text` | the attacker's chosen transcript |
| `source_tokens`, `target_tokens` | token counts |
| `eps50` | smallest L∞ bound achieving verified targeted success; empty if censored |
| `censored` | `True` if the attack never succeeded within the ceiling |
| `eps50_per_token` | `eps50` divided by `target_tokens` |
| `bracket_width` | final width of the bisection bracket |
| `snr_db` | signal-to-noise ratio of the successful perturbation |
| `decoded` | what the model actually output |
| `exact_match` | whether `decoded` equals `target_text` after Unicode normalisation |
| `model_dir` | the checkpoint attacked, recorded per row |

Two things to know before using these files.

**Rows are duration-sorted, not in manifest order.** Joining two conditions by
row index silently pairs different utterances. Always join on `utt_id`.

**Censored rows carry a high `snr_db`** (68–88 dB), which is the 16-bit
quantisation floor of the unperturbed audio, not a perturbation. Exclude
censored rows from any SNR summary.

## Fetching the rest

Needs a Kaggle account and an API token at `~/.kaggle/kaggle.json`.

```bash
pip install kaggle
python data/fetch_results.py
```

The script downloads only the small result files, not the multi-gigabyte
checkpoints. See `../docs/kaggle-inventory.md` for the full list, including the
checkpoints.

## The frozen target set

`task_b_targets_n100.csv` ships inside every `eps50-*-n100` dataset. Its
SHA-256 is

```
8711dc56202b33f90e769c9735231b0fa8cf27fcbc20151f5466bcd9cc2b1256
```

Verify any copy with `sha256sum`. Every notebook checks this hash at startup and
refuses to run on a mismatch.
