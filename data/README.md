# Data

Per-utterance attack results, the frozen target set, and the provenance record
for every run. Model checkpoints are 5–8.8 GB each and stay on Kaggle; see
[`../docs/kaggle-inventory.md`](../docs/kaggle-inventory.md).

`python3 analysis/verify_all.py .`, run from the repository root, re-derives
from these files the figures the paper quotes and exits non-zero if any one
disagrees. Its checks on the manuscript text are skipped here, because the
manuscript is not in this repository.

## Layout

```
targets/task_b_targets_n100.csv    the frozen attack targets
results/eps50_*_T5_n100.csv        one file per attacked checkpoint, 100 rows each
results/attackrep_*.csv            re-attacks of byte-identical weights (NB-23)
results/strength_*.csv             stronger-search re-attacks of censored utterances (NB-24, NB-24b)
results/margins_*.csv              decoder confidence margins (NB-25)
results/eps50_*_T1_n38.csv         negation: deletion of the particle, 38 utterances
results/eps50_*_T5_n38.csv         negation: full retarget of the same 38
results/eps50_*_T6_n38.csv         negation: the matched non-negation deletion control (NB-26)
results/attack_eval_t1.csv         the 38 negation utterances
results/bn_attack_targets.csv      their T1 and T5 targets
results/t6_targets_n38.csv         their T6 targets
results/t1_pairs_validated.csv     the native speaker's decision on each candidate shown:
                                   y accepted, n rejected, blank never decided (one row)
provenance/*.json                  package versions, GPU, seed and config per run
```

## Verifying the target set

```
SHA-256  8711dc56202b33f90e769c9735231b0fa8cf27fcbc20151f5466bcd9cc2b1256
```

```bash
sha256sum data/targets/task_b_targets_n100.csv
```

Every notebook recomputes this at startup and refuses to run on a mismatch. The
copy here was checked against all six Kaggle datasets that ship it; all six are
byte-identical.

## Verifying the results

Each file reproduces the attack-success figure reported in the paper:

| file | attack success |
|---|---:|
| `eps50_rung1_final_T5_n100.csv` | 75.0% |
| `eps50_rung5_final_T5_n100.csv` | 79.0% |
| `eps50_rung25_final_T5_n100.csv` | 66.0% |
| `eps50_rung100_milestone-1000_T5_n100.csv` | 37.0% |
| `eps50_rung100_milestone-2000_T5_n100.csv` | 46.0% |
| `eps50_rung100_milestone-3500_T5_n100.csv` | 53.0% |
| `eps50_rung1-s1338_T5_n100.csv` | 57.0% |
| `eps50_rung1-s1339_T5_n100.csv` | 39.0% |
| `eps50_rung5-s1338_T5_n100.csv` | 96.0% |
| `eps50_rung5-s1339_T5_n100.csv` | 59.0% |
| `eps50_rung25-s1338_T5_n100.csv` | 78.0% |
| `eps50_rung25-s1339_T5_n100.csv` | 57.0% |
| `eps50_rung1-trajfinal_T5_n100.csv` | 41.0% |
| `eps50_rung100-s1338-ms1000_T5_n100.csv` | 34.0% |
| `eps50_rung100-s1338-ms2000_T5_n100.csv` | 48.0% |
| `eps50_rung100-s1338-ms3500_T5_n100.csv` | 56.0% |
| `eps50_rung100-final_T5_n100.csv` | 44.0% (outside the pooled fit; reported descriptively) |

Attack success is the share of rows with `censored == False`:

```python
import csv
rows = list(csv.DictReader(open("data/results/eps50_rung5-s1338_T5_n100.csv")))
cens = [r["censored"].lower() == "true" for r in rows]
print(100 * (len(cens) - sum(cens)) / len(cens))   # 96.0
```

## One file that is not a paper row

`eps50_rung1-traj-SECOND-MEASUREMENT_T5_n100.csv` is a **second measurement of
a model the paper already reports**, and it is named to make that obvious.

The paper's 1-hour "re-run" row reports **41.0%** attack success, and that file
is here too as `eps50_rung1-trajfinal_T5_n100.csv`. The
`SECOND-MEASUREMENT` file is an earlier measurement of the *same weights*,
giving **61.0%**.

Both are included because hiding the disagreement would be worse than reporting
it. Two attacks on one checkpoint, 20 points apart, is itself evidence about
measurement variance — which is the subject of the paper. The two runs differed
in `transformers` version and the earlier one's version was not recorded, so we
cannot attribute the gap and do not try.

Do not merge this file with the paper's 41.0% row. They are the same model
measured twice, not two conditions.

> **Naming hazard.** On Kaggle this file is called
> `eps50_rung1_final_T5_n100.csv` — the same name as the 1-hour rung's own
> result. Any analysis that looks results up by filename alone will silently
> take whichever mounted first. It is renamed here to prevent that.

## Clean word error rates

`clean_wer_replicates_a.csv` and `_b.csv` hold clean WER on the same 100
attacked utterances for every checkpoint. This is the covariate in the paper's
analysis of covariance. The two files overlap on `ckpt-xlsr-1h`, which was
scored in both runs as a cross-check and gives 0.6355 in each.

## Derived tables

`derived/` holds the output of `NB-17-variance-components`, which is what the
paper's tables are built from:

| file | |
|---|---|
| `runs.csv` | all 13 attacked checkpoints with exposure, clean WER and attack success |
| `sigma.csv` | between-run standard deviation: raw 20.279, bias-corrected 21.574, 4 d.f., 95% CI [12.15, 58.27] |
| `variance_by_group.csv` | the four replicate pairs behind that figure |
| `power.csv` | runs required per condition: 6, 7 and 44 for effects of 38, 33 and 13 points |
| `tables.tex` | the same numbers as LaTeX, including the pooled fit |

The pooled fit recorded in `tables.tex`:

```
beta1 = +0.0986  CI [+0.0302, +0.1670]  (G=11 training runs, N=1300, df=10)
```

That fit predates the 100-hour replicate. The paper's pooled fit adds it and
comes from `analysis/reanalysis.py`:

```
beta1 = +0.0888  CR1 CI [+0.024, +0.154]  (G=12 training runs, N=1600, df=11)
```

## The re-attack, stronger-search, margin and negation files

* `attackrep_*.csv` re-attack three replicate checkpoints with a different
  attack seed. They have the schema below plus `attack_seed`, `steps_total`,
  `n_stall` and `n_stepcap`. `analysis/attack_replication.py` compares each
  with its first measurement.
* `strength_*.csv` hold one row per censored utterance that was re-attacked:
  `was_censored`, `now_success`, `restart_used`, `steps_used` and the `eps`
  reached. NB-24b's two files add `within_published_budget`, `n_restarts`,
  `strong_steps`, `sampled_from` and `sample_seed`.
* `margins_n100.csv` gives one row per checkpoint and utterance: CTC loss on
  the reference and the decoder's per-frame margin, entropy and blank share.
  `margins_by_checkpoint.csv` averages it per checkpoint.
* The `_n38` files follow the schema below for the 38 negation utterances. T1
  and T6 were searched over [1e-5, 2e-3] in six rounds, T5 over [1e-4, 2e-3] in
  five; `bracket_width` records each. The T6 files add `tag`, `neg_token` and
  `ctl_token`, and the 1-hour T1 and T5 files add `eps_lo` and `eps_hi`. The T6
  runs used newer library versions than the T1 and T5 runs they are compared
  with.

## Schema

One row per attacked utterance, 100 rows per condition (38 in the negation
files).

| column | meaning |
|---|---|
| `utt_id` | utterance identifier; **join on this, never on row position** — files are duration-sorted |
| `rung` | training rung in hours |
| `edit_type` | `T5` full retarget, `T1` negation deletion, `T6` matched non-negation deletion |
| `checkpoint` | `final` or `milestone-N` |
| `duration` | clip length, seconds |
| `source_text` / `target_text` | true transcript, attacker's target |
| `source_tokens` / `target_tokens` | token counts |
| `eps50` | smallest L∞ bound achieving verified targeted success; empty if censored |
| `censored` | `True` if the attack never succeeded within the ceiling |
| `eps50_per_token` | `eps50 / target_tokens` |
| `bracket_width` | final bisection bracket width |
| `snr_db` | signal-to-noise ratio of the perturbation |
| `decoded` | what the model output |
| `exact_match` | `decoded == target_text` after Unicode normalisation |
| `model_dir` | the checkpoint attacked, recorded per row |

Two traps worth knowing.

**Rows are duration-sorted, not in manifest order.** Joining two conditions by
row index pairs different utterances. Join on `utt_id`.

**Censored rows carry `snr_db` of 68–88 dB.** That is the 16-bit quantisation
floor of the unperturbed audio, not a perturbation. Exclude censored rows from
any SNR summary.
