# Kaggle inventory

Model checkpoints are 5–8.8 GB each and are not in this repository. Every
per-utterance result file the paper uses is committed under
[`../data/results/`](../data/results/); the pages below are the original
Kaggle records of those files and the notebooks that produced them.

Every link below was opened signed-out on 17 November 2026 and returned its
public page. For the result datasets, Kaggle's public file list was also
compared with this repository: each file in `data/results/` that came from one
of them has the same name and the same size in bytes. For notebook output,
Kaggle lists file names only; those names match.

## Model checkpoints

| dataset | what it is |
|---|---|
| `mahdihasanqurishi/ckpt-xlsr-1h` | 1 h rung, seed 1337 — the original ladder |
| `mahdihasanqurishi/ckpt-xlsr-5h` | 5 h rung, seed 1337 |
| `mahdihasanqurishi/ckpt-xlsr-25h` | 25 h rung, seed 1337 |
| `mahdihasanqurishi/ckpt-xlsr-100h` | 100 h rung, seed 1337, with the three milestone checkpoints |
| `mahdihasanqurishi/ckpt-xlsr-1h-traj` | the 1 h re-run, same seed 1337, which diverged |
| `zoayriaabedin/ckpt-xlsr-1h-s1338` | 1 h replicate |
| `zoayriaabedin/ckpt-xlsr-1h-s1339` | 1 h replicate |
| `zoayriaabedin/ckpt-xlsr-5h-s1338` | 5 h replicate |
| `mahdihasanqurishi/ckpt-xlsr-5h-s1339` | 5 h replicate |
| `mahdihasanqurishi/ckpt-xlsr-25h-s1338` | 25 h replicate |
| `mahdihasanqurishi/ckpt-xlsr-25h-s1339` | 25 h replicate |
| `mahdihasanqurishi/ckpt-xlsr-100h-s1338` | 100 h replicate, seed 1338, with milestones at steps 1000, 2000 and 3500 |

Twelve training runs: four at 1 hour, three each at 5 and 25 hours, and two at
100 hours. The `model_dir` column of every result file names one of these
twelve.

## Splits and evaluation sets

| dataset | what it is |
|---|---|
| `mahdihasanqurishi/bn-asr-splits` | the frozen train, development and test manifests, plus the vocabulary |
| `zoayriaabedin/bn-attack-eval-100` | the frozen 100-utterance evaluation set |
| `mahdihasanqurishi/bn-attack-targets` | the 38-utterance negation set, its T1 and T5 targets and the native speaker's decision on each candidate |

## Attack results (T5, 100 utterances)

```
zoayriaabedin/eps50-rung1-final-t5-n100
zoayriaabedin/eps50-rung5-final-t5-n100
zoayriaabedin/eps50-rung25-final-t5-n100
zoayriaabedin/eps50-rung100-milestone-1000-t5-n100
zoayriaabedin/eps50-rung100-milestone-2000-t5-n100
zoayriaabedin/eps50-rung100-milestone-3500-t5-n100
mahdihasanqurishi/eps50-rung100-final-complete
mahdihasanqurishi/eps50-queue-n100
mahdihasanqurishi/eps50-s1338-ms1000-ms2000partial-n100
mahdihasanqurishi/eps50-s1338-ms2000-n100
mahdihasanqurishi/eps50-s1338-ms3500-n100
```

* `eps50-rung100-final-complete` is the 100-hour final on all 100 utterances
  (44.0%). `eps50-queue-n100` holds the 1-hour re-run
  (`eps50_rung1-trajfinal_T5_n100.csv`), but it also carries an **80-row
  partial** of the 100-hour final under the same file name. Use the complete
  dataset for that file.
* The 100-hour replicate was attacked in three NB-21 sessions.
  `eps50-s1338-ms1000-ms2000partial-n100` holds the full `ms1000` result and an
  **80-row partial** of `ms2000`. The complete `ms2000` and `ms3500` files are
  in `eps50-s1338-ms2000-n100` and `eps50-s1338-ms3500-n100`. Do not attach the
  partial beside the complete file.

The 1-, 5- and 25-hour replicates are notebook output of NB-16, one notebook
per session:

```
zoayriaabedin/nb-16-eps50-replicates-runa
zoayriaabedin/nb-16-eps50-replicates-runb
mahdihasanqurishi/nb-16-eps50-replicates-runc
mahdihasanqurishi/nb-16-eps50-replicates-rund
```

## Re-attacks of byte-identical weights (NB-23)

```
mahdihasanqurishi/attack-replicate-a2026-sessiona
mahdihasanqurishi/attack-replicate-a2026-sessionb
```

## Stronger search on censored utterances (notebook output)

| notebook | output used |
|---|---|
| `zoayriaabedin/nb-24-attack-strength-check` | `strength_rung1-final_T5.csv` |
| `mahdihasanqurishi/nb-24b-false-negative-rate` | `strength_rung25-final_T5.csv` and the 16-row `strength_rung100-ms3500_T5.csv` |

NB-24's output also holds a 4-row `strength_rung100-ms3500_T5.csv`, a partial
with the same name. NB-24b's 16-row file includes those 4 rows; use it.

## Decoder margins (NB-25)

`mahdihasanqurishi/margin-covariate` — `margins_n100.csv` and
`margins_by_checkpoint.csv`.

## Clean word error rates

| notebook | dataset |
|---|---|
| `mahdihasanqurishi/nb-02-clean-eval-isower-ipynb` | `mahdihasanqurishi/clean-wer-results` |
| `mahdihasanqurishi/nb-02b-clean-eval-n100` | `mahdihasanqurishi/clean-wer-results-n100` |
| `mahdihasanqurishi/nb-02c-clean-wer-traj-error-free` | notebook output, `clean_wer_traj.csv` |
| NB-15 | `mahdihasanqurishi/clean-wer-replicates-a`, `mahdihasanqurishi/clean-wer-replicates-b` |
| `mahdihasanqurishi/nb-20-clean-wer-100h-replicate` | `mahdihasanqurishi/clean-wer-replicates-c` |

## Negation (38 utterances)

| source | what it holds |
|---|---|
| `mahdihasanqurishi/eps50-rung100-final-t1-paired38` | T1 on the 100-hour final, published as `eps50_rung100_final_T1.csv` |
| `mahdihasanqurishi/eps50-rung100-final-t5-paired38` | T5 on the 100-hour final, published as `eps50_rung100_final_T5.csv` |
| `mahdihasanqurishi/nb-13-t1-vs-t5-low-rung`, version `scriptVersionId=354188660` | T1 and T5 on the 1-hour final, notebook output |
| `zoayriaabedin/eps50-t6-wordcontrol-n38` | the T6 control on both models, and its targets |
| `mahdihasanqurishi/nb-03-attack-targets-ipynb` | NB-03, which built the set and records the candidate counts |

The two 100-hour files are renamed here to the `_n38` convention the other
negation files use. The 1-hour T1 and T5 files have no dataset; link the
pinned NB-13 version, so a later re-run cannot change what the link shows.

## Language-model sweep (NB-06)

`mahdihasanqurishi/nb-06-lm-weight-sweep` and its dataset
`mahdihasanqurishi/lm-sweep-results`. A separate dataset with the same name
exists under `zoayriaabedin`, created about 12 hours earlier with smaller
sweep files; it is not the one the paper uses.

## Codec chain (NB-07)

| notebook | dataset |
|---|---|
| `mahdihasanqurishi/nb-07-pipeline-survival-1h-ipynb` | `mahdihasanqurishi/pipeline-survival-rung1` |
| `mahdihasanqurishi/nb-07-pipeline-survival-5h-ipynb` | `mahdihasanqurishi/pipeline-survival-rung5` |
| `zoayriaabedin/nb-07-pipeline-survival-25h` | `mahdihasanqurishi/pipeline-survival-rung25` |
| `zoayriaabedin/nb-07-pipeline-survival-100h` | `mahdihasanqurishi/pipeline-survival-rung100` |

## Over the air

The sessions ran in `zoayriaabedin/taskasecurity`. Recordings:
`zoayriaabedin/ota-recordings-rung100` (the first 100-hour take),
`zoayriaabedin/ota-recordings-rung1-take2` and
`zoayriaabedin/ota-recordings-rung100-take2` (the second takes as recorded),
and `zoayriaabedin/ota-recordings-rung1-take22` and
`zoayriaabedin/ota-recordings-rung100-take22` (the second takes as the notebook
wrote them). The first 1-hour take's recordings are not public. The paper
reports session results, not recordings.

## Attack notebooks

```
mahdihasanqurishi/taskbsecurity2
zoayriaabedin/taskbsecurity2
```

## One dataset with a broken archive

`zoayriaabedin/eps50-rung1-traj-final-t5-n100` — the page loads but the
download returns 404. Kaggle reports *"Data not available. This dataset may
need its archive recreated."*

**Nothing in the paper depends on it, and its one useful file is already here**
as `data/results/eps50_rung1-traj-SECOND-MEASUREMENT_T5_n100.csv`. That is the
61.0% measurement discussed in `data/README.md`.

We have deliberately not recreated the archive. Doing so would require
uploading a new version, and the original `_provenance.json` for that run is
inside the archive we cannot download — so a rebuilt version would silently
lose the provenance record it is supposed to carry. A broken archive that
nobody needs is better than a dataset whose provenance no longer matches its
contents.

> **Naming hazard, which is why that file is renamed here.** On Kaggle it is
> called `eps50_rung1_final_T5_n100.csv` — the same name as the 1-hour rung's
> own result, for a different model. Any analysis that looks results up by
> filename alone takes whichever mounts first.

## Audio

The audio is the **Bengali.AI Speech Recognition** competition dataset on
Kaggle. It is not redistributed here or in any dataset above. Accept the
competition rules to access it.
