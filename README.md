# Adversarial Robustness in Bengali ASR

Code, figures and provenance for:

> **One Training Run Is Not Enough: Measuring How Training-Data Volume Affects
> Adversarial Robustness in Low-Resource Speech Recognition**
> Mahdi Hasan Qurishi, Umme Zoayria Abedin, A.K.M. Fakhrul Hossain
> Department of CSE, Shahjalal University of Science and Technology

---

## What the study found

We fine-tuned one architecture on 1, 5, 25 and 100 hours of Bengali, holding the
tokenizer, optimiser and step budget identical, and measured how hard each model
is to attack with targeted adversarial examples. Clean accuracy was neutralised
by *iso-WER matching*: each converged low-data model is compared against an
intermediate checkpoint of the 100-hour run that makes about the same number of
transcription errors.

Then, instead of running the experiment once, we ran it **ten times**.

**More training audio does make the model harder to attack.** Pooling all runs
with standard errors clustered on the training run, the smallest perturbation
achieving verified targeted success rises by a factor of **1.26 per decade** of
training audio at matched clean accuracy (β₁ = +0.0986, 95% CI [+0.030, +0.167],
11 training runs, 1,300 utterance-level observations).

**No single pair of checkpoints can show this.** Retraining a condition and
changing nothing but the run gives a between-run standard deviation in targeted
attack success of **21.6 percentage points** (95% CI [12.2, 58.3]). A matched
pair contrasts one run against one run, so its between-run standard error is
30.5 points — and the three pair effects we measured are 38, 33 and 13 points,
or 1.25, 1.08 and 0.43 times that standard error. Ten of twelve within-condition
comparisons are statistically significant, where chance would give one.

The clearest single case is a pair we never designed: two runs of the 5-hour
condition that agree on word error rate to **0.0035** — seventeen times tighter
than any pair we built on purpose — and differ by **37 points** of attack
success (McNemar exact, p = 7.8 × 10⁻¹⁰).

The practical recommendation: **train every condition at least twice and report
both runs.**

---

## Repository layout

```
notebooks/     the full pipeline, in run order (see below)
analysis/      make_figures.py — regenerates every figure from the reported numbers
figures/       the five figures in the paper, as vector PDFs
data/          one sample result file; the rest are hosted on Kaggle (see data/README.md)
docs/          inventory of the public Kaggle datasets
```

## Notebooks, in run order

| notebook | what it does | hardware | cost |
|---|---|---|---|
| `NB-00-data-prep` | builds the four nested training subsets and freezes the splits | CPU | minutes |
| `NB-01-train-ladder` | trains the original 1 / 5 / 25 / 100-hour ladder | T4 | ~3.5 h per rung |
| `NB-03-attack-targets` | generates and freezes the attack targets | CPU | minutes |
| `NB-05-eps50-batched` | ε₅₀ by bisection, the core attack measurement | T4 | ~4.6 h per condition |
| `NB-02b` / `NB-02c` | clean word error rate on the evaluation sets | T4 | minutes |
| `NB-06-lm-weight-sweep` | language-model fusion at decode time | T4 | ~1.5 h |
| `NB-07-pipeline-survival` | survival through a digital codec chain | T4 | ~1 h |
| `NB-10-over-the-air` | over-the-air playback (all four sessions failed their controls) | T4 | ~1 h |
| `NB-12-overnight` | queue runner for the first replication | T4 | ~11 h |
| `NB-13-t1-vs-t5-low-rung` | negation vs full retarget on the 1-hour model | T4 | ~3 h |
| **`NB-14-train-replicate`** | **trains one replicate at a declared seed** | T4 | ~3.1 h per run |
| **`NB-15-clean-wer-replicates`** | **clean WER for every replicate — the ANCOVA covariate** | T4 | ~4 min per checkpoint |
| **`NB-16-eps50-replicates`** | **ε₅₀ on the replicates, plus the variance analysis** | T4 | ~4.6 h per condition |
| **`NB-17-variance-components`** | **σ, the run-clustered ANCOVA, the power table** | **CPU** | ~30 s |

The four in bold are the replication study. Everything above them is the
original single-run experiment they correct.

### Reproducing the replication study

1. **`NB-14`**, six commits: `(RUNG, SEED)` ∈ {1, 5, 25} × {1338, 1339}.
   **Internet ON** — this is the only notebook that needs it, because it
   fine-tunes from `facebook/wav2vec2-xls-r-300m` on the Hugging Face Hub.
   Publish each output as its own dataset; never as a new version of an existing
   one.
2. **`NB-15`**, clean WER for every checkpoint. Internet off. The agreement
   check must reproduce 0.6355 and 0.6251 for the two published 1-hour models
   before any new number is trusted.
3. **`NB-16`**, three sessions, one rung per session so a session that dies
   loses one rung's pair rather than half of every rung. Internet off.
4. **`NB-17`**, CPU only. Its reproduction check refits the two previously
   published coefficients (+0.1517 and +0.1007) from the raw result files before
   computing anything new. If that check fails, stop.

Total: about 50 GPU-hours. Every run in this study was performed on Kaggle's
free tier; the study cost nothing in paid compute.

---

## Data

Model checkpoints and per-utterance results are too large for Git and live on
Kaggle. See [`data/README.md`](data/README.md) for the inventory, and
[`docs/kaggle-inventory.md`](docs/kaggle-inventory.md) for the dataset slugs.

### Integrity: the frozen target set

Every attack in this study uses one frozen file of 100 utterance IDs and their
attacker-chosen target transcripts, `task_b_targets_n100.csv`, with

```
SHA-256  8711dc56202b33f90e769c9735231b0fa8cf27fcbc20151f5466bcd9cc2b1256
```

Every notebook recomputes that hash at startup and refuses to run if it does not
match:

```python
SHA = hashlib.sha256(open(TGT, "rb").read()).hexdigest()
assert SHA == "8711dc56...cc2b1256", (
    "These are NOT the targets the published runs used. "
    "The comparison would be meaningless.")
```

Change one byte of the target file and every downstream run stops. That is why
13 conditions measured across five weeks are byte-comparable by construction
rather than by assurance. To verify your own copy:

```bash
sha256sum task_b_targets_n100.csv
```

---

## Software environment

The original ladder trained under `transformers` 4.44.2; the six replicates
under 5.16.1, because the version pin could not be restored in an offline
session. This is reported in the paper rather than hidden. Each replicate *pair*
shares an environment and was attacked in one session, which is why the
between-run σ is pooled within environment.

| | original ladder | replicates |
|---|---|---|
| `transformers` | 4.44.2 | 5.16.1 |
| `accelerate` | 1.13.0 | 1.14.0 |
| `torch` | 2.10.0+cu128 | 2.11.0+cu128 |
| `numpy` | 2.0.2 | 2.1.3 |
| Python | 3.11 | 3.13 |

Training is **not** deterministic at a fixed seed here, and we did not configure
it to be: mixed precision is enabled, `cudnn.benchmark` is left on so kernel
selection is autotuned by timing, and two dataloader workers interleave. Those
are ordinary settings, kept deliberately, because the question is what varies
between runs a practitioner would actually perform. Two of our runs used the
*same* seed and still diverged by 34 points of attack success.

---

## What reproduces, and what does not

This repository makes a claim about reproducibility that needs stating
carefully, because the paper's central finding is that training this model is
*not* reproducible. Both things are true. They apply at different levels.

| level | reproducible? | evidence |
|---|---|---|
| **Analysis** — same result files in, same numbers out | **Exactly** | `NB-17` refits the two previously published coefficients (+0.1517 and +0.1007) to four decimal places from the raw files, every run, before computing anything new |
| **Measurement** — same model weights, attack it again | **Within ~12 points** | two attacks on byte-identical weights whose clean WER agreed to four decimal places differed by 12 points of attack success |
| **Training** — same recipe, same seed, train again | **No** | 21.6 points of standard deviation; two runs at the *same* seed differed by 34 points |

So: **do not expect to reproduce 96%, or 59%, or any individual number in our
tables.** You will not. Neither did we, and that is the result.

What you should expect to reproduce is the *distribution*: a between-run
standard deviation of the same order as ours, a positive exposure coefficient,
and the finding that no single matched pair clears the noise. Those are the
claims the paper makes, and they are claims about a distribution rather than
about any one draw from it.

This is ordinary practice everywhere that a measured quantity is noisy. Nobody
expects to reproduce an individual sample from a process with a known standard
deviation; they expect to reproduce the mean and the spread. What is unusual in
this area is not our position — it is that the spread has not previously been
measured, so single-run numbers have been read as if the spread were zero.

---

## Figures

`analysis/make_figures.py` regenerates all five figures. Every value in it is
copied from a run log or from `NB-17`'s output; nothing is fitted, smoothed or
extrapolated except the power curve, which is a formula evaluated at the
measured σ and says so in its own caption.

```bash
pip install matplotlib numpy
python analysis/make_figures.py
```

---

## Citation

A preprint is on arXiv (eess.AS, cross-listed cs.CR and cs.CL). See
`CITATION.cff`, or:

```bibtex
@misc{qurishi2026onerun,
  title  = {One Training Run Is Not Enough: Measuring How Training-Data Volume
            Affects Adversarial Robustness in Low-Resource Speech Recognition},
  author = {Qurishi, Mahdi Hasan and Abedin, Umme Zoayria and
            Hossain, A.K.M. Fakhrul},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

Code in this repository is MIT-licensed (see `LICENSE`). The underlying audio is
the Bengali.AI Speech corpus and remains subject to that competition's terms;
model checkpoints and result files on Kaggle carry the licence stated on each
dataset page.

## Scope

This is defensive security research. It measures how the amount of training data
affects a model's vulnerability, and reports a methodological flaw that makes
such measurements unreliable when each condition is trained once. No attack code
is provided for deployed systems, and the over-the-air results we attempted
failed their own controls and are reported as a negative result.
