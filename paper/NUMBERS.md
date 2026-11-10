# Traceability — every number in `thesis.tex` → its source

Rule: nothing enters the paper that is not in this table. When a new experiment
finishes, add its rows here first, then write the prose.

## Section V — Experimental Setup

| Paper value | Source |
|---|---|
| 953,848 utterances / 1179.9 h | RUNLOG, NB-00 v2 row |
| normalisation changed 39.96%, 100% NFC | RUNLOG, NB-00 v2 row |
| 25-utterance attack set, 15 containing না | 02_KAGGLE_SETUP_GUIDE.md "Gate before moving on" |
| effective batch 32 (8 × 4), MAX_STEPS 6000 | 01_FULL_THESIS_PLAN.pdf §7.2; NB-01 |
| fp32 seed spread 8.9%, AMP vs fp32 0.0%, 2.40× | RUNLOG, NB-04b row |
| torch 2.10.0+cu128, transformers 4.44.2, numpy 2.0.2, accelerate 1.13.0 | RUNLOG, Environment table |
| pos_conv_embed repair | RUNLOG, Known limitations #2 |

## Section VI-A — Table I (the ladder)

| Rung | Utts | Hours | test WER | attack WER | FLEURS WER |
|---|---|---|---|---|---|
| 1 h | 809 | 0.98 | 0.6705 | 0.6471 | 0.6778 |
| 5 h | 4,086 | 4.90 | 0.4739 | 0.4755 | 0.4992 |
| 25 h | 20,363 | 24.60 | 0.3484 | 0.3235 | 0.3865 |
| 100 h | 80,773 | 98.02 | 0.3174 | 0.2500 | 0.3530 |

Source: RUNLOG "The ladder" + "NB-02 results" tables.

**⚠ OPEN — resolve before submission.** The 5 h rung is 4,086 utts in the ladder
table but **4,051** in the "Training compute is ruled out" section, which says it
read NB-01's provenance. Table III in the paper uses 4,051. Pick one from the
provenance file. Hours also differ: ~5.0 vs 4.90.

## Section VI-B — Table II (ε50)

| Condition | attack WER | ASR | cens | ε50 ok | ε50 imp |
|---|---|---|---|---|---|
| 1 h final | 0.6471 | 80% | 5 | 6.369e-4 | 7.843e-4 |
| 5 h final | 0.4755 | 92% | 2 | 4.686e-4 | 5.146e-4 |
| 25 h final | 0.3235 | 68% | 8 | 6.815e-4 | 9.911e-4 |
| 100 h final | 0.2500 | 48% | 13 | 8.363e-4 | 2.000e-3 |
| 100 h ms-1000 | 0.6667 | 40% | 15 | 8.877e-4 | 2.000e-3 |
| 100 h ms-2000 | 0.4804 | 52% | — | — | — |
| 100 h ms-3500 | 0.3480 | 60% | 10 | 9.911e-4 | — |

Source: RUNLOG "Ladder complete for T5 finals" + "ISO-WER PAIR 2" + "RQ1 COMPLETE".

**⚠ OPEN.** ms-2000 and ms-3500 censoring / imputed medians are not in RUNLOG.
Pull from the published `eps50-*` datasets when building NB-09.

**⚠ CHECK.** 25 h final imputed median (9.911e-4) and ms-3500 successes-only
median (9.911e-4) are identical. Plausible if both land on the same bisection
grid point, but verify it is not a transcription slip.

Step-to-step stats: 5→25 h ratio 1.93×, z=+2.36, p=0.0092; 25→100 h ratio 2.02×,
z=+1.75, p=0.0404. Non-monotonic 1→5 h: 0.66×, p=0.0245.
Spearman(attack WER, imputed ε50) = +0.10 over 5 conditions.

## Section VI-C — Table III (iso-WER pairs)

| Pair | ASR | Fisher p | ratio [CI] | MW p | exposure |
|---|---|---|---|---|---|
| 1 | 80→40% | 0.0043 | 2.55× [1.10, 3.54] | 0.0024 | 40× |
| 2 | 92→52% | 0.0018 | 3.22× [1.75, 4.91] | <0.0001 | 16× |
| 3 | 68→60% | 0.3844 | 1.21× [0.83, 2.80] | 0.0887 | 4× |

Source: RUNLOG "RQ1 COMPLETE — all three iso-WER pairs".

**⚠ OPEN — CI conflict.** Pair 1's CI appears as **[1.10, 3.89]** in the
"ISO-WER TEST PASSED" section and **[1.10, 3.54]** in the final table. A third
value, [1.00, 3.54], belongs to a *different* comparison (1 h vs 100 h final).
Recompute once in NB-09 and use that everywhere.

## Section VI-D — Table IV (compute control)

| Condition | steps | samples | epochs | distinct clips |
|---|---|---|---|---|
| 1 h final | 2500 | 80,000 | 98.9 | 809 |
| 5 h final | 5000 | 160,000 | 39.5 | 4,051 |
| 100 h ms-1000 | 1000 | 32,000 | 0.4 | 32,000 |
| 100 h ms-2000 | 2000 | 64,000 | 0.8 | 64,000 |

Source: RUNLOG "Training compute is ruled out in BOTH pairs".

## Section VI — ANCOVA (NB-09 v2, 2026-09-26)

| term | beta | 95% CI | p | p_holm |
|---|---|---|---|---|
| log10(hours) | +0.2015 | [+0.063, +0.340] | 0.0121 | 0.0482 |
| clean_WER | +0.4632 | [+0.136, +0.790] | 0.0134 | — |
| constant | −3.4700 | [−3.817, −3.123] | <0.0001 | — |

175 utterance rows, 7 conditions, cluster-robust SE, t with df=6.
**Use the t-based figures.** NB-09 v1 used a normal approximation and reported p=0.00038;
that run is superseded. Source: `ancova.csv` in `thesis-figures`.

## Section VI — log-rank

| pair | chi2 | p | ratio |
|---|---|---|---|
| 1h vs ms-1000 | 9.20 | 0.0024 | 2.55× |
| 5h vs ms-2000 | 18.80 | <0.0001 | 3.22× |
| 25h vs ms-3500 | 1.20 | 0.2726 | 1.21× |

Source: `logrank.csv`. Pair-1 CI **[1.10, 3.54]** around the ratio **of medians** 2.55×.
The 2.02× figure is a median of bootstrapped ratios — a different estimator, do not quote it.

## Section VII — RQ2, paired T1 vs T5

| quantity | T1 | T5 |
|---|---|---|
| ASR@2e-3 | 100% (38/38) | 36.8% (14/38) |
| censored | 0 | 24 |
| median eps50 successes | 1.332e-4 | 6.978e-4 |
| median eps50 imputed | 1.332e-4 | 2.000e-3 |
| median SNR | 56.1 dB | 40.6 dB |
| mean target tokens | 46.5 | 46.6 |
| eps50 per token | 2.988e-6 | 2.013e-5 |

Paired ratio **9.34× [7.38, 13.57]** · Wilcoxon p=3.865e-08 · Fisher p=5.133e-10 ·
per-token 6.74×. Source: `rq2.csv`.

Corpus: 12,278 pool, 2,687 substring, 277 token, 202 sentence-final, 90% FP,
67 judged, 38 accepted (57%). Source: NB-03 `_provenance.json`.

## Section VIII — RQ3

| model | exposure | alpha* | clean WER | attack |
|---|---|---|---|---|
| 1h-final | 0.98 h | 3.0 | 0.6324 → 0.4412 | 56% → 4% |
| 100h-ms1000 | 38.8 h | 1.0 | 0.6667 → 0.3922 | 20% → 0% |
| 100h-final | 98.02 h | 0.5 | 0.2451 → 0.1716 | 40% → 0% |

Spearman rho = −1.0 (n=3, directional). LM held-out entropy 1.511 nats/char.
Source: `rq3.csv`, `lm_sweep_*.csv`.

## Holm correction

13 tests, **10 survive** at alpha=0.05. The three failures are all pair 3, already n.s.
ANCOVA survives at p_holm=0.0482 — by 0.0018. Source: `tests.csv`.

## Convergence — limitation

Per-condition shift on doubling CONFIRM_STEPS: 8.9, 45.6, 4.8, 47.1, 17.8, 17.1, 26.4 (T5
ladder), 4.2 (T1), 0.0 (T5 paired). **Max 47.1%.** Report eps50 as budget-relative.

## Not in the paper, deliberately

| Excluded | Why |
|---|---|
| All NB-06 α-sweep numbers | Run invalidated by the decoder defect. Nothing from it is reportable |
| Stouffer combined z = 4.98 | Pairs share utterances and one training run; overstates evidence. Mentioned as a caveat only |
| "100× data" phrasing | Superseded by the exposure correction: ~40× and ~16× |
| 1 h run-2 figures (ASR 60%, 10 censored) | Run 3 supersedes; run 2 may be cited as a robustness note |
| NB-09 v1 ANCOVA p=0.00038 | Normal approximation, anti-conservative at 7 clusters. v2's t-based p=0.0121 supersedes |
| "LM helps the low-exposure model more" | Three measures disagree; counts 7/25 vs 2/25 are inside sampling variation. Reported as unresolved |

## 2026-09-27 — pair-1 procedure control (NB-05c) and the revised RQ1 framing

| Figure in paper | Value | Source |
|---|---|---|
| ANCOVA slope (now the primary estimate) | β=+0.2015, CI [+0.063,+0.340], p=0.012, 7 clusters | NB-09 `ancova.csv`; RUNLOG 2026-09-23 |
| 10x data -> eps50 multiplier | 1.59x | 10^0.2015 |
| Pair 1, **single-pass** probe both sides | 64.0% (16/25) -> 44.0% (11/25) = **−20 pp**, Fisher p=0.2563, dWER 0.0098 | `eps50c-rung1-final-fixedeps-t5-origckpt` + `eps50-rung100-ms1000-fixedeps-t5` |
| Pair 1, **bisection** both sides (as published) | 80% -> 40% = −40 pp, MW p=0.0024, eps50 ratio 2.55x | RUNLOG 2026-09-23 |
| Procedure effect, model held fixed (Tab. procedure A->B) | −16 pp, Fisher p=0.3451 | A = RUNLOG 2026-09-23; B = NB-05c |
| **Checkpoint effect, data+procedure fixed (B->C)** | **−28 pp, Fisher p=0.0887** | B = NB-05c; C = `eps50b-rung1-final-fixedeps-t5` |
| Total A->C | −44 pp, Fisher p=0.0037 | additive: −16 + −28 |
| Tab. procedure row A | step 2500, bisection, WER 0.6471, ASR 80% | RUNLOG 2026-09-23 |
| Tab. procedure row B | step 2500, single pass, WER 0.6569, ASR 64% | NB-05c provenance |
| Tab. procedure row C | step 5500, single pass, WER 0.6716, ASR 36% | NB-05b run 4 provenance |
| 100h bisection-minus-single-pass spread | −4, −8, +12, −4 pp | RUNLOG 2026-09-26 cross-check |
| 1h bisection-minus-single-pass | +16 pp | A vs B above |
| Noise floor, identical inputs re-run | 8 pp ASR (cuDNN autotune); 8.9% eps50 across seeds | RUNLOG; NB-04b run 2 |
| Checkpoint pair dev WER gap | 0.6762 vs 0.6810 = 0.5 pp | NB-01 provenance, both 1h runs |
| Checkpoint pair attack WER gap | 0.6471 vs 0.6716 = 1.5 pp | rows A and C |

**Framing decided 2026-09-27:** ANCOVA is the primary estimate; pairwise figures are a
sign-agreeing robustness check with an unestablished magnitude, reported as a 20--40 pp bracket.
Checkpoint variance is reported as the first limitation, not as a contribution.

## RQ4 — deployment-pipeline survival (NB-07 x4, aggregated by NB-09)

| Figure in paper | Value | Source |
|---|---|---|
| overall survival | 12/576 = **2.1%** | `rq4_by_transform.csv` |
| mulaw survival | 9/72 = 12.5% | `rq4_by_transform.csv` |
| aac_64 survival | 3/72 = 4.2% | `rq4_by_transform.csv` |
| six other transforms | 0/72 each | `rq4_by_transform.csv` |
| reverted total | 6/576 | `rq4_by_transform.csv` |
| per-rung survival | 2.63 / 3.75 / 0.66 / 0.89 % | `rq4_by_rung.csv` |
| raw ASR per rung | 76 / 80 / 76 / 56 % | NB-07 logs, `pipeline-survival-rung{1,5,25,100}` |
| low vs high exposure | 3.21% vs 0.76%, ratio 4.2x | pooled from `rq4_by_rung.csv`; **no p-value reported, units not independent** |
| mulaw-only test | 17.9% vs 6.1%, ratio 3.0, p=0.17 | Fisher, one observation per verified attack |
| Spearman (pre-specified) | rho = −0.60, p = 0.40, n=4 | `tests.csv` |
| survival vs waveform damage | r = −0.63 | 32 transform-rung cells |
| transform damage values | 0.007–0.119 | NB-07 self-test on a real clip |
| AAC encoder delay | 3424 samples = 214 ms; rel. err 0.775 -> 0.0008 after alignment | measured 2026-09-27 |

**Figures:** `fig4_trajectory.png` (early-checkpoint control, curves in training order),
`km_curves.png` (iso-WER survival curves). Both first produced 2026-09-27.

---

# FINAL — all numbers verified against the NB-16/NB-17 logs, 2026-10-26

Data collection closed. 10 fine-tuning runs, 13 attacked checkpoints, 1300
utterance-level rows. NB-17's reproduction check returned +0.1517 and +0.1007 to
four decimals, so the analysis pipeline is verified against the previously
published figures.

## The pooled fit (`tab:ancova`, primary)

| quantity | value | source |
|---|---|---|
| β₁ | +0.0986 | NB-17, all 13 runs clustered by run |
| β₁ 95% CI | [+0.0198, +0.1775] | t, df = G−1 = 12 |
| per decade | 1.255× [1.047, 1.505] | 10^β₁ |
| β₂ (clean WER) | +0.6199 [+0.1965, +1.0433] | same fit |
| G / N | 13 / 1300 | — |
| df = 5 (exposure levels − 1) | [+0.0055, +0.1917] | still excludes zero |
| df = 3 (replicate groups − 1) | [−0.0166, +0.2138] | includes zero; reported |

## Between-run σ (`tab:sigma`)

| group | runs | ASR range | sd |
|---|---:|---|---:|
| 1 h, tf 4.44.2 | 2 | 41.0–75.0% | 24.0 pp |
| 1 h, tf 5.16.1 | 2 | 39.0–57.0% | 12.7 pp |
| 5 h, tf 5.16.1 | 2 | 59.0–96.0% | 26.2 pp |
| 25 h, tf 5.16.1 | 2 | 57.0–78.0% | 14.8 pp |

Raw pooled sd 20.28, df 4. c₄(4) = 0.9400 → **σ = 21.57 pp**, χ² CI
**[12.15, 58.27]**. Across-environment pooling: 16.37 pp at df 7 (smaller, so
the library change is not inflating it).

## The pair-power table (`tab:pairpower`)

σ√2 = **30.50 pp**. Pair effects 38.0 / 33.0 / 13.0 pp = **1.25 / 1.08 / 0.43**
standard errors. None significant.

## The new headline pair (`tab:replication`) — 5 h, seeds 1338 vs 1339

| | clean WER n=100 | n=25 | dev | ASR | censored |
|---|---:|---:|---:|---:|---:|
| s1338 | 0.4152 | 0.4461 | 0.4682 | 96.0% | 4 |
| s1339 | 0.4187 | 0.4755 | 0.4858 | 59.0% | 41 |
| gap | **0.0035** | 0.0294 | 0.0176 | **37.0 pp** | |

McNemar 39/2 discordant, **p = 7.84e-10**. Ratios to the deliberate pair gaps
(0.0588 / 0.0519 / 0.0334): **17.0× / 15.0× / 9.7×** tighter.

**CORRECTION APPLIED.** A draft of `sec:replication` claimed these two runs were
"identical to four decimal places" on the 25-utterance subset. They are not —
they differ by 0.0294 there and by 0.0176 on dev. The text now states all three
measurement sets and claims only that no accuracy measure available would
separate them as different conditions (0.0294 is still tighter than pair 3's
0.0334).

## Within-condition McNemar count

12 possible pairs across the three replicated rungs; **10 significant** at
α = 0.05. Significant gaps: −37, −36, −34, −21, −20, −18, −18, +17, +16, +12 pp.
Non-significant: −9 (25 h, p = 0.122) and −2 (1 h, p = 0.851).

## Power (`tab:power`)

| target | runs/condition | CI |
|---|---:|---|
| 38 pp | 6 | 2–37 |
| 33 pp | 7 | 2–49 |
| 13 pp | 44 | 14–315 |

Smallest detectable effect: 60.4 / 49.4 / 38.2 / 27.0 pp at 2 / 3 / 5 / 10 runs.
Full design at the 13 pp requirement: 308 runs, ~1078 GPU-h training +
~1417 attacks, ~83 weeks of quota.

## The ten runs (`tab:runs`)

dev WER spread within a rung: 0.0352 (1 h), 0.0176 (5 h), 0.0212 (25 h) — an
order of magnitude smaller than the attack-success spread, which is the point.

## Unchanged from the previous version

RQ2 (negation asymmetry 9.34× at 100 h, 4.33× at 1 h), RQ3 (α\* ordering), RQ4
(2.1% codec survival), RQ5 (LM fusion 56% → 4%), the over-the-air failure
analysis, and the three iso-WER pair effects themselves (−38 / −33 / −13 pp with
their McNemar p-values). Only their interpretation changed.

---

# CLUSTERING CORRECTED, 2026-10-26 — supersedes the pooled-fit figures above

The primary fit now clusters on the **training run** (G = 11), not the condition label
(G = 13). The three 100-hour milestones are steps 1000, 2000 and 3500 of one trajectory and
are not independent clusters.

| quantity | was (G=13) | **now (G=11)** |
|---|---|---|
| β₁ | +0.0986 | **+0.0986** (unchanged — OLS) |
| clustered SE | 0.0362 | **0.0307** |
| 95% CI | [+0.0198, +0.1775] | **[+0.0302, +0.1670]** |
| t df | 12 | **10** |
| per decade | 1.255× [1.047, 1.505] | **1.255× [1.072, 1.469]** |
| β₂ (clean WER) | +0.6199 [+0.1965, +1.0433] | **+0.6199 [+0.1886, +1.0512]** |

**The correction strengthened the result.** The label-level fit had a *larger* clustered
standard error, so it was conservative; merging the three milestone clusters tightened the
interval rather than widening it. That was luck, not a reason to have left it.

## df sensitivity (recomputed inside NB-17, not by hand)

| df | rationale | t | 95% CI | |
|---:|---|---:|---|---|
| 10 | G−1, training runs = 11 (**primary**) | 2.228 | [+0.0302, +0.1670] | excludes zero |
| 5 | distinct exposure levels (6) − 1 | 2.571 | [+0.0197, +0.1776] | excludes zero |
| 2 | replicate groups (3) − 1 | 4.303 | [−0.0335, +0.2307] | includes zero |

Reported in the manuscript as a table. Two of three exclude zero; the df = 2 row is reported
rather than hidden.

## Unchanged by the correction

σ = 21.57 pp [12.15, 58.27] at df 4; σ√2 = 30.50 pp; the pair-power table (1.25 / 1.08 / 0.43);
the 5 h headline pair; the 10-of-12 McNemar count; the power table; every RQ2–RQ5 figure.
σ never used the ANCOVA clustering.

## The 100-hour replicate (NB-19 / NB-20 / NB-21 session B)

| Paper value | Source |
|---|---|
| 12 fine-tuning runs, 14 attacked checkpoints, 1,400 measurements | count: 11 runs + `ckpt-xlsr-100h-s1338`; 13 conditions + `rung100-s1338-ms3500` |
| 100 h seed 1338, dev WER 0.3184, CER 0.0693, 6000 steps | RUNLOG, NB-19 entry 2026-11-03 |
| 100 h seed 1337, dev WER 0.3238, CER 0.0728 | RUNLOG, NB-01 RUNG=100 row |
| clean WER on attack set, ms-3500: 0.3449 (s1337) vs 0.3495 (s1338), gap 0.46 pp | RUNLOG, NB-20 entry 2026-11-03; `clean-wer-replicates-c` |
| ASR@2e-3, ms-3500: 53.0% (s1337) vs 56.0% (s1338), gap 3.0 pp | published: `eps50_rung100_milestone-3500_T5_n100.csv`; replicate: `eps50_rung100-s1338-ms3500_T5_n100.csv` |
| sigma at the 100 h rung: 2.1 raw, 2.7 pp after c4(1), df 1 | sqrt(3.0^2 / 2) = 2.121; 2.121 / 0.7979 = 2.659 |
| ratio to the pooled low-rung sigma: 8.1x | 21.574 / 2.659 |
| reproduction check: 0.6943 / 0.4752 / 0.3449 reproduced to 4 dp under transformers 5.20.2 | RUNLOG, NB-20 agreement check |
| pair 3 recomputed: 13.0 pp published arm, 10.0 pp replicate arm, 11.5 pp arm mean | 66.0 - 53.0; 66.0 - 56.0; 66.0 - 54.5 |
| transformers 5.20.2, accelerate 1.15.0, numpy 2.1.3, torch 2.11.0+cu130 | NB-19 and NB-21 session B provenance dumps |

**Still open.** `rung100-s1338-ms1000` and `rung100-s1338-ms2000` are in
measurement (NB-21 session A, plus a session C to finish ms2000). When they land,
the 100-hour rung has three pairs instead of one, `sec:rungsigma` can quote a
pooled sigma on three degrees of freedom instead of a bound on one, and the
leave-one-cluster-out diagnostic in the ANCOVA gets a full second cluster at
38.80, 77.70 and 98.02 hours. Do not promote the rung-dependence claim to the
abstract or the contributions list until then.

### Second 100-hour pair (NB-21 session A, ms-1000)

| Paper value | Source |
|---|---|
| 15 attacked checkpoints, 1,500 measurements | 13 published + `rung100-s1338-ms3500` + `rung100-s1338-ms1000` |
| superseded by 16 / 1,600 once ms-2000 landed | see the ms-2000 block below |
| ASR@2e-3, ms-1000: 37.0% (s1337) vs 34.0% (s1338), gap 3.0 pp | published: `eps50_rung100_milestone-1000_T5_n100.csv`; replicate: `eps50_rung100-s1338-ms1000_T5_n100.csv` |
| clean WER, ms-1000: 0.6943 vs 0.6955, gap 0.12 pp | RUNLOG, NB-20 entry |
| sigma at the 100 h rung, two pairs: 2.1 raw, 2.4 pp after c4(2), df 2 | sqrt(mean(3.0^2/2, 3.0^2/2)) = 2.121; 2.121 / 0.8862 = 2.394 |
| ratio to the pooled low-rung sigma: 9.0x | 21.574 / 2.394 |
| trajectory shape reproduced: 37.0 to 53.0 (s1337), 34.0 to 56.0 (s1338) | the two ASR series above |

**CLOSED 2026-11-04.** `rung100-s1338-ms2000` only. Session A measured about 64 of 100
rows before its 11-hour wall; session C resumed it from session A's published
partial and completed all 100. See the next block.

### Third 100-hour pair (NB-21 session C, ms-2000) -- rung closed

| Paper value | Source |
|---|---|
| 16 attacked checkpoints, 1,600 measurements | 13 published + ms-3500 + ms-1000 + ms-2000 |
| ASR@2e-3, ms-2000: 46.0% (s1337) vs 48.0% (s1338), gap 2.0 pp | published: `eps50_rung100_milestone-2000_T5_n100.csv`; replicate: `eps50_rung100-s1338-ms2000_T5_n100.csv` (80 rows session A + 20 rows session C) |
| clean WER, ms-2000: 0.4752 vs 0.4960, gap 2.08 pp | RUNLOG line 7443, NB-20 entry |
| exposure at step 2,000: 77.70 h | exposure-hours table |
| sigma at the 100 h rung, three pairs: 1.9 raw, **2.1 pp** after c4(3), df 3 | sqrt(mean(3.0^2/2, 3.0^2/2, 2.0^2/2)) = 1.9149; 1.9149 / 0.92132 = 2.0784 |
| ratio to the pooled low-rung sigma: **10.4x** | 21.574 / 2.0784 = 10.38 |
| trajectory shape reproduced: 37.0 / 46.0 / 53.0 (s1337) vs 34.0 / 48.0 / 56.0 (s1338) | the three ASR pairs above |
| gap sign is not constant: -3.0, +2.0, +3.0 (s1338 minus s1337) | 34-37, 48-46, 56-53 |
| anti-confound: largest clean-WER gap (2.08 pp, ms-2000) carries the smallest ASR gap (2.0 pp) | the WER and ASR rows above |
| session C wall clock 1.62 h; resume read 80 rows from the session A dataset | NB-21 session C log, 5866.8s |

**Gate released.** The rung-dependence claim is now in the preprint abstract, the
IJST abstract and the IJST contributions list, quoting 2.1 pp on df 3 and a ~10x
ratio. All three exposure levels at the 100-hour rung are replicated, so the
leave-one-cluster-out diagnostic has a full second cluster at 38.80, 77.70 and
98.02 hours.

**Deliberately not done.** The three s1338 conditions are NOT added to
`analysis/reanalysis.py` `COND`, and the pooled ANCOVA is unchanged: it still
reports beta_1 = +0.0986 from eleven runs, 13 conditions, 1,300 observations.
The papers now state that exclusion explicitly where `tab:runs` is introduced.
Adding them would change the headline slope and is a separate decision.

**Still open.** NB-17's replicate regex
`eps50_rung(\d+)-s(\d+)_{EDIT_TYPE}_n{N_EVAL}\.csv` does not match
`eps50_rung100-s1338-ms{STEP}_T5_n100.csv`, so NB-17 silently omits all three
new conditions. Same bug class as the NB-06 delimiter defect. Widen the regex
before rerunning NB-17.

---

## PRE-REGISTERED WORDING RULE for claim B — written 2026-11-04 15:49:47 +06, BEFORE the 12-cluster fit was run

The three `eps50_rung100-s1338-ms*` CSVs were staged into `data/results/` but
`analysis/reanalysis.py` had **not** yet been run with
`INCLUDE_100H_REPLICATE=1` when this was written. This rule fixes how claim B
(the data-volume slope beta_1) is worded, so the result cannot choose its own
description.

1. Report the 11-cluster fit (published, 2026-10-29) **and** the 12-cluster fit
   (with the 100-hour replicate) side by side, each with: CR1 interval, wild
   cluster bootstrap p, run-level fit, leave-one-cluster-out.
2. The wording is decided by the **12-cluster wild cluster bootstrap p**, and by
   nothing else:
   * **p < 0.05:** "a positive effect that reached significance only once the
     100-hour replicate was added; suggestive, not established."
   * **p >= 0.05:** "a positive point estimate that cannot be distinguished from
     zero."
3. In neither case is claim B called established, a finding, or a result in the
   abstract or the contributions list. One added training run moving the
   evidence is itself the reason.
4. The headline fit is never chosen by which version has the smaller p.
5. Claim A (run-to-run variance in measured attack success, and its dependence
   on rung) is unaffected by this rule; it does not use the regression.

### Claim B, twelve-cluster fit — RESULT (run after the rule above)

Source: `analysis/reanalysis-output-with-replicate.txt` (INCLUDE_100H_REPLICATE=1).
Published fit reproduced unchanged in `analysis/reanalysis-output-published.txt`.
Inputs: 15 CSVs from the GitHub repo `data/results/` + the three NB-21 CSVs,
each verified at 100 rows, matching ASR, correct s1338 checkpoint path, and the
same 100 utterances as the ladder.

| Paper value | 11 runs | 12 runs |
|---|---|---|
| n / G | 1300 / 11 | 1600 / 12 |
| beta1 per decade | +0.0986 | **+0.0888** |
| per decade multiplier | 1.26x | **1.23x** [1.06, 1.42] |
| CR1 95% CI | [+0.030, +0.167] | [+0.024, +0.154] |
| two-way (run, utterance) | [+0.036, +0.162] | [+0.026, +0.152] |
| condition means, clustered on run | [+0.024, +0.174] | [+0.019, +0.158] |
| + library-version term (pre/post 4.44.2) | [+0.020, +0.185] | [+0.025, +0.152] |
| leave-one-run-out | drop 100h-r1 includes 0 | no drop includes 0 |
| **wild cluster bootstrap p** | 0.1878 | **0.0608** |
| df sensitivity, df 5 | [+0.020, +0.178] | [+0.013, +0.164] |
| df sensitivity, df 3 (replicated rungs - 1) | — | [-0.005, +0.182] includes 0 |
| WER coefficient | +0.620 | +0.550 |
| censored | 517/1300 | 679/1600 (42.4%) |
| ceiling 1e-2 slope | +0.183 | +0.170 |
| SNR, successful attacks | n 783, median 41.3 dB | n 921, median 40.8, 5th pct 30.1, min 22.0, none < 20 dB |

**Verdict under the pre-registered rule: p = 0.0608 >= 0.05, so claim B is "a
positive point estimate that cannot be distinguished from zero."** Applied in
`preprint.tex` (abstract, contributions, sec:strict, conclusion).

New facts used in the preprint:

| Paper value | Source |
|---|---|
| re-attack of identical 1 h weights: 41% vs 61%, discordant 13/33, exact McNemar p = 0.0045 | computed from `eps50_rung1-trajfinal` vs `eps50_rung1-traj-SECOND-MEASUREMENT` |
| rung-variance F(4,3) = 112.16, p = 0.00135; exact permutation p = 1/35 = 0.0286 | NB-22 cell, dry-run locally |
| two runs at sigma 2.1 resolve ~6 pp | sqrt(15.70 * 2.1^2 / 2) = 5.9 |

**Corrections made in this round.**
* `reanalysis.py` version term coded `env == "tf5.16.1"`, which put the 5.20.2
  replicate on the OLD-version side. Now `env != "tf4.44.2"`. 11-run numbers
  unchanged (no 5.20.2 rows there).
* `reanalysis.py` printed hardcoded narrative ("ONE cluster", "One cluster
  decides it", "13 condition means") that became false with 12 clusters. Now
  computed from the data. All previously published numbers verified identical.
* The section-4 "run-level" fit treated milestone means as independent; a
  version clustered on training run is now printed beside it.
* A second 100-hour run does **not** break the exposure/training-step
  collinearity (both runs checkpointed at the same steps). It adds an
  independent cluster. Fixed in the script, NB-22 and the preprint.
* Preprint SNR line called the minimum-SNR case "the quietest"; it is the
  loudest perturbation.
* Preprint said "ten runs" where the pre-replicate fit has eleven training runs.

### Tobit, floor, practical size, availability (2026-11-04, text/CPU round)

| Paper value | Source |
|---|---|
| Tobit beta1, 12 runs: +0.156, cluster 95% CI [+0.057, +0.255], 1.43x/decade | `reanalysis-output-with-replicate.txt` section 6b |
| Tobit beta1, 11 runs: +0.171, [+0.065, +0.276] | `reanalysis-output-published.txt` section 6b |
| censoring: 679 right at 2e-3; 4 left below 1.0981e-4 (= 1e-4 * 20^(1/32), lowest bisection step) | section 6b counts |
| Tobit estimator validated: no censoring -> equals OLS (+0.13845 both); 32% censored -> recovers +0.139 vs imputed +0.084 | recovery test on simulated data, this session |
| practical size: 20 * 0.0888 = 1.78 dB per decade; 3.6 dB over 100-fold; median SNR 40.8 dB | eps50 is an amplitude bound, so dB = 20 * log10 |
| 18 per-utterance result files = 16 analysed + 100 h final (80 rows) + 1 h re-run second measurement | repo after the pending push |
| IJST abstract 242 words (limit 250), reframed to match the preprint | `ijst-main.tex`, `ijst-main-anonymous.tex` |

**Availability claim depends on a push.** The preprint now says the repo holds 18
result files and the reanalysis script. True only after `analysis/reanalysis.py`,
both outputs and the three s1338 CSVs are pushed. Verified: the script runs in a
fresh clone with those files added and reproduces the project output exactly.

### Final verification pass, 2026-11-04

Mechanical recheck of every number against the committed CSVs.

**Passed.** All 18 ASR values recomputed from `data/results/` match the papers
exactly (0 mismatches). sigma 21.574 (df 4) and 2.078 (df 3), ratio 10.38x,
sigma*sqrt2 = 30.51, pair effects 1.25/1.08/0.43, 16.375 -> 23.16, 1.78 dB per
decade, 10^0.0888 = 1.227. SNR over the 16 analysis conditions: n = 921,
median 40.8 dB, min 22.0 dB, none below 20 dB. df-sensitivity rows for the
12-run fit verified at 7 decimals. All three documents agree on every headline
number. No Whisper mention, no Claude/Anthropic attribution anywhere.

**Two errors found and fixed** in `tab:strict`, both in the historical 11-run
column, both caused by rounding an already-rounded 4-decimal printout:

| row | printed | true value | was | now |
|---|---|---|---|---|
| two-way, run and utterance | +0.1615 | +0.1614581 | +0.162 | **+0.161** |
| library-version term | +0.0195 | +0.0194738 | +0.020 | **+0.019** |

Neither changes any verdict. Guard for next time: take table values from a
high-precision run, not from the 4-decimal console output.

**Not verifiable locally.** The negation numbers (9.34x, 4.33x, 38 utterances),
the LM sweep (56% -> 4%) and pipeline survival (2.1%) have no per-utterance
files in `data/results/`, which holds only the T5 n=100 conditions. Their
notebooks are public; their result files are not.
