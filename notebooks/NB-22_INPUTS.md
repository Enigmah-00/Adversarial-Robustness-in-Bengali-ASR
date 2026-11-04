# NB-22 inputs

Accelerator **None**. Internet **OFF**. 16 inputs, 17 CSVs.

Owners and slugs below are taken from RUNLOG.md. Where RUNLOG does not record
one, it says so. A wrong or missing attachment makes NB-22 stop and name the
file it could not find; it never fits on fewer conditions.

## Code (1)

| # | input | type | carries |
|---|---|---|---|
| 1 | `<you>/thesis-analysis-code` | **new** dataset | `reanalysis.py` (the patched version, 582 lines, defines `REPLICATE_100H`) |

## Published ladder (6) — all `zoayriaabedin`

| # | input | carries | ASR |
|---|---|---|---|
| 2 | `zoayriaabedin/eps50-rung1-final-t5-n100` | `eps50_rung1_final_T5_n100.csv` | 75% |
| 3 | `zoayriaabedin/eps50-rung5-final-t5-n100` | `eps50_rung5_final_T5_n100.csv` | 79% |
| 4 | `zoayriaabedin/eps50-rung25-final-t5-n100` | `eps50_rung25_final_T5_n100.csv` | 66% |
| 5 | `zoayriaabedin/eps50-rung100-milestone-1000-t5-n100` | `eps50_rung100_milestone-1000_T5_n100.csv` | 37% |
| 6 | `zoayriaabedin/eps50-rung100-milestone-2000-t5-n100` | `eps50_rung100_milestone-2000_T5_n100.csv` | 46% |
| 7 | `zoayriaabedin/eps50-rung100-milestone-3500-t5-n100` | `eps50_rung100_milestone-3500_T5_n100.csv` | 53% |

## The two measurements of the 1-hour trajectory weights (2)

| # | input | carries | ASR |
|---|---|---|---|
| 8 | `eps50-queue-n100` (NB-12 output) — **owner not recorded** | `eps50_rung1-trajfinal_T5_n100.csv` | 41% |
| 9 | `zoayriaabedin/eps50-rung1-traj-final-t5-n100` (Task B run 7) | `eps50_rung1_final_T5_n100.csv` — **same name as #2** | 61% |

**#9 is safe in NB-22 and nowhere else.** NB-22 takes that file only from that
dataset and stages it as `eps50_rung1-traj-SECOND-MEASUREMENT_T5_n100.csv`.
The standing rule is unchanged: never attach #9 to NB-16 or NB-17.

## NB-16 seed replicates (4) — attach as **Notebook Output**

These were never published as datasets. Add Input -> Your Work / Notebooks.

| # | input | carries | ASR |
|---|---|---|---|
| 10 | `zoayriaabedin/nb-16-eps50-replicates-runa` | `rung1-s1338` (100) + `rung1-s1339` (80, partial) | 57% |
| 11 | NB-16 **Run B** — **slug and owner not recorded** | `rung5-s1338`, `rung5-s1339` (inferred: the only replicates A, C and D do not hold) | 96%, 59% |
| 12 | `mahdihasanqurishi/nb-16-eps50-replicates-runc` | `rung25-s1338` (100) + `rung25-s1339` (96, partial) | 78% |
| 13 | NB-16 **Run D** — **slug and owner not recorded** | `rung1-s1339` (100), `rung25-s1339` (100) | 39%, 57% |

Runs A and D both carry `rung1-s1339`; Runs C and D both carry `rung25-s1339`.
That is a resume chain and NB-22 takes the 100-row copy. **Run D is required**:
without it both files stop at 80 and 96 rows and the notebook stops.

## NB-21, the 100-hour replicate (3)

| # | input | carries | ASR |
|---|---|---|---|
| 14 | `mahdihasanqurishi/eps50-s1338-ms1000-ms2000partial-n100` | `rung100-s1338-ms1000` (100) + `ms2000` (80, partial) | 34% |
| 15 | NB-21 **session C** output — dataset if published, else Notebook Output | `rung100-s1338-ms2000` (100) | 48% |
| 16 | NB-21 **session B** output — dataset if published, else Notebook Output | `rung100-s1338-ms3500` (100) | 56% |

#14 is confirmed published: session C's log resumed from it.
#14 and #15 both carry `ms2000`; NB-22 takes #15's 100-row copy.

## Not needed

No checkpoint, no audio, no `bengaliai-speech`, no `bn-asr-splits`, no targets
file, no clean-WER dataset. NB-22 loads no model.

## Shorter route, if the notebook attachments are a nuisance

The 13 ladder files and the re-attack file are already committed to
`github.com/Enigmah-00/Adversarial-Robustness-in-Bengali-ASR` under
`data/results/` (commits `78dfdfc`, `1a3aeac`). Clone it, download only #14 to
#16 from Kaggle into `data/results/`, and run locally — pure standard library,
no numpy, no GPU:

    python3 analysis/reanalysis.py .

Use the patched `analysis/reanalysis.py`, not the committed one.
