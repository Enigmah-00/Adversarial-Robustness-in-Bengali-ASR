# Kaggle inventory

Everything too large for Git. Checkpoints are 5–8.8 GB each; result files are
under 150 KB each.

## Result files (small — fetch these)

Confirmed public in the project record:

| dataset | contents |
|---|---|
| `zoayriaabedin/eps50-rung1-final-t5-n100` | 1 h rung, seed 1337 |
| `zoayriaabedin/eps50-rung5-final-t5-n100` | 5 h rung, seed 1337 |
| `zoayriaabedin/eps50-rung25-final-t5-n100` | 25 h rung, seed 1337 |
| `zoayriaabedin/eps50-rung100-milestone-1000-t5-n100` | 100 h run, step 1000 |
| `zoayriaabedin/eps50-rung100-milestone-2000-t5-n100` | 100 h run, step 2000 |
| `zoayriaabedin/eps50-rung100-milestone-3500-t5-n100` | 100 h run, step 3500 |
| `zoayriaabedin/eps50-rung1-traj-final-t5-n100` | second 1 h training run, measured under the Task B procedure |

Each also contains `task_b_targets_n100.csv`, the frozen target set.

> **Naming collision, important.** Task B named each output after the rung and
> checkpoint, so `eps50-rung1-final-t5-n100` and
> `eps50-rung1-traj-final-t5-n100` **both contain a file called
> `eps50_rung1_final_T5_n100.csv`** — for two *different* models (75% and 61%
> attack success). Any analysis that looks results up by filename will silently
> take whichever mounted first. `NB-17` detects this and refuses to run; if you
> write your own analysis, key on the dataset, not the filename.

## Replicate outputs

The six replicate checkpoints and the `NB-16` result files were produced in the
replication round and are split across two accounts
(`zoayriaabedin` and `mahdihasanqurishi`):

```
ckpt-xlsr-1h-s1338      ckpt-xlsr-1h-s1339
ckpt-xlsr-5h-s1338      ckpt-xlsr-5h-s1339
ckpt-xlsr-25h-s1338     ckpt-xlsr-25h-s1339
clean-wer-replicates-a  clean-wer-replicates-b
```

`NB-16`'s per-condition results were attached as notebook outputs rather than
published as datasets, so they are reached through the notebook pages rather
than the dataset pages.

## Original ladder checkpoints

```
ckpt-xlsr-1h     ckpt-xlsr-1h-traj
ckpt-xlsr-5h     ckpt-xlsr-25h      ckpt-xlsr-100h
bn-asr-splits    (the frozen train/dev/test manifests)
```

## Audio

The audio itself is the **Bengali.AI Speech Recognition** competition dataset on
Kaggle. It is not redistributed here or in any of the datasets above; accept the
competition rules to access it.

## Notebooks

The committed Task B attack notebook is public at
`kaggle.com/code/mahdihasanqurishi/taskbsecurity2`.

---

**Before citing this inventory in a paper, check each link resolves.** Some
datasets in the replication round may still be private. Set visibility in the
dataset's Settings page — do **not** publish a new version to make it public,
because publishing replaces the dataset contents and deletes any file a partial
run did not regenerate.
