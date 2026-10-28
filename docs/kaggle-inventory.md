# Kaggle inventory

Model checkpoints are 5–8.8 GB each and are not in this repository. Everything
below was confirmed publicly reachable on 29 October 2026. Result files are
already committed under [`../data/`](../data/); this page is for the
checkpoints and for anyone who wants the original dataset pages.

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

Ten training runs. The 100-hour rung is the one that was not replicated.

## Splits

`mahdihasanqurishi/bn-asr-splits` — the frozen train, development and test
manifests, plus the vocabulary. Every notebook reads these rather than
regenerating them.

## Result datasets

The files are committed in [`../data/results/`](../data/results/), so these are
only needed if you want the original pages.

```
zoayriaabedin/eps50-rung1-final-t5-n100
zoayriaabedin/eps50-rung5-final-t5-n100
zoayriaabedin/eps50-rung25-final-t5-n100
zoayriaabedin/eps50-rung100-milestone-1000-t5-n100
zoayriaabedin/eps50-rung100-milestone-2000-t5-n100
zoayriaabedin/eps50-rung100-milestone-3500-t5-n100
mahdihasanqurishi/eps50-queue-n100
mahdihasanqurishi/clean-wer-replicates-a
mahdihasanqurishi/clean-wer-replicates-b
```

The four `NB-16-eps50-replicates` runs were saved as notebook outputs rather
than datasets; reach them from the notebook pages.

## Notebooks

```
mahdihasanqurishi/taskbsecurity2
zoayriaabedin/taskbsecurity2
zoayriaabedin/taskasecurity
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
