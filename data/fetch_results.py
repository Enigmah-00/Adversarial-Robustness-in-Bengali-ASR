#!/usr/bin/env python3
"""Download the per-utterance result files from Kaggle into data/results/.

Needs the Kaggle CLI and an API token at ~/.kaggle/kaggle.json:
    pip install kaggle

Only the small result files are fetched. Model checkpoints are several GB each
and are listed in docs/kaggle-inventory.md for anyone who wants them.

The script does not overwrite a file that already exists, and it verifies the
frozen target set by SHA-256 if it finds one.
"""
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

# Datasets confirmed public in the project record. If one 404s, it is private;
# ask the authors rather than assuming the file does not exist.
DATASETS = [
    "zoayriaabedin/eps50-rung1-final-t5-n100",
    "zoayriaabedin/eps50-rung5-final-t5-n100",
    "zoayriaabedin/eps50-rung25-final-t5-n100",
    "zoayriaabedin/eps50-rung100-milestone-1000-t5-n100",
    "zoayriaabedin/eps50-rung100-milestone-2000-t5-n100",
    "zoayriaabedin/eps50-rung100-milestone-3500-t5-n100",
    "zoayriaabedin/eps50-rung1-traj-final-t5-n100",
]

TARGETS_SHA256 = "8711dc56202b33f90e769c9735231b0fa8cf27fcbc20151f5466bcd9cc2b1256"


def main():
    os.makedirs(OUT, exist_ok=True)
    failed = []
    for slug in DATASETS:
        print(f"fetching {slug}")
        r = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", OUT, "--unzip"],
            capture_output=True, text=True)
        if r.returncode != 0:
            failed.append((slug, r.stderr.strip().splitlines()[-1:] or ["unknown"]))

    tgt = os.path.join(OUT, "task_b_targets_n100.csv")
    if os.path.exists(tgt):
        got = hashlib.sha256(open(tgt, "rb").read()).hexdigest()
        ok = got == TARGETS_SHA256
        print(f"\ntarget set SHA-256 {'MATCHES' if ok else 'DOES NOT MATCH'}")
        if not ok:
            print(f"  expected {TARGETS_SHA256}")
            print(f"  got      {got}")
            print("  These are not the targets the published runs used. Stop.")
            sys.exit(1)
    else:
        print("\ntarget set not found; cannot verify the hash")

    csvs = sorted(f for f in os.listdir(OUT) if f.startswith("eps50_"))
    print(f"\n{len(csvs)} result files in {OUT}")
    for f in csvs:
        print("   ", f)
    if failed:
        print("\ncould not fetch:")
        for slug, err in failed:
            print(f"   {slug}: {err[0] if err else ''}")
        print("These may be private. Ask the authors.")


if __name__ == "__main__":
    main()
