"""Kaggle kernel: pretrain G-Mini, resuming across sessions.

A free session dies after a few hours without warning and the budget here is
about sixty, so this is written to be run over and over: each run picks up the
previous checkpoint, trains until the session ends, and leaves a checkpoint for
the next one. `--resume` is what makes that safe — weights, optimiser moments,
step counter and RNG are all restored together.

**Attach these datasets before running:**
  - `minig-data` — pl_train.bin / pl_val.bin, built by 01-prep.py
  - `minig-ckpt` — the previous session's checkpoint (absent on the first run)
  - `microg-ckpt` — G-Micro's pretrained weights, only needed for the first run

Traps this file exists to avoid, every one of them paid for during G-Micro:

  Mount depth is not fixed. `/kaggle/input/<slug>/` and
  `/kaggle/input/datasets/<owner>/<slug>/` have both been observed in the same
  session. A fixed-depth glob silently finds nothing, and G-Micro once restarted
  from a stale checkpoint because of it — recursive globs everywhere.

  This file and the deployed kernel drift. G-Micro kept a copy of its kernel
  outside the repo and the two diverged silently. Here the kernel *is* this
  file: whatever is in git is what runs.

  Warm starting wants a gentler learning rate. The transplanted blocks already
  sit in a good basin and the from-scratch peak (6e-4) can knock them out of
  it, so the first run passes 3e-4.
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-mini.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run1"

if os.path.exists(f"{WORK}/g-mini"):
    subprocess.run(["git", "-C", f"{WORK}/g-mini", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-mini"], check=True)
os.chdir(f"{WORK}/g-mini")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"], check=True)


def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits[0] if hits else None


data_prefix = find("pl_train.bin")
assert data_prefix, "brak pl_train.bin — podepnij dataset minig-data (01-prep.py)"
data_prefix = data_prefix[: -len("_train.bin")]
print(f"dane: {data_prefix}")

os.makedirs(OUT, exist_ok=True)
resume = find("ckpt.pt")
cmd = [sys.executable, "train/train.py", "--data", data_prefix, "--out", OUT]

if resume:
    # A previous session's checkpoint. Copy it in rather than pointing at the
    # read-only mount, because training writes back to the same filename.
    subprocess.run(["cp", resume, f"{OUT}/ckpt.pt"], check=True)
    print(f"wznawiam z {resume}")
    cmd += ["--resume", "--lr", "3e-4"]
else:
    warm = find("warm_start.pt")
    if warm:
        print(f"pierwszy przebieg, ciepły start z {warm}")
        cmd += ["--warm-start", warm, "--lr", "3e-4"]
    else:
        print("pierwszy przebieg, trening od zera (brak warm_start.pt)")
        cmd += ["--lr", "6e-4"]

subprocess.run(cmd, check=True)
print(f"\ncheckpoint w {OUT} — opublikuj jako minig-ckpt przed następnym przebiegiem")
