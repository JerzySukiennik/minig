"""Kaggle kernel: measure G-Mini's real throughput. Paste THIS into the notebook.

Everything else in kaggle/ follows the same pattern, and the pattern matters:
these files are the only ones meant to be pasted into a Kaggle cell. The
scripts they call — bench/size_probe.py, train/train.py — import from the repo
and use `__file__` to find it, so pasting one of those directly fails with
`NameError: name '__file__' is not defined` before it reaches anything
interesting. That is what happened on 2026-07-27.

Needs a T4 x2 accelerator. Runs in a few minutes and spends no more than that.

What to look for in the output:

  G-Micro (kotwica) should come back near 16,800 tok/s at micro-batch 8. It is
  there to prove the measurement, not the model — if this figure is far off,
  distrust the second line too.

  G-Mini (docelowy) should hold micro-batch 8 with peak memory under ~13GB. If
  it drops to micro-batch 4 the throughput falls by roughly a quarter and 3.7B
  tokens no longer fit in 60 hours; the model needs to lose two layers rather
  than the budget quietly going over.
"""

import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-mini.git"
WORK = "/kaggle/working"

if os.path.exists(f"{WORK}/g-mini"):
    subprocess.run(["git", "-C", f"{WORK}/g-mini", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-mini"], check=True)
os.chdir(f"{WORK}/g-mini")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"], check=True)
subprocess.run([sys.executable, "bench/size_probe.py"], check=True)
