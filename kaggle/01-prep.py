"""Kaggle kernel: fetch the corpus and pack it into token files.

Runs on CPU — no GPU quota is spent here, which matters because the quota is
shared with Gedit and every hour of it is contested.

The corpus is built **on Kaggle rather than uploaded**, the decision G-Micro
made and proved: 9GB of Polish text over a home connection is a day of
uploading, while Kaggle pulls it from Hugging Face in under an hour. Only code
travels from the laptop, via git clone.

Output: a `minig-data` dataset holding pl_train.bin and pl_val.bin, which
02-train.py attaches.
"""

import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-mini.git"
WORK = "/kaggle/working"

# Enough characters to reach ~3.7B tokens after packing. Polish runs about 3.6
# characters per token with this tokenizer, and the split favours the cleaner
# sources: Wikipedia is the smallest but the best-written, so it is taken whole
# and the web crawls make up the volume.
TARGETS = {"wiki": 3_000_000_000, "fineweb": 7_000_000_000, "culturax": 4_000_000_000}

if os.path.exists(f"{WORK}/g-mini"):
    subprocess.run(["git", "-C", f"{WORK}/g-mini", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-mini"], check=True)
os.chdir(f"{WORK}/g-mini")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "datasets", "tokenizers", "zstandard"], check=True)

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass   # public datasets still work; gated ones will fail loudly below

corpora = []
for source, chars in TARGETS.items():
    out = f"{WORK}/corpus_{source}.txt"
    print(f"\n=== {source}: cel {chars/1e9:.1f} mld znaków ===", flush=True)
    # fetch_corpus.py already retries within a source (dropped HF connections
    # are routine on a multi-gigabyte streaming pull). What is caught here is
    # the case that survives all of those retries — a source that is simply
    # unreachable this session. Losing one of three corpora is a worse
    # training run, not a failed kernel: continuing with wiki + fineweb beats
    # aborting the whole prep and spending the next CPU session re-downloading
    # both of the sources that had already succeeded.
    result = subprocess.run([sys.executable, "data/fetch_corpus.py", source,
                             "--out", out, "--max-chars", str(chars)])
    if result.returncode != 0:
        print(f"  {source} nie udało się nawet po retry — pomijam, "
              f"trening pojedzie na pozostałych źródłach", flush=True)
        continue
    corpora.append(out)

assert corpora, "żadne źródło się nie pobrało — sprawdź połączenie / HF_TOKEN"

print("\n=== pakowanie tokenów ===", flush=True)
subprocess.run([sys.executable, "data/pack_data.py", *corpora,
                "--tokenizer", "data/tokenizer.json",
                "--out-prefix", f"{WORK}/pl"], check=True)

# The corpus text itself is not part of the output — it is tens of gigabytes
# and fully regenerable from this script, while the packed .bin files are what
# training actually reads.
for path in corpora:
    os.remove(path)
print("\ngotowe: pl_train.bin, pl_val.bin w /kaggle/working")
