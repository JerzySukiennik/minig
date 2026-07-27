"""
Tokenise the corpus into a flat uint16 binary that training can memory-map.

Why a binary and not the text file? Training reads a random 1024-token window
tens of thousands of times per run. Re-tokenising text every time would make
the GPU wait on the CPU. Tokenise once, store token ids, then a batch is just
a slice of an array.

Why uint16? Our vocabulary is 32000, which fits in 16 bits. That halves the
file compared to uint32 — and since training is I/O bound on the shuffle, half
the bytes is close to half the time. Guarded by an assert below, because if
vocab_size ever exceeds 65535 this silently corrupts every token.

Document boundaries become <|endoftext|>. Without it the model never learns
that text ends, and generation rambles past the point where it should stop.
"""

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

DOC_SEP = "<|doc|>"
EOT = "<|endoftext|>"
BATCH_DOCS = 1000        # documents per encode_batch call


def iter_documents(path: Path):
    """Yield documents one at a time without loading the whole corpus in RAM."""
    buf: list[str] = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.rstrip("\n") == DOC_SEP:
                doc = "".join(buf).strip()
                buf.clear()
                if doc:
                    yield doc
            else:
                buf.append(line)
    tail = "".join(buf).strip()
    if tail:
        yield tail


def pack(corpus_paths: list[Path], tokenizer_path: Path, out_prefix: Path,
         val_fraction: float):
    tok = Tokenizer.from_file(str(tokenizer_path))
    assert tok.get_vocab_size() <= 65535, "vocab too large for uint16 storage"
    eot_id = tok.token_to_id(EOT)
    assert eot_id is not None, f"{EOT} missing from tokenizer vocabulary"

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    train_f = open(f"{out_prefix}_train.bin", "wb")
    val_f = open(f"{out_prefix}_val.bin", "wb")

    n_docs = n_train = n_val = 0
    batch: list[str] = []

    def flush(batch):
        nonlocal n_train, n_val, n_docs
        if not batch:
            return
        for i, enc in enumerate(tok.encode_batch(batch)):
            ids = np.fromiter(enc.ids, dtype=np.uint16, count=len(enc.ids))
            ids = np.append(ids, np.uint16(eot_id))
            # Deterministic split by document index: a document is either
            # entirely in train or entirely in val. Splitting mid-document
            # would leak the same sentences into both sides and make the
            # validation loss optimistic.
            if (n_docs + i) % int(1 / val_fraction) == 0:
                ids.tofile(val_f); n_val += len(ids)
            else:
                ids.tofile(train_f); n_train += len(ids)
        n_docs += len(batch)
        batch.clear()

    for path in corpus_paths:
        print(f"packing {path} ...", flush=True)
        for doc in iter_documents(path):
            batch.append(doc)
            if len(batch) >= BATCH_DOCS:
                flush(batch)
                if n_docs % 100_000 == 0:
                    total = n_train + n_val
                    print(f"  {n_docs:>9,} docs  {total/1e6:>8.1f}M tokens", flush=True)
    flush(batch)

    train_f.close(); val_f.close()
    total = n_train + n_val
    print(f"\ndone — {n_docs:,} docs, {total/1e6:.1f}M tokens")
    print(f"  train: {n_train/1e6:8.1f}M -> {out_prefix}_train.bin")
    print(f"  val:   {n_val/1e6:8.1f}M -> {out_prefix}_val.bin")
    return n_train, n_val


def verify(out_prefix: Path, tokenizer_path: Path):
    """Round-trip a slice back through the tokenizer and eyeball it.

    Catches the failure mode this script is most prone to: a packer that runs
    happily to completion and writes plausible-looking garbage.
    """
    tok = Tokenizer.from_file(str(tokenizer_path))
    eot_id = tok.token_to_id(EOT)
    arr = np.memmap(f"{out_prefix}_train.bin", dtype=np.uint16, mode="r")

    print("\n--- verification ---")
    print(f"tokens on disk:  {len(arr):,}")
    print(f"max token id:    {int(arr.max())} (vocab {tok.get_vocab_size()})")
    assert int(arr.max()) < tok.get_vocab_size(), "token id outside vocabulary!"

    n_eot = int((arr[:2_000_000] == eot_id).sum())
    print(f"<|endoftext|> in first 2M tokens: {n_eot:,} "
          f"(~1 per {2_000_000 // max(n_eot, 1):,} tokens)")
    assert n_eot > 0, "no document separators found — boundaries were lost!"

    print("\nsample decode from the middle of the file:")
    mid = len(arr) // 2
    print(" ", tok.decode([int(t) for t in arr[mid:mid + 60]])[:300].replace("\n", " "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    ap.add_argument("--out-prefix", type=Path, default=Path("data/pl"))
    ap.add_argument("--val-fraction", type=float, default=0.005)
    args = ap.parse_args()

    pack(args.corpus, args.tokenizer, args.out_prefix, args.val_fraction)
    verify(args.out_prefix, args.tokenizer)
