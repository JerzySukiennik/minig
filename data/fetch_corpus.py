"""
Download Polish text corpora and flatten them into one plain-text file.

Two sources, deliberately:

  wiki     — Wikipedia PL. Clean, factual, well-formed sentences, but written
             in encyclopedic register ("X jest miastem w województwie Y").
             A model trained on this alone answers like a lexicon entry.
  fineweb  — FineWeb-2 (pol_Latn). Filtered and deduplicated web text: reviews,
             articles, forum prose. Messier, but it is where the model learns
             what ordinary Polish actually sounds like.

We take both. Mixture matters more than raw token count — a bigger pile of
encyclopedia would not teach conversation.

Output format: one document per block, terminated by DOC_SEP on its own line.
A blank line cannot be the separator because articles contain blank lines
between their own paragraphs, and the pretraining packer needs unambiguous
document boundaries to place <|endoftext|>.
"""

import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset

DOC_SEP = "<|doc|>"

# G-Micro trained on the first two. G-Mini needs roughly twice the text (3.7B
# tokens against 2.0B), and taking it all from one web crawl would buy quantity
# without variety, so CulturaX joins them — a separately cleaned multi-source
# corpus with its own filtering, rather than more of the same pages.
SOURCES = {
    "wiki":     dict(path="wikimedia/wikipedia", name="20231101.pl"),
    "fineweb":  dict(path="HuggingFaceFW/fineweb-2", name="pol_Latn"),
    "culturax": dict(path="uonlp/CulturaX", name="pl"),
}


def read_token(env_path: Path = Path(".env")) -> str | None:
    """Read HF_TOKEN from the environment, falling back to .env (gitignored)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def main(source: str, out: Path, max_chars: int | None, min_chars: int):
    token = read_token()
    spec = SOURCES[source]

    # streaming=True: iterate shards as they arrive instead of materialising the
    # whole dataset first. On a slow connection we start writing within seconds,
    # and we can stop at any point with a usable corpus.
    ds = load_dataset(spec["path"], spec["name"], split="train",
                      streaming=True, token=token)

    out.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = chars = 0

    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            # Very short documents are overwhelmingly stubs, navigation chrome
            # and disambiguation pages: high boilerplate, low language signal.
            if len(text) < min_chars:
                skipped += 1
                continue
            f.write(text)
            f.write(f"\n{DOC_SEP}\n")
            kept += 1
            chars += len(text)

            if kept % 20000 == 0:
                print(f"  {kept:>9,} docs  {chars/1e6:>9.1f}M chars"
                      f"  (~{chars/3.5/1e6:.0f}M tokens)", flush=True)
            if max_chars and chars >= max_chars:
                print("  reached --max-chars, stopping", flush=True)
                break

    print(f"\ndone -> {out}")
    print(f"  kept {kept:,} docs, skipped {skipped:,} short ones")
    print(f"  {chars/1e6:.1f}M chars  ~{chars/3.5/1e6:.0f}M tokens (rough)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=sorted(SOURCES))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-chars", type=float, default=None,
                    help="stop once this many characters are written")
    ap.add_argument("--min-chars", type=int, default=500)
    args = ap.parse_args()
    main(args.source, args.out,
         int(args.max_chars) if args.max_chars else None,
         args.min_chars)

    # Leave without running interpreter shutdown.
    #
    # `datasets` streaming keeps HTTP worker threads alive, and on some
    # versions they are still touching the GIL when Python finalises, which
    # aborts the process:
    #     Fatal Python error: PyGILState_Release: thread state must be current
    # The corpus is already written and flushed at this point, so the crash is
    # cosmetic — except that it returns SIGABRT, and any caller checking the
    # exit status treats a completed download as a failure. os._exit skips
    # finalisation entirely and reports success.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
