"""Train G-Mini's Polish BPE tokenizer — 48k vocabulary, digits split apart.

Two changes from G-Micro's tokenizer, both measured rather than assumed.

**Digits are split into individual characters.** G-Micro's vocabulary contains
241 multi-digit tokens, including whole years like 1998, 2011 and 2012 as
single symbols. Measured 2026-07-27, that is exactly where its grounding
fails: handed a text saying a castle was built in 1417 it answered 1418, and
8 431 came back as 8 321. "1417" tokenises as ['14','17'], so the model copies
the first token and guesses the second, while 1963 and 1964 are single
neighbouring symbols with nearly identical embeddings. Splitting every digit
turns copying a number into four independent one-character copies, which is
the easy operation the model already performs well on names. Llama and Mistral
do the same thing for the same reason.

**Vocabulary grows 32k -> 48k.** Polish inflection produces many surface forms
of the same stem, and a larger vocabulary spends fewer tokens on the same text
— which at a fixed token budget means the model reads more Polish. It also
offsets the extra tokens that digit splitting costs. Embeddings are tied to
the output head, so the whole increase is 16000 x 768 = 12.3M parameters once,
not twice.
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
from tokenizers.models import BPE

# Reserved control tokens. The chat format is built from these, so they have to
# exist from the first step of pretraining — bolting them on afterwards leaves
# their embeddings as untrained noise. <|context|> earned its place in G-Micro:
# the SFT rounds that taught grounding all depend on it.
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|context|>",
]


def build_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token=None, byte_fallback=True))

    # Digits() runs before ByteLevel and never lets a merge span two digits.
    # individual_digits=True is the whole point: without it the trainer happily
    # rebuilds "2011" as one symbol, which is the failure being fixed.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def train(corpus_files: list[Path], vocab_size: int, out_path: Path):
    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train([str(p) for p in corpus_files], trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f"saved -> {out_path}  (vocab {tokenizer.get_vocab_size()})")
    return tokenizer


def report(tokenizer):
    """Check the two things this tokenizer exists to get right."""
    import re

    vocab = tokenizer.get_vocab()
    multi_digit = [t for t in vocab if re.fullmatch(r"\s?\d{2,}", t)]
    print(f"\nwielocyfrowe tokeny w słowniku: {len(multi_digit)}  (G-Micro miał 241)")
    if multi_digit:
        print(f"  UWAGA, nie powinno ich być: {multi_digit[:10]}")

    for probe in ["1417", "8431", "1963", "W 1998 roku"]:
        ids = tokenizer.encode(probe).ids
        print(f"  {probe!r:16} -> {[tokenizer.decode([i]) for i in ids]}")

    for probe in ["Zamek zbudowano w tysiąc czterysta siedemnastym roku.",
                  "Przeuczony model językowy nie generalizuje."]:
        n = len(tokenizer.encode(probe).ids)
        print(f"  {n:>3} tokenów na {len(probe)} znaków — {probe[:40]}…")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+", type=Path)
    ap.add_argument("--vocab-size", type=int, default=48000)
    ap.add_argument("--out", type=Path, default=Path("data/tokenizer.json"))
    args = ap.parse_args()
    tok = train(args.corpus, args.vocab_size, args.out)
    report(tok)
