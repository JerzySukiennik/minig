"""Build G-Mini's initial weights out of G-Micro's finished ones.

G-Micro spent 4,060 steps and roughly ten GPU-hours learning Polish. G-Mini is
the same architecture one size up, so most of that is transferable — and at a
two-week budget where training from scratch buys only ~9% over G-Micro, a head
start is worth more than any extra layer.

Three transplants, each with a different problem to solve.

**Embeddings — the hard one.** G-Mini's tokenizer is not G-Micro's: 48k instead
of 32k, and digits split apart. A new vocabulary normally means throwing the
embedding table away and feeding random vectors into a pretrained stack, which
is how a warm start turns into an expensive way to corrupt good weights. But
measured on the two trained tokenizers, **30,688 of G-Mini's 48,000 tokens are
byte-identical strings in G-Micro's vocabulary** — 64%. Those embeddings are
copied straight across. The remaining 17,312 are initialised as the mean of
whatever G-Micro's tokenizer breaks their string into, which is the standard
approach for extending a vocabulary and is far closer to right than noise. A
token spelled " architektonicznych" starts life as the average of the pieces
G-Micro used to spell it.

**Blocks — the easy one.** n_embd, n_head and ffn_hidden are identical by
design (that is the whole reason G-Mini is 20x768 rather than the ladder's
nominal 250M), so G-Micro's twelve blocks load into G-Mini's first twelve
unchanged.

**The eight new blocks — the subtle one.** Random initialisation would inject
noise into the middle of a working stack. Instead each new block is a copy of
one of G-Micro's, with its two residual output projections zeroed. A block whose
output projection is zero contributes exactly nothing to the residual stream,
so at step zero the twenty-layer model computes precisely what the twelve-layer
model did; the new capacity then grows in from identity rather than fighting
its way out of noise. This is the trick behind depth up-scaling in SOLAR and
LLaMA-Pro.

None of this is assumed to work. `--verify` reports how much was transferred
and asserts the identity property, and the plan is still to spend 2-3 hours
comparing this against a cold start before committing the rest of the quota.
"""

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

G_MICRO = Path.home() / "Downloads/Claude/Projects/AIe/G-Micro"

# The two matrices that write back into the residual stream. Taken from the
# real checkpoint rather than guessed: a first version matched ".down.weight",
# which never fires because the key is "ffn.w_down.weight" — an underscore, not
# a dot. Zeroing nothing would have silently produced random new blocks while
# reporting success.
RESIDUAL_OUT = ("attn.proj.weight", "ffn.w_down.weight")

# Tied to tok_emb in the model, but stored as its own key, so it carries the
# old vocabulary's shape and has to be replaced alongside the embeddings.
TIED_HEAD = "lm_head.weight"


def strip_dataparallel(state):
    """DataParallel prefixes every key with 'module.'."""
    return {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}


def transplant_embeddings(old_emb, old_tok, new_tok):
    """New embedding table: copy what exists, average what does not.

    Returns the table and a count of how many rows came from each path, because
    "the warm start worked" is a claim that needs a number behind it.
    """
    old_vocab = old_tok.get_vocab()
    new_vocab = new_tok.get_vocab()
    dim = old_emb.shape[1]

    # Same scale as the rows being copied, so the two paths stay comparable.
    new_emb = torch.empty(len(new_vocab), dim)
    torch.nn.init.normal_(new_emb, mean=0.0, std=float(old_emb.std()))

    copied = averaged = fallback = 0
    for token, new_id in new_vocab.items():
        old_id = old_vocab.get(token)
        if old_id is not None:
            new_emb[new_id] = old_emb[old_id]
            copied += 1
            continue
        # Not in the old vocabulary — spell it with the old tokenizer and take
        # the mean of the pieces. add_special_tokens is off so control tokens
        # cannot leak into the average.
        piece_ids = old_tok.encode(token, add_special_tokens=False).ids
        piece_ids = [i for i in piece_ids if i < old_emb.shape[0]]
        if piece_ids:
            new_emb[new_id] = old_emb[piece_ids].mean(dim=0)
            averaged += 1
        else:
            fallback += 1   # keeps its random row
    return new_emb, dict(copied=copied, averaged=averaged, random=fallback)


def block_keys(state, layer):
    prefix = f"blocks.{layer}."
    return {k: v for k, v in state.items() if k.startswith(prefix)}


def build(old_ckpt: Path, new_layers: int, out_path: Path, verify: bool):
    ck = torch.load(old_ckpt, map_location="cpu", weights_only=False)
    old = strip_dataparallel(ck["model"])
    old_layers = 1 + max(int(k.split(".")[1]) for k in old if k.startswith("blocks."))
    print(f"G-Micro: {old_layers} bloków, krok {ck.get('step', '?')}")
    if new_layers < old_layers:
        raise SystemExit(f"G-Mini ma mieć {new_layers} bloków, mniej niż {old_layers} — "
                         "ten skrypt tylko powiększa")

    old_tok = Tokenizer.from_file(str(G_MICRO / "data" / "tokenizer-v2.json"))
    new_tok = Tokenizer.from_file("data/tokenizer.json")

    emb_key = next(k for k in old if k.endswith("tok_emb.weight"))
    new_emb, stats = transplant_embeddings(old[emb_key], old_tok, new_tok)
    total = sum(stats.values())
    print(f"osadzenia: {stats['copied']:,} skopiowane ({stats['copied']/total*100:.1f}%), "
          f"{stats['averaged']:,} uśrednione, {stats['random']:,} losowe")

    new_state = {k: v.clone() for k, v in old.items() if not k.startswith("blocks.")}
    new_state[emb_key] = new_emb
    if TIED_HEAD in new_state:
        new_state[TIED_HEAD] = new_emb.clone()

    for i in range(old_layers):
        new_state.update({k: v.clone() for k, v in block_keys(old, i).items()})

    # New blocks: copy a trained one, then zero its residual outputs so the
    # block is the identity function until training moves it.
    zeroed = 0
    for i in range(old_layers, new_layers):
        source = old_layers - (new_layers - i)      # reuse the deepest blocks
        for k, v in block_keys(old, source).items():
            new_key = k.replace(f"blocks.{source}.", f"blocks.{i}.")
            t = v.clone()
            # attn.proj and ffn.down are the two paths back into the residual
            # stream; zeroing them makes the whole block contribute nothing.
            if any(k.endswith(s) for s in RESIDUAL_OUT):
                t.zero_()
                zeroed += 1
            new_state[new_key] = t
    print(f"bloki: {old_layers} przeniesione, {new_layers - old_layers} nowych "
          f"(wyzerowano {zeroed} macierzy wyjściowych — start jako identyczność)")

    if verify:
        run_checks(new_state, old, old_layers, new_layers, new_emb, new_tok, old_tok)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": new_state, "step": 0,
                "warm_start_from": str(old_ckpt),
                "warm_start_stats": stats}, out_path)
    print(f"zapisano -> {out_path}")


def run_checks(new_state, old, old_layers, new_layers, new_emb, new_tok, old_tok):
    print("\n--- weryfikacja ---")
    for i in range(old_layers):
        for k, v in block_keys(old, i).items():
            assert torch.equal(new_state[k], v), f"blok {i} nie przeniósł się wiernie"
    print(f"  bloki 0-{old_layers-1} identyczne z G-Micro: OK")

    for i in range(old_layers, new_layers):
        outs = [v for k, v in new_state.items()
                if k.startswith(f"blocks.{i}.")
                and any(k.endswith(s) for s in RESIDUAL_OUT)]
        assert outs, f"blok {i}: nie znaleziono macierzy wyjściowych do wyzerowania"
        assert all(float(o.abs().max()) == 0.0 for o in outs), f"blok {i} nie jest identycznością"
    print(f"  bloki {old_layers}-{new_layers-1} startują jako identyczność: OK")

    # A word both tokenizers spell the same way must have the same vector.
    shared = [t for t in list(new_tok.get_vocab())[:4000] if t in old_tok.get_vocab()][:5]
    for t in shared:
        a = new_emb[new_tok.get_vocab()[t]]
        b = old[next(k for k in old if k.endswith("tok_emb.weight"))][old_tok.get_vocab()[t]]
        assert torch.equal(a, b), f"osadzenie {t!r} nie przeniosło się"
    print(f"  osadzenia wspólnych tokenów przeniesione wiernie: OK ({len(shared)} sprawdzonych)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-ckpt", type=Path,
                    default=G_MICRO / "checkpoints" / "run1" / "best.pt",
                    help="checkpoint G-Micro (domyślnie baza pretreningu, nie SFT)")
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/warm_start.pt"))
    ap.add_argument("--verify", action="store_true", default=True)
    args = ap.parse_args()
    if not args.from_ckpt.exists():
        raise SystemExit(f"nie znaleziono {args.from_ckpt}")
    build(args.from_ckpt, args.layers, args.out, args.verify)
