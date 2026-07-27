"""Does the whole pipeline actually hold together, before any quota is spent?

MicroG's pretraining once restarted from a stale checkpoint and burned real GPU
hours because a path assumption failed silently. Everything here is the kind of
thing that fails loudly on a laptop in a minute and expensively on Kaggle in an
hour: the model builds at the intended size, the warm-start checkpoint loads
into it, a forward and backward pass run, and the tokenizer round-trips a
number without merging its digits.

    python bench/smoke.py
"""

import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model.gpt import GPT, GPTConfig  # noqa: E402

ok = True


def check(label, condition, detail=""):
    global ok
    ok = ok and bool(condition)
    print(f"  [{'OK ' if condition else 'BŁĄD'}] {label}" + (f" — {detail}" if detail else ""))


print("=== model ===")
cfg = GPTConfig()
model = GPT(cfg)
n = model.num_params()
check("kształt zgodny z planem", (cfg.n_layer, cfg.n_embd, cfg.vocab_size) == (20, 768, 48000),
      f"{cfg.n_layer}x{cfg.n_embd}, słownik {cfg.vocab_size}")
check("rozmiar ~178M", 175e6 < n < 182e6, f"{n/1e6:.1f}M")
check("n_embd zgodne z MicroG (warunek ciepłego startu)", cfg.n_embd == 768)

print("\n=== tokenizer ===")
tok_path = REPO / "data" / "tokenizer.json"
if tok_path.exists():
    tok = Tokenizer.from_file(str(tok_path))
    check("słownik 48k", tok.get_vocab_size() == 48000, str(tok.get_vocab_size()))
    ids = tok.encode("1417").ids
    check("cyfry rozbite pojedynczo", len(ids) == 4,
          f"1417 -> {[tok.decode([i]) for i in ids]}")
    for special in ("<|user|>", "<|assistant|>", "<|endoftext|>", "<|context|>"):
        check(f"token {special} istnieje", tok.token_to_id(special) is not None)
    check("słownik modelu = słownik tokenizera", tok.get_vocab_size() == cfg.vocab_size)
else:
    check("plik tokenizera istnieje", False, str(tok_path))

print("\n=== ciepły start ===")
warm = REPO / "checkpoints" / "warm_start.pt"
if warm.exists():
    ck = torch.load(warm, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    check("ładuje się do modelu bez brakujących wag", not missing, str(missing[:3]))
    check("bez nadmiarowych wag", not unexpected, str(unexpected[:3]))
    stats = ck.get("warm_start_stats", {})
    check("zero losowych osadzeń", stats.get("random") == 0, str(stats))
    # The eight new blocks must start as identity, or the transplant injected
    # noise into the middle of a trained stack.
    zeros = [k for k in ck["model"]
             if k.startswith(("blocks.12.", "blocks.13.", "blocks.14.", "blocks.15.",
                              "blocks.16.", "blocks.17.", "blocks.18.", "blocks.19."))
             and k.endswith(("attn.proj.weight", "ffn.w_down.weight"))]
    check("nowe bloki startują jako identyczność", zeros and
          all(float(ck["model"][k].abs().max()) == 0.0 for k in zeros), f"{len(zeros)} macierzy")
else:
    print("  (pominięte — uruchom model/warm_start.py)")

print("\n=== krok treningu ===")
x = torch.randint(0, cfg.vocab_size, (2, 128))
y = torch.randint(0, cfg.vocab_size, (2, 128))
logits, loss = model(x, targets=y, return_logits=False)
check("forward zwraca skończoną stratę", torch.isfinite(loss), f"{loss.item():.3f}")
import math  # noqa: E402
# Not compared against ln(vocab): the inputs are random token ids, and a model
# carrying MicroG's weights confidently predicts Polish-shaped continuations
# that random targets never match, so it scores *worse* than uniform here. That
# is the expected reading, and calling it "close to random" would be wrong.
# What matters is only that the number is finite and in a sane range.
check("strata w sensownym zakresie", 1.0 < loss.item() < 30.0,
      f"{loss.item():.2f} (uniform = ln(48000) = {math.log(cfg.vocab_size):.2f}; "
      f"wyżej jest oczekiwane na losowych tokenach)")
loss.backward()
grads = [p.grad for p in model.parameters() if p.grad is not None]
check("backward wypełnia gradienty", len(grads) > 0, f"{len(grads)} tensorów")
check("gradienty skończone", all(torch.isfinite(g).all() for g in grads))

print("\n=== dane SFT ===")
home = REPO / "data" / "home_sft.jsonl"
check("komendy Home Assistant wygenerowane", home.exists(),
      f"{sum(1 for _ in home.open(encoding='utf-8')):,} par" if home.exists() else "brak")

print("\n" + ("WSZYSTKO PRZESZŁO" if ok else "SĄ BŁĘDY — nie odpalaj treningu"))
sys.exit(0 if ok else 1)
