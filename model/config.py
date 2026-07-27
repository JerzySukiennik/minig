"""MiniG's shape, and why every number in it is the one it is.

Level 2 of the ladder (MicroG 100M -> MiniG -> CoreG 500M -> MegaG 1B+). The
architecture itself is MicroG's — RoPE, RMSNorm, SwiGLU, byte-level BPE — which
is deliberate twice over: it is known to train, and keeping n_embd identical is
what lets MicroG's finished blocks be reused (see warm_start.py).

Every figure below came off `bench/size_probe.py` on a real Kaggle T4 x2 rather
than out of an extrapolation. Two earlier estimates in this project were wrong
by roughly 2x, both by assuming throughput scales cleanly with parameters.

  n_layer 20, n_embd 768        178.5M parameters with the 48k vocabulary
  block_size 1024               2048 was measured and rejected — see below
  vocab_size 48000              digits split apart; data/train_tokenizer.py

**Why 20 layers of 768 rather than the ladder's nominal 250M.** The T4's memory
wall sits between 204M and 282M: up to 204M a micro-batch of 8 fits and the card
runs at 8,900-16,800 tok/s, above it the micro-batch halves and throughput
collapses to 6,150. Past that wall a bigger model is *slower to a worse result*
for the same quota. Within the usable range every size lands within 0.7 points
of the others after two weeks, so the tie-break is warm-starting, and that needs
n_embd to match MicroG exactly. 20 layers puts the token budget at ~20.7 tokens
per parameter, which is Chinchilla's optimum for 60 GPU-hours.

**Why 1024 tokens of context and not 2048.** Attention activations grow with
sequence length, so 2048 drops every configuration one micro-batch step and
costs 20-25% throughput. At a two-week budget that turns a ~9% improvement over
MicroG into ~2%. RoPE extrapolates, so context can be extended later without
retraining from scratch — this is a deferral, not a ceiling.

**What this model is honestly expected to be.** About 9% better than MicroG per
character, which nobody notices in conversation. The improvements that will
actually show are qualitative and independent of size: numbers that survive
being copied, the whole SFT recipe from MicroG's rounds A-D, and Home Assistant
commands. Anyone reading this expecting a large jump should read that sentence
again before spending the quota.
"""

from dataclasses import dataclass


@dataclass
class MiniGConfig:
    vocab_size: int = 48000
    block_size: int = 1024
    n_layer: int = 20
    n_head: int = 12          # head_dim = 64, as in MicroG
    n_embd: int = 768         # MUST match MicroG for warm starting
    ffn_hidden: int = 2048    # SwiGLU width, ~8/3 x n_embd
    rope_theta: float = 10000.0
    dropout: float = 0.0      # 0.0 for pretraining, raised for fine-tuning


# Measured on Kaggle T4 x2, block_size 1024, gradient accumulation to a
# 65,536-token step — the same shape the real training loop uses.
MEASURED = {
    "micro_batch": 8,
    "tok_per_s_single_gpu": 10_070,   # interpolated for 20 layers; 16 layers
                                      # measured 12,948 and 18x896 measured 8,946
    "dataparallel_speedup": 1.7,      # measured on MicroG, not the nominal 2.0
}

TRAINING = {
    "target_tokens": 3_700_000_000,   # ~20.7 per parameter, Chinchilla optimum
    "estimated_hours": 60,            # two weeks of a 30h/week quota
    "tokens_per_step": 65_536,
}


def summary() -> str:
    cfg = MiniGConfig()
    emb = cfg.vocab_size * cfg.n_embd
    per_layer = 4 * cfg.n_embd ** 2 + 3 * cfg.n_embd * cfg.ffn_hidden
    total = emb + cfg.n_layer * per_layer
    steps = TRAINING["target_tokens"] // TRAINING["tokens_per_step"]
    return (f"MiniG: {cfg.n_layer}x{cfg.n_embd}, vocab {cfg.vocab_size}, "
            f"ctx {cfg.block_size}\n"
            f"  parametry: {total/1e6:.1f}M  (osadzenia {emb/1e6:.1f}M, "
            f"warstwy {cfg.n_layer*per_layer/1e6:.1f}M)\n"
            f"  budżet: {TRAINING['target_tokens']/1e9:.1f}B tokenów = "
            f"{steps:,} kroków po {TRAINING['tokens_per_step']:,}\n"
            f"  {TRAINING['target_tokens']/total:.1f} tokenów na parametr")


if __name__ == "__main__":
    print(summary())
