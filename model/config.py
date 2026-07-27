"""Why G-Mini has the shape it has, and what training it is expected to cost.

The shape itself lives in `GPTConfig` in gpt.py — this module deliberately does
not restate it. G-Micro shipped two copies of its Kaggle kernel that drifted
apart silently and cost real GPU hours; a second definition of the model's
dimensions would fail the same way, and this file would be the one that lies.

Everything below came off `bench/size_probe.py` running on a real Kaggle T4 x2.
Two earlier estimates in this project were wrong by roughly 2x, both from
assuming throughput scales cleanly with parameter count. It does not: memory
does, and memory decides the micro-batch, and the micro-batch decides
throughput.

**Why 20 layers of 768 rather than the ladder's nominal 250M.** The T4's memory
wall sits between 204M and 282M. Up to 204M a micro-batch of 8 fits and the card
holds 8,900-16,800 tok/s; above it the micro-batch halves and throughput
collapses to 6,150. Past that wall a bigger model is slower to a worse result
for the same quota. Inside the usable range every size lands within 0.7 points
of the others after two weeks, so the tie-break is warm starting — and that
requires n_embd to match G-Micro exactly.

**Why 1024 tokens of context.** Attention activations grow with sequence
length, so 2048 costs every configuration one micro-batch step and 20-25% of
its throughput. At this budget that turns a ~9% gain over G-Micro into ~2%. RoPE
extrapolates, so context can be widened later without retraining — a deferral,
not a ceiling.

**What this model is honestly expected to be.** About 9% better than G-Micro per
character, which nobody notices in conversation. Perplexity is not comparable
across different tokenizers, so the larger-looking numbers are an artefact; per
character is the fair comparison and it is single digits. What will actually
show is qualitative and size-independent: numbers that survive being copied,
Home Assistant commands, and the whole SFT recipe from G-Micro's rounds A-D.
Anyone about to spend two weeks of quota should read that paragraph twice.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.gpt import GPT, GPTConfig  # noqa: E402

# Measured directly on 2026-07-27 for this exact shape, after the interpolated
# figures turned out to be optimistic. micro_batch 8 does NOT fit: at a 48k
# vocabulary the logits alone are 8 x 1024 x 48000, and that is what pushed it
# over — a 204M model with a 32k vocabulary had fitted at 8. The anchor run
# returned G-Micro at 16,527 tok/s against its known ~16,800, so the numbers
# below are trustworthy.
MEASURED = {
    "micro_batch": 4,                 # 8 -> OOM
    "grad_accum": 16,                 # 4 x 16 x 1024 = 65,536 tokens per step
    "tok_per_s_single_gpu": 9_291,    # interpolation had said 10,070
    "dataparallel_speedup": 1.7,      # measured on G-Micro; the nominal 2.0 is wrong
    "peak_memory_gb": 8.8,            # of the T4's 15.6, at micro_batch 4
}

TRAINING = {
    "target_tokens": 3_600_000_000,   # ~20.2 per parameter — Chinchilla's optimum here
    "estimated_hours": 63,            # measured, not interpolated: 2.1 weeks of quota
    "tokens_per_step": 65_536,
    "warm_start": True,               # see model/warm_start.py
}


def summary() -> str:
    cfg = GPTConfig()
    model = GPT(cfg)
    total = model.num_params()
    non_emb = model.num_params(non_embedding=True)
    steps = TRAINING["target_tokens"] // TRAINING["tokens_per_step"]
    hours = (TRAINING["target_tokens"]
             / (MEASURED["tok_per_s_single_gpu"] * MEASURED["dataparallel_speedup"]) / 3600)
    return (f"G-Mini: {cfg.n_layer}x{cfg.n_embd}, słownik {cfg.vocab_size}, "
            f"kontekst {cfg.block_size}\n"
            f"  parametry: {total/1e6:.1f}M  (poza osadzeniami {non_emb/1e6:.1f}M)\n"
            f"  budżet: {TRAINING['target_tokens']/1e9:.1f}B tokenów = {steps:,} kroków "
            f"po {TRAINING['tokens_per_step']:,}\n"
            f"  {TRAINING['target_tokens']/total:.1f} tokenów na parametr, "
            f"szacowane {hours:.0f} h na T4x2 ({hours/30:.1f} tyg. quoty)")


if __name__ == "__main__":
    print(summary())
