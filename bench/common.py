"""Shared loading code for the MicroG benchmark suite."""

import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.gpt import GPT, GPTConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load_model(ckpt_path, device="cpu"):
    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck.get("step", "?"), ck.get("best_val", float("nan"))


def load_tokenizer(path=None):
    path = path or (REPO / "data" / "tokenizer-v2.json")
    return Tokenizer.from_file(str(path))


@torch.no_grad()
def score_sentence(model, tok, text, bos_context=""):
    """Total log-probability the model assigns to `text` under teacher forcing.

    Higher (less negative) means the model finds the sentence more plausible.
    Used both by the perplexity benchmark (scored over held-out corpus
    windows) and the inflection probe (scored over hand-written minimal
    pairs) — same operation, different inputs.
    """
    ids = tok.encode(bos_context + text).ids
    if len(ids) < 2:
        raise ValueError(f"sentence too short to score: {text!r}")
    x = torch.tensor([ids[:-1]])
    y = torch.tensor([ids[1:]])
    _, loss = model(x, targets=y, return_logits=False)
    n_tokens = len(ids) - 1
    return -loss.item() * n_tokens, n_tokens  # (total log-prob, token count)
