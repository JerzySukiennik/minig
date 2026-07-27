"""
Instruction tuning — turn the pretrained base model into something that answers.

Pretraining left MicroG able to continue Polish text. Asked a question it
produces more questions, because that is what follows a question on the open
web. This stage teaches the one missing habit: after <|assistant|>, answer.

Three differences from pretraining, all of them load-bearing:

  Masked loss. Only the assistant's reply is scored. Training on the user's
  turn as well would reinforce question-writing, the exact behaviour we are
  removing.

  Much lower learning rate. The model already knows Polish; 6e-4 would wash
  that out in favour of 90k Alpaca rows. 3e-5 adjusts behaviour without
  destroying the pretrained knowledge (catastrophic forgetting).

  Windows start at example boundaries, never mid-reply.

Run:
    python train/finetune.py --init checkpoints/run1/best.pt --out checkpoints/sft
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.gpt import GPT, GPTConfig  # noqa: E402
from train.train import lr_at, save_ckpt  # noqa: E402


class SFTData:
    """Windows over the instruction stream, aligned to example starts."""

    def __init__(self, prefix: str, block_size: int, device: str, val_frac=0.02):
        self.tokens = np.memmap(f"{prefix}_tokens.bin", dtype=np.uint16, mode="r")
        self.mask = np.memmap(f"{prefix}_mask.bin", dtype=np.uint8, mode="r")
        offsets = np.fromfile(f"{prefix}_offsets.bin", dtype=np.int64)
        # Hold out the tail rather than a random sample: the two source corpora
        # are concatenated, so a random split would put near-duplicates on both
        # sides and flatter the validation loss.
        cut = int(len(offsets) * (1 - val_frac))
        self.train_off, self.val_off = offsets[:cut], offsets[cut:]
        self.block_size = block_size
        self.device = device

    def batch(self, batch_size: int, split="train"):
        offs = self.train_off if split == "train" else self.val_off
        pick = offs[torch.randint(len(offs), (batch_size,)).numpy()]
        B, T = batch_size, self.block_size
        x = np.zeros((B, T), dtype=np.int64)
        y = np.full((B, T), -1, dtype=np.int64)   # -1 = ignored by cross_entropy
        for i, s in enumerate(pick):
            s = int(s)
            chunk = self.tokens[s:s + T + 1].astype(np.int64)
            m = self.mask[s:s + T + 1].astype(bool)
            n = len(chunk) - 1
            if n <= 0:
                continue
            x[i, :n] = chunk[:n]
            # Predicting position t+1 is scored only if t+1 is inside the reply.
            tgt = chunk[1:n + 1]
            y[i, :n] = np.where(m[1:n + 1], tgt, -1)
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        if self.device == "cuda":
            return xt.pin_memory().to("cuda", non_blocking=True), \
                   yt.pin_memory().to("cuda", non_blocking=True)
        return xt.to(self.device), yt.to(self.device)


@torch.no_grad()
def evaluate(model, data, batch_size, iters, ctx):
    model.eval()
    out = []
    for _ in range(iters):
        x, y = data.batch(batch_size, "val")
        with ctx:
            _, loss = model(x, targets=y, return_logits=False)
        out.append(loss.mean().item())
    model.train()
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True, help="pretrained checkpoint")
    ap.add_argument("--data", type=str, default="data/pl_sft")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/sft"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min-lr", type=float, default=3e-6)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--single-gpu", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    # See train.py: is_bf16_supported() is True on T4 without tensor cores for
    # it, which halved throughput there. Gate on Ampere+ instead.
    bf16 = use_amp and torch.cuda.get_device_capability()[0] >= 8
    ctx = (torch.autocast(device_type="cuda",
                          dtype=torch.bfloat16 if bf16 else torch.float16)
           if use_amp else torch.autocast(device_type="cpu", enabled=False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not bf16)

    cfg = GPTConfig()
    model = GPT(cfg).to(device)

    ck = torch.load(args.init, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"loaded base from {args.init} (pretrain step {ck.get('step','?')}, "
          f"val {ck.get('best_val', float('nan')):.4f})")

    decay = [p for _, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=(device == "cuda"))

    data = SFTData(args.data, cfg.block_size, device)
    tokens_per_step = args.batch_size * args.grad_accum * cfg.block_size
    total_tokens = len(data.tokens) * args.epochs
    max_steps = max(1, int(total_tokens / tokens_per_step))
    print(f"{len(data.tokens)/1e6:.1f}M tokens x {args.epochs} epochs "
          f"-> {max_steps} steps of {tokens_per_step:,}")

    step, best_val = 0, float("inf")
    if args.resume and (args.out / "ckpt.pt").exists():
        rc = torch.load(args.out / "ckpt.pt", map_location=device, weights_only=False)
        model.load_state_dict(rc["model"]); opt.load_state_dict(rc["optimizer"])
        step, best_val = rc["step"], rc.get("best_val", float("inf"))
        print(f"resumed at step {step}")

    if not args.single_gpu and device == "cuda" and torch.cuda.device_count() > 1:
        print(f"using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "log.jsonl"
    model.train()
    t0 = time.time()

    while step < max_steps:
        lr = lr_at(step, args.warmup, max_steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)

        for _ in range(args.grad_accum):
            x, y = data.batch(args.batch_size)
            with ctx:
                _, loss = model(x, targets=y, return_logits=False)
                loss = loss.mean() / args.grad_accum
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt); scaler.update()
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            tl = loss.item() * args.grad_accum
            print(f"step {step:>5}/{max_steps}  loss {tl:6.3f}  lr {lr:.2e}  "
                  f"{tokens_per_step*args.log_every/dt/1e3:6.1f}k tok/s", flush=True)
            with log.open("a") as f:
                f.write(json.dumps({"step": step, "train_loss": tl, "lr": lr}) + "\n")
            t0 = time.time()

        if step % args.eval_every == 0:
            v = evaluate(model, data, args.batch_size, 20, ctx)
            print(f"  -> val {v:.4f}  (ppl {math.exp(min(v,20)):.1f})", flush=True)
            with log.open("a") as f:
                f.write(json.dumps({"step": step, "val_loss": v}) + "\n")
            if v < best_val:
                best_val = v
                save_ckpt(args.out / "best.pt", model, opt, step, best_val, cfg, args)
            t0 = time.time()

        if step % args.ckpt_every == 0:
            save_ckpt(args.out / "ckpt.pt", model, opt, step, best_val, cfg, args)

    save_ckpt(args.out / "ckpt.pt", model, opt, step, best_val, cfg, args)
    print(f"done — {max_steps} steps, best val {best_val:.4f}")


if __name__ == "__main__":
    main()
