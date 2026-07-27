"""
Pretraining loop for G-Micro.

Designed around one hard constraint: free Kaggle/Colab sessions die after a few
hours, without warning. So every piece of state that the run depends on —
weights, optimiser moments, step counter, RNG — is checkpointed together and
restored together. A resumed run must be indistinguishable from an
uninterrupted one, otherwise "train once" quietly becomes "train badly".

Run:
    python train/train.py --data data/pl --out checkpoints/run1
    python train/train.py --data data/pl --out checkpoints/run1 --resume
"""

import argparse
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.gpt import GPT, GPTConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TokenData:
    """Random windows over a memory-mapped token array, prefetched on a
    background thread.

    np.memmap means we never load the whole corpus into RAM — the OS pages in
    only the slices we touch. A 2B-token corpus is 4 GB on disk and costs us
    almost nothing resident.

    Two things matter for throughput here, and the original version had
    neither: on a live Kaggle T4x2 run this class alone held the GPUs at
    ~24k tok/s (~0.68s per micro-batch) — implausibly slow for a 110M model
    with tensor cores, meaning the GPUs were idle waiting on Python.

      Vectorised gather. Building each of the 16 rows with its own Python-level
      slice + astype + torch.stack call pays interpreter overhead 16 times
      over. A single fancy-index gather does the same read in one call.

      Prefetching. Without it, CPU batch prep and GPU compute strictly
      alternate — the GPU sits idle for every millisecond Python spends
      building the next batch. A background thread prepares batch N+1 while
      the GPU is still busy with batch N, so the two overlap instead of
      serialising.
    """

    def __init__(self, path: str, block_size: int, device: str, prefetch: int = 3):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.device = device
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{path} holds only {len(self.data)} tokens")
        self._queue = queue.Queue(maxsize=prefetch)
        self._batch_size = None
        self._thread = None

    def _make_cpu_batch(self, batch_size: int):
        T = self.block_size
        starts = np.random.randint(0, len(self.data) - T - 1, size=batch_size)
        # One gather for x and its y-shifted-by-one neighbour together, instead
        # of batch_size separate slices — the loop moves from Python into numpy's
        # C implementation.
        offsets = starts[:, None] + np.arange(T + 1, dtype=np.int64)[None, :]
        chunk = self.data[offsets].astype(np.int64)
        x = torch.from_numpy(np.ascontiguousarray(chunk[:, :T]))
        # y is x shifted by one: position t predicts t+1. Getting this wrong is
        # the classic silent bug — the loss looks great and the model is useless.
        y = torch.from_numpy(np.ascontiguousarray(chunk[:, 1:T + 1]))
        return x, y

    def _prefetch_loop(self, batch_size: int):
        while True:
            self._queue.put(self._make_cpu_batch(batch_size))

    def batch(self, batch_size: int):
        if self._thread is None:
            self._batch_size = batch_size
            self._thread = threading.Thread(
                target=self._prefetch_loop, args=(batch_size,), daemon=True)
            self._thread.start()
        assert batch_size == self._batch_size, \
            "batch_size changed after prefetching started"

        x, y = self._queue.get()
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x.to(self.device), y.to(self.device)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def lr_at(step, warmup, total, lr_max, lr_min):
    """Linear warmup, then cosine decay.

    Warmup exists because Adam's second-moment estimate is garbage for the
    first few hundred steps; taking full-size steps then can wreck the model
    before it has learned anything. Cosine decay spends the end of training
    taking small steps, which is where most of the final quality is won.
    """
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step >= total:
        return lr_min
    ratio = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * ratio))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_ckpt(path: Path, model, opt, step, best_val, cfg, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # DataParallel prefixes every key with "module.". Strip it so a checkpoint
    # trained on two GPUs still loads into a plain model on the laptop.
    core = model.module if hasattr(model, "module") else model
    torch.save({
        "model": core.state_dict(),
        "optimizer": opt.state_dict(),      # Adam moments — dropping these
                                            # causes a visible loss spike on resume
        "step": step,
        "best_val": best_val,
        "config": cfg.__dict__,
        "args": vars(args),
        "torch_rng": torch.get_rng_state(),
    }, tmp)
    # Atomic replace: a session killed mid-write leaves the previous good
    # checkpoint intact rather than a truncated file.
    os.replace(tmp, path)


def load_ckpt(path: Path, model, opt, device):
    # weights_only=False: PyTorch 2.6 flipped the default, which then refuses
    # to unpickle the plain pathlib.Path stored in ck["args"] below. This is
    # our own checkpoint, not an untrusted download, so that's fine.
    ck = torch.load(path, map_location=device, weights_only=False)
    core = model.module if hasattr(model, "module") else model
    core.load_state_dict(ck["model"])
    if opt is not None and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
    if "torch_rng" in ck:
        torch.set_rng_state(ck["torch_rng"].cpu().to(torch.uint8))
    return ck["step"], ck.get("best_val", float("inf"))


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, batch_size, iters, ctx):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = data.batch(batch_size)
        with ctx:
            _, loss = model(x, targets=y, return_logits=False)
        losses.append(loss.mean().item())   # .mean() for the DataParallel case
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/pl", help="prefix: <data>_train.bin")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/run1"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--warm-start", type=str, default=None,
                    help="checkpoint z model/warm_start.py — startuje z bloków G-Micro "
                         "zamiast z losowych wag (ignorowane przy --resume)")

    ap.add_argument("--batch-size", type=int, default=4)      # zmierzone: przy 8 brak pamięci
                                                             # (logity 8x1024x48000 nie mieszczą się)
    ap.add_argument("--grad-accum", type=int, default=16)     # 4*16*1024 = 65 536 tokenów/krok
    ap.add_argument("--max-steps", type=int, default=54932)   # 3.6B tokenów / 65 536
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--single-gpu", action="store_true",
                    help="ignore extra GPUs; use if DataParallel misbehaves")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 has fp32's exponent range, so it needs no loss scaling. But
    # torch.cuda.is_bf16_supported() answers "can this run at all", not "does
    # this have tensor cores for it" — it reports True on Turing (T4, compute
    # capability 7.5) even though bf16 there has no hardware acceleration and
    # runs on the slow path. Measured on a Kaggle T4x2: bf16 gave 28k tok/s,
    # roughly half of what fp16 tensor cores should deliver. Bf16 tensor cores
    # only exist from Ampere (capability 8.0) onward, so gate on that directly.
    use_amp = device == "cuda"
    bf16 = use_amp and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
           if use_amp else torch.autocast(device_type="cpu", enabled=False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not bf16)

    print(f"device={device}  amp={'bf16' if bf16 else 'fp16' if use_amp else 'off'}")

    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    print(f"params: {model.num_params():,}")

    # Weight decay only on matrices. Biases, norm gains and embeddings are
    # 1-D or lookup tables; decaying them mostly just degrades the model.
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        fused=(device == "cuda"),
    )

    train_data = TokenData(f"{args.data}_train.bin", cfg.block_size, device)
    val_data = TokenData(f"{args.data}_val.bin", cfg.block_size, device)

    step, best_val = 0, float("inf")
    ckpt_path = args.out / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        step, best_val = load_ckpt(ckpt_path, model, opt, device)
        print(f"resumed from step {step} (best val {best_val:.4f})")
    elif args.warm_start:
        # Start from G-Micro's finished blocks rather than from noise. Weights
        # only — no optimiser state, no step count: this is step 0 of a new
        # model, not the continuation of an old run. Built by
        # model/warm_start.py, which also transplants the embedding table
        # across the two different tokenizers.
        wpath = Path(args.warm_start)
        if not wpath.exists():
            raise SystemExit(f"brak checkpointu ciepłego startu: {wpath} "
                             "(uruchom model/warm_start.py)")
        wck = torch.load(wpath, map_location=device, weights_only=False)
        core = model.module if hasattr(model, "module") else model
        core.load_state_dict(wck["model"])
        stats = wck.get("warm_start_stats", {})
        print(f"warm start z {wck.get('warm_start_from', '?')}: "
              f"{stats.get('copied', '?')} osadzeń skopiowanych, "
              f"{stats.get('averaged', '?')} uśrednionych, "
              f"{stats.get('random', '?')} losowych")
        if args.lr > 4e-4:
            # Transplanted weights are already in a good basin; the from-scratch
            # peak LR is enough to knock them out of it. Warned rather than
            # silently overridden — a surprise LR is worse than a loud one.
            print(f"UWAGA: lr={args.lr} to wartość dla treningu od zera. "
                  f"Przy ciepłym starcie zalecane ~3e-4, inaczej ryzykujesz "
                  f"rozbicie przeniesionych wag.")

    if args.compile:
        model = torch.compile(model)

    # Kaggle hands out two T4s. DataParallel is the crude way to use both — it
    # replicates the model each step and gathers gradients on GPU 0 — but it is
    # one line and works inside a notebook, whereas DDP needs a process launcher
    # that Kaggle notebooks make awkward. Expect ~1.7x, not 2x.
    n_gpu = 0 if args.single_gpu else (torch.cuda.device_count() if device == "cuda" else 0)
    if n_gpu > 1:
        print(f"using {n_gpu} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    tokens_per_step = args.batch_size * args.grad_accum * cfg.block_size
    print(f"tokens/step: {tokens_per_step:,}   "
          f"total: {tokens_per_step * args.max_steps / 1e9:.2f}B")

    log_path = args.out / "log.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)
    model.train()
    t0 = time.time()

    while step < args.max_steps:
        lr = lr_at(step, args.warmup, args.max_steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        # Gradient accumulation: a 500k-token batch will not fit in memory, so
        # we sum gradients over many small batches. Dividing the loss by
        # grad_accum keeps the gradient magnitude identical to one big batch.
        for micro in range(args.grad_accum):
            x, y = train_data.batch(args.batch_size)
            with ctx:
                # return_logits=False: the logits are unused here and gathering
                # them across GPUs costs a gigabyte a step.
                _, loss = model(x, targets=y, return_logits=False)
                # DataParallel returns one loss per GPU; mean() collapses it
                # back to a scalar. On a single device this is a no-op.
                loss = loss.mean() / args.grad_accum
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            tps = tokens_per_step * args.log_every / dt
            train_loss = loss.item() * args.grad_accum
            print(f"step {step:>6}  loss {train_loss:6.3f}  lr {lr:.2e}  "
                  f"{tps/1e3:7.1f}k tok/s", flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step, "train_loss": train_loss,
                                    "lr": lr, "tok_per_s": tps}) + "\n")
            t0 = time.time()

        if step % args.eval_every == 0:
            val = evaluate(model, val_data, args.batch_size, args.eval_iters, ctx)
            ppl = math.exp(min(val, 20))
            print(f"  -> val loss {val:.4f}  (perplexity {ppl:.1f})", flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step, "val_loss": val, "ppl": ppl}) + "\n")
            if val < best_val:
                best_val = val
                save_ckpt(args.out / "best.pt", model, opt, step, best_val, cfg, args)
            t0 = time.time()

        if step % args.ckpt_every == 0:
            save_ckpt(ckpt_path, model, opt, step, best_val, cfg, args)

    save_ckpt(ckpt_path, model, opt, step, best_val, cfg, args)
    print(f"done at step {step}, best val {best_val:.4f}")


if __name__ == "__main__":
    main()
