"""
G-Mini — a ~178M parameter decoder-only transformer, written from scratch.

Level 2 of the family ladder (G-Micro 110M -> G-Mini -> CoreG 500M -> MegaG 1B+).
The code here is G-Micro's, unchanged apart from the default shape: a model that
already trained successfully for 4,060 steps is not worth rewriting, and every
dimension except depth is deliberately identical so G-Micro's finished blocks can
be transplanted into this one (model/warm_start.py).

Architecture is GPT-2 sized but modernised (this is essentially Llama in miniature):
  - RoPE instead of learned positional embeddings
  - RMSNorm instead of LayerNorm
  - SwiGLU feed-forward instead of GELU MLP
  - no bias terms anywhere
  - pre-normalisation (norm before each sublayer, not after)

Every forward pass can optionally record its internals into `Capture`, which is
what the live network panel in the chat UI reads from.
"""

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 48000   # G-Mini tokenizer: 48k, digits split apart
    block_size: int = 1024    # context length in tokens
    n_layer: int = 20         # 12 come warm from G-Micro, 8 start as identity
    n_head: int = 12          # head_dim 64, unchanged from G-Micro
    n_embd: int = 768         # MUST equal G-Micro's: warm starting depends on it
    ffn_hidden: int = 2048    # SwiGLU hidden width (~8/3 * n_embd, rounded)
    rope_theta: float = 10000.0
    dropout: float = 0.0      # 0.0 for pretraining, raise for fine-tuning


# ---------------------------------------------------------------------------
# Capture — the hook the visualisation panel reads
# ---------------------------------------------------------------------------

@dataclass
class Capture:
    """Collects per-layer internals during a forward pass.

    Disabled by default: during training we want zero overhead. The runtime
    turns it on for a single token step, reads it, and clears it.
    """
    enabled: bool = False
    attn: list = field(default_factory=list)        # [n_layer] (n_head, T, T) attention weights
    activations: list = field(default_factory=list) # [n_layer] (T, n_embd) block output
    ffn_gate: list = field(default_factory=list)    # [n_layer] (T, ffn_hidden) neuron firings

    def clear(self):
        self.attn.clear()
        self.activations.clear()
        self.ffn_gate.clear()


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class KVCache:
    """Stores past keys and values so generation stops re-reading the context.

    Without it, emitting token N re-runs attention over all N-1 previous
    tokens — the cost of a reply grows with the square of its length. But the
    keys and values of past tokens are a pure function of those tokens, and
    those tokens do not change. So compute them once and keep them.

    With the cache each step feeds exactly one token through the network and
    attends it against the stored past.
    """

    def __init__(self, n_layer: int):
        self.k = [None] * n_layer
        self.v = [None] * n_layer

    def append(self, layer_idx: int, k, v):
        if self.k[layer_idx] is not None:
            k = torch.cat([self.k[layer_idx], k], dim=2)  # concat along time
            v = torch.cat([self.v[layer_idx], v], dim=2)
        self.k[layer_idx] = k
        self.v[layer_idx] = v
        return k, v

    @property
    def length(self) -> int:
        """How many tokens are already cached (0 before the first forward)."""
        return 0 if self.k[0] is None else self.k[0].size(2)

    def trim(self, max_len: int):
        """Drop the oldest entries so the cache never exceeds the context window."""
        if self.length <= max_len:
            return
        cut = self.length - max_len
        for i in range(len(self.k)):
            self.k[i] = self.k[i][:, :, cut:, :]
            self.v[i] = self.v[i][:, :, cut:, :]


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Normalise each vector by its root-mean-square, then rescale.

    LayerNorm subtracts the mean and divides by std. RMSNorm skips the mean
    entirely — it turns out the re-centering does almost nothing, and dropping
    it is cheaper and just as stable.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# RoPE — rotary position embeddings
# ---------------------------------------------------------------------------

def build_rope_cache(head_dim: int, seq_len: int, theta: float, device, dtype):
    """Precompute cos/sin tables for rotary embeddings.

    Idea: instead of *adding* a position vector to the token, we *rotate* the
    query and key vectors by an angle proportional to their position. Because
    a dot product between two rotated vectors depends only on the difference of
    their angles, attention automatically sees relative distance rather than
    absolute position. That is why RoPE extrapolates better than learned
    position embeddings.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)          # (T, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """Rotate (B, n_head, T, head_dim) by the cached angles."""
    x1, x2 = x.chunk(2, dim=-1)                 # split the head into two halves
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Each token builds a query ("what am I looking for?"), a key ("what do I
    offer?") and a value ("what do I pass on?"). Every token scores itself
    against all *earlier* tokens, softmaxes those scores into weights, and
    takes a weighted average of their values. The causal mask is what makes
    this a language model rather than a text encoder — a token may never look
    at its own future.
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.layer_idx = layer_idx
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, capture: Capture | None = None, kv=None):
        """kv: optional KVCache. When present, x holds only the *new* tokens and
        the keys/values of everything before them come from the cache."""
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): every head gets its own subspace
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # cos/sin were already sliced by the caller to cover exactly the
        # absolute positions of these T tokens, so a cached decode step gets
        # the rotation for position `past_len`, not for position 0.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv is not None:
            # Keys and values of past tokens never change — that is the whole
            # reason this cache is correct. Only append and reuse.
            k, v = kv.append(self.layer_idx, k, v)

        T_k = k.size(2)

        if capture is not None and capture.enabled:
            # Explicit path: we need the attention matrix itself to draw the
            # heatmap, and the fused kernel below never materialises it.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Query i (absolute position T_k - T + i) may attend to key j
            # whenever j <= that absolute position.
            causal = torch.ones(T, T_k, dtype=torch.bool, device=x.device).tril(T_k - T)
            att = att.masked_fill(~causal, float('-inf'))
            att = F.softmax(att, dim=-1)
            capture.attn.append(att[0].detach().float().cpu())  # (n_head, T, T_k)
            y = att @ v
        else:
            # Fused kernel — same maths, far less memory. Used for training
            # and for normal generation.
            #
            # is_causal=True assumes queries and keys are the same sequence. In
            # a cached decode step there are T queries against T_k > T keys, and
            # every query legitimately sees the whole cache, so the flag would
            # mask the wrong cells. Only pass it when the shapes actually match.
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(T == T_k and T > 1),
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-merge the heads
        return self.resid_drop(self.proj(y))


# ---------------------------------------------------------------------------
# Feed-forward (SwiGLU)
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """Gated feed-forward network.

    A classic MLP is `W2(gelu(W1 x))`. SwiGLU splits the first projection in
    two: one branch produces candidate values, the other produces a *gate*
    that decides how much of each value survives. The multiplication is what
    makes it "gated" — the network can learn to suppress features
    conditionally rather than always applying the same nonlinearity.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.w_up = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.w_down = nn.Linear(config.ffn_hidden, config.n_embd, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, capture: Capture | None = None):
        gate = F.silu(self.w_gate(x))
        hidden = gate * self.w_up(x)
        if capture is not None and capture.enabled:
            # These are the closest thing this model has to individual
            # "neurons firing" — one scalar per hidden unit per token.
            capture.ffn_gate.append(hidden[0].detach().float().cpu())
        return self.drop(self.w_down(hidden))


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """One transformer layer: attention, then feed-forward, both residual.

    Note the shape `x = x + sublayer(norm(x))`. The residual stream `x` is
    never overwritten — each layer only ever *adds* to it. That is why you can
    read the residual stream at any depth and get a meaningful, progressively
    refined representation, which is exactly what the layer-activation strip
    in the UI is showing.
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.norm_attn = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.norm_ffn = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)

    def forward(self, x, cos, sin, capture: Capture | None = None, kv=None):
        x = x + self.attn(self.norm_attn(x), cos, sin, capture, kv)
        x = x + self.ffn(self.norm_ffn(x), capture)
        if capture is not None and capture.enabled:
            capture.activations.append(x[0].detach().float().cpu())
        return x


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.norm_final = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the embedding matrix and the output matrix are the same
        # tensor. Saves ~25M parameters and consistently helps at this scale —
        # "which token is this" and "which token comes next" want the same
        # vocabulary geometry.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale down the residual-path outputs so that the residual stream does
        # not blow up as depth grows (GPT-2 trick).
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('w_down.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self._rope_cache = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def _rope(self, start, T, device, dtype):
        """cos/sin for absolute positions [start, start+T)."""
        need = start + T
        if (self._rope_cache is None
                or self._rope_cache[0].shape[0] < need
                or self._rope_cache[0].device != device
                or self._rope_cache[0].dtype != dtype):
            self._rope_cache = build_rope_cache(
                self.config.n_embd // self.config.n_head,
                max(need, self.config.block_size),
                self.config.rope_theta, device, dtype,
            )
        cos, sin = self._rope_cache
        return cos[start:need], sin[start:need]

    def forward(self, idx, targets=None, capture: Capture | None = None, kv=None,
                return_logits: bool = True):
        """idx: (B, T) token ids. Returns (logits, loss).

        With `kv`, idx holds only the tokens not yet seen and the rest of the
        sequence is read from the cache.
        """
        B, T = idx.shape
        past = kv.length if kv is not None else 0
        assert past + T <= self.config.block_size, \
            f"sequence of {past + T} exceeds block_size {self.config.block_size}"

        if capture is not None and capture.enabled:
            capture.clear()

        x = self.drop(self.tok_emb(idx))
        # Positions continue from where the cache left off — otherwise every
        # generated token would be rotated as if it sat at position 0.
        cos, sin = self._rope(past, T, idx.device, x.dtype)

        for block in self.blocks:
            x = block(x, cos, sin, capture, kv)

        x = self.norm_final(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            # Training discards the logits, and under DataParallel every return
            # value is shipped back to GPU 0 and concatenated. At batch 16 the
            # logits are 16x1024x32000 — over 1 GB per step, moved for nothing
            # and a plausible way to run a 16 GB card out of memory.
            return (logits if return_logits else None), loss

        # Inference: only the last position matters, so only project that one.
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50,
                 repetition_penalty=1.3,
                 capture: Capture | None = None, use_cache: bool = True):
        """Sample tokens autoregressively, yielding (token_id, probs) each step.

        With use_cache the prompt is processed once in a single "prefill" pass,
        and every step after that feeds exactly one token.

        A model this small has no real notion of "I already said this" — low
        temperature/top_k make output more locally coherent but also more
        likely to lock into a repeat loop ("Warszawa, Warszawa, Warszawa..."),
        since nothing discourages a token that already looks probable from
        winning again next step. repetition_penalty pushes already-seen
        tokens' logits toward zero (CTRL-style: divide positive logits,
        multiply negative ones) so the same choice doesn't keep re-winning.
        """
        self.eval()
        kv = KVCache(self.config.n_layer) if use_cache else None

        # Prefill: the model has seen nothing yet, so the whole prompt goes in.
        idx_cond = idx[:, -self.config.block_size:]
        step_in = idx_cond

        for _ in range(max_new_tokens):
            logits, _ = self(step_in, capture=capture, kv=kv)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                seen = torch.unique(idx[0])
                seen_logits = logits[0, seen]
                logits[0, seen] = torch.where(
                    seen_logits > 0, seen_logits / repetition_penalty, seen_logits * repetition_penalty)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            yield next_id.item(), probs[0]

            if kv is not None:
                # Keep room for the token we are about to append.
                kv.trim(self.config.block_size - 1)
                step_in = next_id           # decode: one token at a time
            else:
                step_in = idx[:, -self.config.block_size:]


if __name__ == '__main__':
    cfg = GPTConfig()
    m = GPT(cfg)
    print(f"total params:     {m.num_params():,}")
    print(f"non-embedding:    {m.num_params(non_embedding=True):,}")
    # Targets must be the inputs shifted left by one: position t predicts t+1.
    # Feeding unshifted targets lets the model see the answer and reports a
    # misleadingly low loss.
    x = torch.randint(0, cfg.vocab_size, (2, 65))
    logits, loss = m(x[:, :-1], targets=x[:, 1:])
    print(f"forward ok — logits {tuple(logits.shape)}, loss {loss.item():.3f}")
    print(f"expected at random init: {math.log(cfg.vocab_size):.3f} (uniform guess)")
