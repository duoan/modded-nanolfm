"""LFM2-style hybrid backbone (training-only, single-file).

This module mirrors the architecture of HuggingFace's ``Lfm2`` dense model
(``transformers/models/lfm2/modular_lfm2.py``) but strips out everything that's
irrelevant for a from-scratch *training* speedrun:

* No HuggingFace ``Config``/``PreTrainedModel`` plumbing.
* No KV-cache, no incremental decoding, no fast-path ``causal_conv1d`` import.
* No YaRN / dynamic RoPE scaling. Plain Gemma-style RoPE.
* No attention-dropout. No bias terms (``bias=False`` everywhere).

The block layout mirrors the LFM hybrid:

* ``ShortConv`` ("Gated-Conv") blocks contain ``in_proj(D -> 3D) -> split(B, C, x)
  -> B*x -> depthwise causal Conv1d(K=3) -> C*conv_out -> out_proj``.
* ``Attention`` blocks are GQA with per-head RMSNorm on Q and K (QK-Norm).
* The per-layer choice is driven by ``LFMConfig.layer_types[layer_idx]``
  (``"conv"`` or ``"attention"``). Default is the spec's 3:1 conv:attention
  pattern with the first two layers anchored as conv.

The ``LFM`` ``forward`` returns ``(logits, loss)`` to match the modded-nanogpt
trainer conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Config
# =============================================================================


def default_layer_types(num_layers: int) -> list[str]:
    """Spec-compliant LFM hybrid pattern.

    Layers 0-1 are anchored as ``"conv"`` (stable lower gradients). From layer 2
    on, every 4th layer is ``"attention"``: 3 conv then 1 attention. For the
    default 12-layer baseline this yields::

        conv, conv, conv, conv, conv, attn, conv, conv, conv, attn, conv, conv

    Note this is the R00 baseline pattern -- later records may retune which
    layers carry attention vs. conv.
    """
    types: list[str] = []
    for i in range(num_layers):
        if i < 2:
            types.append("conv")
        else:
            types.append("attention" if (i - 2) % 4 == 3 else "conv")
    return types


@dataclass
class LFMConfig:
    """R00 baseline: ~135M parameters (close to GPT-2-small/124M)."""

    # Sized to be near the modded-nanogpt 124M baseline. The actual count comes
    # out to ~135M because LFM adds the SwiGLU third matrix and ShortConv's
    # in_proj triples the hidden dim. Counted by the smoke test.
    vocab_size: int = 50304  # GPT-2 tokens padded to multiple of 128
    hidden_size: int = 768
    intermediate_size: int = 2048  # SwiGLU FFN: ~2.67x hidden
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4   # GQA: 12 queries share 4 K/V heads
    head_dim: int | None = None    # default: hidden_size // num_attention_heads
    conv_kernel_size: int = 3      # K=3 short conv (spec allows K=3 or 5)
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = default_layer_types(self.num_hidden_layers)
        assert len(self.layer_types) == self.num_hidden_layers, (
            f"layer_types length {len(self.layer_types)} != num_hidden_layers {self.num_hidden_layers}"
        )
        for t in self.layer_types:
            assert t in ("conv", "attention"), f"unknown layer type: {t}"
        if self.head_dim is None:
            assert self.hidden_size % self.num_attention_heads == 0
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be a multiple of num_key_value_heads (GQA grouping)"
        )


# =============================================================================
# Primitives: RMSNorm + RoPE
# =============================================================================


class RMSNorm(nn.Module):
    """Identical to ``LlamaRMSNorm`` and HF's ``Lfm2RMSNorm``.

    Computation is performed in fp32 to match the spec's stability guardrail.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.variance_epsilon)
        return (x32 * self.weight).to(in_dtype)


class Rotary(nn.Module):
    """Standard sin/cos rotary embeddings, computed once and cached.

    Matches Gemma2's geometry (which LFM2 inherits): the first ``head_dim/2``
    channels carry cos, the second half carries sin, applied as the classic
    ``(x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)`` rotation.
    """

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len: int = 0
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cached_cos is None
            or seq_len > self._cached_seq_len
            or self._cached_cos.device != device
            or self._cached_cos.dtype != dtype
        ):
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cached_cos = emb.cos().to(dtype)
            self._cached_sin = emb.sin().to(dtype)
            self._cached_seq_len = seq_len
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dim into the first half (and vice-versa)."""
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos / sin: [seq, head_dim]. q, k: [B, n_heads, seq, head_dim].
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# =============================================================================
# Sub-layers
# =============================================================================


class Attention(nn.Module):
    """LFM2-style multi-head attention with GQA + per-head QK-Norm.

    Mirrors HF ``Lfm2Attention`` (q/k/v/out_proj naming and shapes), but uses
    ``F.scaled_dot_product_attention`` directly so torch.compile can fuse it
    with FlashAttention-3 on Blackwell.
    """

    def __init__(self, config: LFMConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = self.q_layernorm(q).transpose(1, 2)  # [B, n_heads, T, head_dim]
        k = self.k_layernorm(k).transpose(1, 2)  # [B, n_kv,    T, head_dim]
        v = v.transpose(1, 2)

        q, k = apply_rotary_emb(q, k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
            scale=self.scaling,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.out_proj(attn)


class ShortConv(nn.Module):
    """LFM2 ``Lfm2ShortConv``: gated depthwise causal Conv1d block.

    Operates on ``[B, T, D]`` by:
      1. ``in_proj`` → ``[B, T, 3D]`` and transpose to ``[B, 3D, T]``.
      2. Chunk into ``B`` (gate-left), ``C`` (gate-right) and ``x`` (signal).
      3. ``Bx = B * x`` (multiplicative gating, no SiLU -- matches HF Lfm2).
      4. Depthwise causal Conv1d with ``padding=K-1``; trim trailing padding.
      5. ``y = C * conv_out`` (second gating).
      6. ``out_proj`` back to ``[B, T, D]``.

    Notes for future records:
      * Adding SiLU on either gate is a one-line change and may help once we
        also fuse with MXFP4 (see ``src/kernels.py``).
      * ``reset_flags`` / ``cu_seqlens`` support is a Kernel-1 concern; the
        eager ``nn.Conv1d`` here does not honour document boundaries.
    """

    def __init__(self, config: LFMConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.kernel_size = config.conv_kernel_size
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.kernel_size,
            groups=config.hidden_size,
            bias=False,
            padding=self.kernel_size - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        BCx = self.in_proj(x).transpose(-1, -2)         # [B, 3D, T]
        gate_left, gate_right, signal = BCx.chunk(3, dim=-2)
        gated = gate_left * signal                       # [B, D, T]
        conv_out = self.conv(gated)[..., :T]             # causal trim
        y = gate_right * conv_out
        y = y.transpose(-1, -2).contiguous()             # [B, T, D]
        return self.out_proj(y)


class MLP(nn.Module):
    """SwiGLU MLP exactly as in HF ``Lfm2MLP`` / Llama."""

    def __init__(self, config: LFMConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DecoderLayer(nn.Module):
    """One hybrid block: norm → (attn or conv), then norm → MLP, both residual.

    Matches HF ``Lfm2DecoderLayer`` minus dropout/cache plumbing.
    """

    def __init__(self, config: LFMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention = config.layer_types[layer_idx] == "attention"
        if self.is_attention:
            self.attn = Attention(config)
        else:
            self.conv = ShortConv(config)
        self.feed_forward = MLP(config)
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = self.operator_norm(x)
        if self.is_attention:
            h = self.attn(h, cos, sin)
        else:
            h = self.conv(h)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# =============================================================================
# Top-level model
# =============================================================================


class LFM(nn.Module):
    """Liquid Foundation Model (training-only).

    forward returns ``(logits, loss)``:
      * ``targets`` is None → return logits at the last position only (for
        cheap eager evaluation; trainer never uses this path).
      * ``targets`` is given → return full logits and the scalar CE loss.
        Trainer typically calls ``return_logits=False`` and discards logits.
    """

    def __init__(self, config: LFMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary = Rotary(config.head_dim, base=config.rope_theta)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def num_parameters(self, only_trainable: bool = True) -> int:
        params = (p for p in self.parameters() if (p.requires_grad or not only_trainable))
        return sum(p.numel() for p in params)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        B, T = idx.shape
        x = self.embed_tokens(idx)
        cos, sin = self.rotary(T, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.embedding_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, -1:, :])
            loss = None

        if not return_logits:
            logits = None
        return logits, loss
