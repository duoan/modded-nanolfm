"""Canonical ``LFMConfig`` presets for each official LFM2 release size.

Source of truth
---------------
- Dense sizes (350M / 700M / 1.2B / 2.6B): `LFM2 Technical Report
  <https://arxiv.org/abs/2511.23404>`__ Table 1, cross-checked against the HF
  ``LiquidAI/LFM2-{350M,700M,1.2B,2.6B}`` ``config.json`` files.
- MoE size (8B-A1B): HF ``LiquidAI/LFM2-8B-A1B/config.json``.

Vocab choice
------------
The official LFM2 release uses a custom **65 536-token BPE** trained on
Liquid AI's corpus. Our speedrun trains on **FineWeb pre-tokenised with the
GPT-2 BPE** (vocab 50 257, padded to 50 304 for mat-mul alignment), to keep
timing comparable with ``modded-nanogpt`` at the same parameter shape.

This means our LFM2-350M (with GPT-2 vocab) has ~335M params instead of the
official 354M -- the architectural shape is identical; the embedding tables
are smaller because of the smaller vocabulary. Pass ``vocab_size=65_536`` to
any factory if you want to match the official release exactly.

Token budgets
-------------
Targets and token budgets per track are documented in
``records/track_*/README.md``; this file only owns the *architecture* presets.

Usage
-----
::

    from src.configs import lfm2_350m, TRACK_CONFIGS

    cfg = lfm2_350m()                       # GPT-2 vocab (50 304)
    cfg = lfm2_350m(vocab_size=65_536)      # official LFM2 vocab

    cfg = TRACK_CONFIGS["d350m"]()          # same as above

In ``train_lfm.py``, setting ``TRACK=d350m`` in the environment picks up the
preset automatically.
"""

from __future__ import annotations

from .model import LFMConfig

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# GPT-2 BPE padded up to a multiple of 128 for tensor-core friendly mat-muls.
# This matches modded-nanogpt's FineWeb tokenisation.
GPT2_VOCAB = 50_304

# Official LFM2 uses 1e6 (long-context friendly). Worth preserving even on
# our short-context speedruns -- larger theta is harmless when seq_len < 8K.
LFM2_ROPE_THETA = 1_000_000.0


# ---------------------------------------------------------------------------
# Layer-type patterns (copied from official HF configs)
# ---------------------------------------------------------------------------
# 0-indexed attention block positions; every other layer is ``"conv"``.
# Source: HF ``config.json`` for each release.

# All three of LFM2-350M / 700M / 1.2B share the same 16-layer backbone.
# Source: ``LiquidAI/LFM2-1.2B`` ``full_attn_idxs`` field.
_ATTN_IDXS_16L: frozenset[int] = frozenset({2, 5, 8, 10, 12, 14})

# LFM2-2.6B (30 layers, 8 attn). From ``LiquidAI/LFM2-2.6B`` ``layer_types``.
_ATTN_IDXS_2_6B: frozenset[int] = frozenset({2, 5, 9, 13, 17, 21, 24, 27})

# LFM2-8B-A1B MoE (24 layers, 6 attn). From ``LiquidAI/LFM2-8B-A1B`` ``layer_types``.
_ATTN_IDXS_MOE: frozenset[int] = frozenset({2, 6, 10, 14, 18, 21})


def _layer_types(num_layers: int, attn_idxs: frozenset[int]) -> list[str]:
    return ["attention" if i in attn_idxs else "conv" for i in range(num_layers)]


# ---------------------------------------------------------------------------
# Dense presets
# ---------------------------------------------------------------------------


def lfm2_350m(vocab_size: int = GPT2_VOCAB) -> LFMConfig:
    """LFM2-350M: 16 layers (10 conv + 6 attn), d=1024, FF=4608, 16Q / 8KV / h=64.

    Official: 354M params with vocab 65 536. With GPT-2 vocab (50 304): ~335M.
    """
    return LFMConfig(
        vocab_size=vocab_size,
        hidden_size=1024,
        intermediate_size=4608,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=8,
        rope_theta=LFM2_ROPE_THETA,
        layer_types=_layer_types(16, _ATTN_IDXS_16L),
    )


def lfm2_700m(vocab_size: int = GPT2_VOCAB) -> LFMConfig:
    """LFM2-700M: 16 layers (10 conv + 6 attn), d=1536, FF=6912, 24Q / 8KV / h=64.

    Official: 742M params. With GPT-2 vocab: ~720M.
    """
    return LFMConfig(
        vocab_size=vocab_size,
        hidden_size=1536,
        intermediate_size=6912,
        num_hidden_layers=16,
        num_attention_heads=24,
        num_key_value_heads=8,
        rope_theta=LFM2_ROPE_THETA,
        layer_types=_layer_types(16, _ATTN_IDXS_16L),
    )


def lfm2_1_2b(vocab_size: int = GPT2_VOCAB) -> LFMConfig:
    """LFM2-1.2B: 16 layers (10 conv + 6 attn), d=2048, FF=8192, 32Q / 8KV / h=64.

    Official: 1.17B params. With GPT-2 vocab: ~1.14B.
    """
    return LFMConfig(
        vocab_size=vocab_size,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        rope_theta=LFM2_ROPE_THETA,
        layer_types=_layer_types(16, _ATTN_IDXS_16L),
    )


def lfm2_2_6b(vocab_size: int = GPT2_VOCAB) -> LFMConfig:
    """LFM2-2.6B: 30 layers (22 conv + 8 attn), d=2048, FF=10752, 32Q / 8KV / h=64.

    Official: 2.57B params. With GPT-2 vocab: ~2.54B.
    """
    return LFMConfig(
        vocab_size=vocab_size,
        hidden_size=2048,
        intermediate_size=10752,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=8,
        rope_theta=LFM2_ROPE_THETA,
        layer_types=_layer_types(30, _ATTN_IDXS_2_6B),
    )


# ---------------------------------------------------------------------------
# MoE preset (architectural spec only -- MoE FFN not yet implemented in model.py)
# ---------------------------------------------------------------------------
# We expose the full MoE config as a plain dict so the MoE track's R00 has a
# precise spec to build against. When we wire MoE into model.py (likely an
# ``LFMMoeConfig`` extending ``LFMConfig`` with ``num_experts`` /
# ``num_experts_per_tok`` / ``moe_intermediate_size`` / ``num_dense_layers``),
# this dict becomes the canonical constructor input.

LFM2_8B_A1B_SPEC: dict = dict(
    # Shared backbone (same fields as LFMConfig).
    vocab_size=GPT2_VOCAB,                 # official: 65_536
    hidden_size=2048,
    intermediate_size=7168,                # dense-FFN size used by the first ``num_dense_layers`` layers
    num_hidden_layers=24,
    num_attention_heads=32,
    num_key_value_heads=8,
    rope_theta=LFM2_ROPE_THETA,
    layer_types=_layer_types(24, _ATTN_IDXS_MOE),
    # MoE-specific extensions (NOT YET supported by ``src/model.py``).
    num_experts=32,
    num_experts_per_tok=4,                 # top-k routing
    moe_intermediate_size=1792,            # per-expert FFN size
    num_dense_layers=2,                    # first N FFNs are dense, rest are MoE
    norm_topk_prob=True,
    use_expert_bias=True,
    routed_scaling_factor=1.0,
)


# ---------------------------------------------------------------------------
# Track registry
# ---------------------------------------------------------------------------
# Public name → zero-arg factory returning the canonical ``LFMConfig`` for
# that track. Used by ``train_lfm.py`` to wire the ``TRACK=<name>`` env var.

TRACK_CONFIGS: dict[str, callable] = {
    "d350m": lfm2_350m,
    "d700m": lfm2_700m,
    "d1_2b": lfm2_1_2b,
    "d2_6b": lfm2_2_6b,
    # "moe_8b" intentionally omitted: ``model.py`` doesn't implement MoE yet.
    # See ``LFM2_8B_A1B_SPEC`` for the target spec.
}


def get_track_config(name: str, vocab_size: int = GPT2_VOCAB) -> LFMConfig:
    """Resolve a track name to its canonical ``LFMConfig``.

    Raises ``KeyError`` with a helpful list of valid names if ``name`` is
    unknown.
    """
    if name not in TRACK_CONFIGS:
        valid = ", ".join(sorted(TRACK_CONFIGS))
        raise KeyError(f"unknown track {name!r}; valid tracks: {valid}")
    return TRACK_CONFIGS[name](vocab_size=vocab_size)
