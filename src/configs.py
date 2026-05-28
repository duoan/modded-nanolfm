"""Architecture presets for the LFM speedrun.

This repo runs **2 tracks** at the same scale and data budget as
``modded-nanogpt``'s GPT-2-small AdamW baseline (124M params, ~5B
FineWeb-GPT2 tokens, 8 x H100):

- ``dense`` -- LFM-hybrid backbone (conv + GQA) with a vanilla
  SwiGLU FFN. Direct apples-to-apples with modded-nanogpt's
  baseline; the only thing that changes is the model architecture.
- ``moe``  -- same backbone, but the FFN in every layer is replaced
  with a top-2 routed Mixture-of-Experts. Tests what a sparse FFN
  buys at roughly fixed active-compute. **Not yet implemented in**
  ``src/model.py``; ``MOE_SMALL_SPEC`` below freezes the target.

Vocab
-----
We use **GPT-2 BPE** (50 257, padded to 50 304 for tensor-core mat-muls)
to keep tokenisation identical to ``modded-nanogpt``. The official
LFM2 release uses a custom 65 536-token BPE -- ignore for the speedrun.

Token budgets
-------------
Documented in ``records/track_*/README.md``. This file only owns the
*architecture* presets.

Usage
-----
::

    from src.configs import get_track_config

    cfg = get_track_config("dense")           # == LFMConfig() default

In ``train_lfm.py``, ``TRACK=dense`` (or unset ``TRACK=``) selects the
dense preset; ``TRACK=moe`` will be wired up once MoE lands in
``src/model.py``.
"""

from __future__ import annotations

from collections.abc import Callable

from .model import LFMConfig

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# GPT-2 BPE padded up to a multiple of 128 for tensor-core friendly mat-muls.
# This matches modded-nanogpt's FineWeb tokenisation.
GPT2_VOCAB = 50_304

# Official LFM2 uses 1e6 (long-context friendly). Harmless on short seq_len.
LFM2_ROPE_THETA = 1_000_000.0


# ---------------------------------------------------------------------------
# Dense preset (Track Dense)
# ---------------------------------------------------------------------------


def dense_baseline(vocab_size: int = GPT2_VOCAB) -> LFMConfig:
    """LFM-hybrid at modded-nanogpt scale: 12 L, d=768, FF=2048, 12Q / 4KV.

    Matches the original ``LFMConfig()`` defaults (~122M params); this is
    what ``TRACK=dense`` and unset ``TRACK=`` both resolve to. Apples-to-
    apples with modded-nanogpt's 124M GPT-2-small AdamW baseline.
    """
    return LFMConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        conv_kernel_size=3,
        rope_theta=LFM2_ROPE_THETA,
    )


# ---------------------------------------------------------------------------
# MoE preset (Track MoE -- architectural spec only, FFN not implemented yet)
# ---------------------------------------------------------------------------
# Exposed as a plain dict so the future ``LFMMoeConfig`` (an extension of
# ``LFMConfig`` with ``num_experts`` / ``num_experts_per_tok`` /
# ``moe_intermediate_size``) has a canonical constructor input.
#
# Design rationale (see ``records/track_moe/README.md`` for the full story):
# - Same backbone as Track Dense (12 L, d=768, GQA 12Q/4KV) so attention +
#   conv compute per token is unchanged.
# - 8 experts top-2: per-token active FF compute = 2 * (2 * 768 * 1024) =
#   3.1M params, the same as Dense's single FFN (2 * 768 * 2048 / 2 = 3.1M
#   effective with SwiGLU). So wall-clock per step should match Dense.
# - Total parameter count grows to ~235M (122M backbone-equivalent active +
#   ~110M extra expert capacity).
# - Aux load-balancing loss weight to be picked at R00 time (Switch-style 1e-2
#   is a sensible default).

MOE_SMALL_SPEC: dict = dict(
    # Backbone fields (same as ``dense_baseline``).
    vocab_size=GPT2_VOCAB,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    conv_kernel_size=3,
    rope_theta=LFM2_ROPE_THETA,
    # MoE FFN -- NOT YET supported by ``src/model.py``.
    num_experts=8,
    num_experts_per_tok=2,             # top-k routing
    moe_intermediate_size=1024,        # per-expert FFN size
    num_dense_layers=0,                # all 12 layers use MoE FFN
    norm_topk_prob=True,               # renormalise selected expert weights
    router_aux_loss_coef=1e-2,         # load-balancing aux loss (Switch-Transformer style)
)


# ---------------------------------------------------------------------------
# Track registry
# ---------------------------------------------------------------------------
# Public name → zero-arg factory returning the canonical ``LFMConfig``.
# Used by ``train_lfm.py`` to wire the ``TRACK=<name>`` env var.

def _moe_not_implemented(**_: object) -> LFMConfig:
    raise NotImplementedError(
        "Track MoE is spec-only: src/model.py implements the dense backbone\n"
        "but not the MoE FFN + router yet. See:\n"
        "  - src/configs.py::MOE_SMALL_SPEC  (architectural target)\n"
        "  - records/track_moe/README.md     (full track spec + token budget)"
    )


TRACK_CONFIGS: dict[str, Callable[..., LFMConfig]] = {
    "dense": dense_baseline,
    "moe": _moe_not_implemented,
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
