"""modded-nanolfm: speedrun an LFM-style hybrid backbone on a single Blackwell node.

This package follows the *modded-nanogpt* methodology: one self-contained trainer
lives at the repo root (``train_lfm.py``) and every accepted record is frozen as
an immutable snapshot under ``records/R<NN>_<NAME>/``. Optimisations such as
Muon, fused Triton kernels, MXFP4 quantisation, FP8 LM head, distillation, etc.
arrive one record at a time -- each justified by an end-to-end wall-clock
improvement against the previous record on the same hardware.

Package layout
--------------
``src.model``      LFM2-style hybrid backbone (RMSNorm + RoPE + GQA/QK-Norm +
                   ShortConv + SwiGLU). Mirrors the HuggingFace ``Lfm2``
                   reference but rewritten as a single, compile-friendly file
                   so it can evolve alongside the trainer.
``src.optimizer``  ``build_optimizers(model, ...)`` returns a *list* of optimisers
                   so the trainer never needs to know how parameters are grouped.
                   In R00 this returns a single AdamW; later records add Muon
                   (and friends) without touching the trainer's stepping loop.
``src.kernels``    Stub. Triton kernels live here as they get introduced.

Parameter-naming protocol (informational for now)
-------------------------------------------------
Future records will route parameters through different optimisers based on
substring matches inside ``model.named_parameters()``. The reserved tokens are::

    weight_projection  -- 2D linear projections (Q/K/V/O, FFN, LM head).
                          Eventual Muon stream.
    weight_conv        -- 1D depthwise convolution kernels in ShortConv blocks.
                          Must bypass Muon (preserves spatial inductive bias).

R00 does not depend on these tokens; the AdamW classifier in ``src.optimizer``
falls back to ``p.ndim``-based heuristics and treats embeddings, biases, and
norms as no-weight-decay parameters.
"""

from . import kernels, model, optimizer

__all__ = ["kernels", "model", "optimizer"]
