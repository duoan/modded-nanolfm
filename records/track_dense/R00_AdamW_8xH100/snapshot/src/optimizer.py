"""Optimiser bifurcation interface.

R00 baseline: single :class:`torch.optim.AdamW` with proper decay / no-decay
parameter groups. ``build_optimizers`` returns a *list* of optimisers and a
matching list of LR functions so the trainer can iterate over them uniformly.
Later records swap pieces of this without the trainer having to know::

    R00  [AdamW(everything)]
    R01  [AdamW(embeddings/biases/norms),  Muon(2D weight_projection)]
    R02  [AdamW(...), Muon(...), AdamW(conv weights w/ no decay)]
    ...

Naming conventions consumed here
--------------------------------
The classifier is **dtype-and-shape based by default**, so the R00 model does
not have to opt-in to any naming scheme. The fall-through rules are:

* ``nn.Embedding.weight``           → no weight decay (AdamW).
* ``ndim < 2`` (biases, norm gains) → no weight decay (AdamW).
* ``ndim >= 2`` (linear/conv kernels) → weight decay (AdamW).

When R01 adds Muon, we'll extend ``_classify`` to recognise the reserved
``weight_projection`` / ``weight_conv`` substrings documented in
``src/__init__.py``. R00 already exposes that name-based hook (``muon_hint``)
so we can flip Muon on in one line later.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

# =============================================================================
# Parameter classification
# =============================================================================


@dataclass(frozen=True)
class _ParamGroup:
    name: str
    weight_decay: float


def _classify(name: str, param: nn.Parameter) -> _ParamGroup:
    """R00 classifier: dtype/shape-driven, name-aware for forward compatibility."""
    # 1D tensors are always biases, norms, or scalar gates -- no weight decay.
    if param.ndim < 2:
        return _ParamGroup(name="no_decay", weight_decay=0.0)

    # Embeddings stay out of weight decay (standard GPT-2 / LLaMA practice).
    # We detect them by the canonical ``embed_tokens.weight`` suffix used by
    # :mod:`src.model`. Embedding tying makes ``lm_head.weight`` an alias of
    # the same tensor; param-group dedup happens in ``build_optimizers``.
    if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
        return _ParamGroup(name="no_decay", weight_decay=0.0)

    # Every other 2D+ parameter (Q/K/V/O, w1/w2/w3, conv kernel, LM head when
    # untied) gets standard decoupled weight decay.
    return _ParamGroup(name="decay", weight_decay=1.0)  # multiplier; final wd applied in build


# =============================================================================
# Public API
# =============================================================================


def split_param_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict]:
    """Return AdamW-style param groups: ``[decay_group, no_decay_group]``.

    Embedding-tied weights are added exactly once (the first occurrence wins).
    """
    seen: set[int] = set()
    groups: dict[str, dict] = {
        "decay": {"params": [], "weight_decay": weight_decay, "name": "decay"},
        "no_decay": {"params": [], "weight_decay": 0.0, "name": "no_decay"},
    }
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        cls = _classify(n, p)
        groups[cls.name]["params"].append(p)
    return [g for g in groups.values() if g["params"]]


def build_optimizers(
    model: nn.Module,
    *,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-10,
    weight_decay: float = 0.1,
    fused: bool = True,
) -> tuple[list[torch.optim.Optimizer], list[Iterable[nn.Parameter]]]:
    """R00 build: one fused AdamW with decay/no-decay groups.

    Returns
    -------
    optimizers : list of optimisers (length 1 in R00; later records grow this).
    param_lists : matching list of underlying parameter iterables, useful for
                  per-optimiser gradient hooks (e.g. norm clipping).
    """
    groups = split_param_groups(model, weight_decay=weight_decay)
    adamw = torch.optim.AdamW(
        groups,
        lr=lr,
        betas=betas,
        eps=eps,
        fused=fused,
    )
    all_params = [p for g in groups for p in g["params"]]
    return [adamw], [all_params]


def summarize_param_groups(model: nn.Module, weight_decay: float = 0.1) -> str:
    """Human-readable param-group inventory; useful in trainer startup logs."""
    groups = split_param_groups(model, weight_decay=weight_decay)
    lines = ["param groups:"]
    total = 0
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        total += n
        lines.append(
            f"  - {g['name']:<10s}  count={len(g['params']):>3d}  numel={n / 1e6:>7.2f}M  wd={g['weight_decay']}"
        )
    lines.append(f"  total: {total / 1e6:.2f}M")
    return "\n".join(lines)
