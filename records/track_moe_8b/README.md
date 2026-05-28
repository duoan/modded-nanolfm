# Track MoE-8B-A1B (LFM2-8B-A1B)

**Architecture**: LFM2-8B-A1B MoE — 24 layers (18 conv + 6 attn) on a 2048-wide
backbone with a dense FFN (size 7168) on the first 2 layers and a 32-expert
top-4 MoE FFN (per-expert size 1792) on the remaining 22. Attention is GQA
32Q / 8KV / `head_dim=64`, conv `k=3`, tied embeddings.

Spec: [`src/configs.py::LFM2_8B_A1B_SPEC`](../../src/configs.py).
Source: `LiquidAI/LFM2-8B-A1B/config.json`.

Official total / active params: **8.3 B / 1.5 B**.

| Item | Value |
|------|-------|
| Hardware (canonical timing) | 8 × B200 |
| Hardware (cloud fallback) | 8 × H100 |
| Data | FineWeb-GPT2 (with router-balancing aux loss) |
| Target val CE on FineWeb | **TBD** (set by R00) |
| Token budget | **TBD** (set by R00) |
| Launch | not yet available — see "Status" below |

## Records

| #   | Time | Description | Date | Log | Contributor |
| --- | ---  | ---         | ---  | --- | ---         |
| —   | —    | track not yet active | — | — | — |

## Status

**Track inactive.** `src/model.py` currently implements the dense LFM2 backbone
only. Activating this track requires:

1. An `LFMMoeConfig` (e.g. extending `LFMConfig` with `num_experts`,
   `num_experts_per_tok`, `moe_intermediate_size`, `num_dense_layers`).
2. An MoE FFN module (top-k routing + expert FFNs + token-dispatch /
   token-combine + load-balancing aux loss). The simplest first cut can use
   `torch.scatter`-based dispatch before introducing a fused kernel.
3. A `TRACK=moe_8b` wiring in `train_lfm.py`.

Until then, the spec lives in `src/configs.py::LFM2_8B_A1B_SPEC` as the
single source of truth so the MoE record can be built against a frozen,
auditable target.

## Why B200 for the canonical timing

- MoE benefits disproportionately from the high-bandwidth memory + sparse
  compute path on Blackwell.
- 8 × H100 (80 GB total = 640 GB) can host this model + activations, but
  expert all-to-all gets bandwidth-limited; 8 × B200 (180 GB total = 1.4 TB +
  NVLink 5) is closer to the regime LFM2 was tuned for.
