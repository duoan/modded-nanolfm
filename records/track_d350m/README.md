# Track D-350M (LFM2-350M)

**Architecture**: LFM2-350M dense (16 layers = 10 conv + 6 attn, `d=1024`,
FF=4608, 16Q / 8KV / `head_dim=64`, GQA group=2, conv `k=3`, tied embeddings).
See [`src/configs.py::lfm2_350m`](../../src/configs.py).

Vocabulary: GPT-2 BPE padded to 50 304 (FineWeb-GPT2 tokens).
Param count with this vocab: **~339 M** (official 354M is with vocab 65 536).

| Item | Value |
|------|-------|
| Hardware (canonical timing) | 8 × H100 |
| Hardware (cloud dev) | 8 × B200, 1 × H100, 1 × B200 |
| Hardware (local dev) | 1 × RTX PRO 6000 Blackwell |
| Data | FineWeb-GPT2, train shards from `KellerJordan/fineweb10B-gpt2` |
| Target val CE on FineWeb | **TBD** (set by the R00 baseline) |
| Token budget | **TBD** (set by R00) |
| Launch | `TRACK=d350m ./run.sh` (local) or `scripts/launch_modal.sh h100x8 d350m_R00` |

## Records

| #   | Time | Description | Date | Log | Contributor |
| --- | ---  | ---         | ---  | --- | ---         |
| R00 | —    | AdamW baseline (LFM2-350M shape + the optimizer / schedule frozen by [the R00 trainer snapshot](../../records/R00_baseline_lfm2/)) | pending | — | — |

## Notes

- R00 for this track has not been run yet. Once the first baseline lands, it
  defines the target val CE for all subsequent submissions on this track.
- The conv-vs-attention layer positions are exactly those of
  `LiquidAI/LFM2-1.2B/config.json::full_attn_idxs = [2, 5, 8, 10, 12, 14]`;
  LFM2 350/700/1.2B share the same 16-layer backbone.
