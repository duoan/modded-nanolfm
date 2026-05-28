# Track D-2.6B (LFM2-2.6B)

**Architecture**: LFM2-2.6B dense (30 layers = 22 conv + 8 attn, `d=2048`,
FF=10752, 32Q / 8KV / `head_dim=64`, GQA group=4, conv `k=3`, tied embeddings).
See [`src/configs.py::lfm2_2_6b`](../../src/configs.py).

Vocabulary: GPT-2 BPE padded to 50 304.
Param count with this vocab: **~2.54 B** (official 2.57B is with vocab 65 536).

| Item | Value |
|------|-------|
| Hardware (canonical timing) | 8 × H100 |
| Hardware (cloud dev) | 8 × B200 |
| Data | FineWeb-GPT2 |
| Target val CE on FineWeb | **TBD** (set by R00) |
| Token budget | **TBD** (set by R00) |
| Launch | `TRACK=d2_6b ./run.sh` or `scripts/launch_modal.sh h100x8 d2_6b_R00` |

## Records

| #   | Time | Description | Date | Log | Contributor |
| --- | ---  | ---         | ---  | --- | ---         |
| R00 | —    | AdamW baseline at the LFM2-2.6B shape | pending | — | — |

## Notes

- This track uses a *different* attention pattern from D-350M/700M/1.2B.
  Source: `LiquidAI/LFM2-2.6B/config.json::layer_types`. The 30-layer stack
  has attention at indices `{2, 5, 9, 13, 17, 21, 24, 27}` (8 attn / 22 conv).
- For single-node training the model + optimizer state easily fits 8 × H100;
  on 8 × B200 there's headroom for context > 16K, useful for record entries
  that play with sequence-length warmup.
