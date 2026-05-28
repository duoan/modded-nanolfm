# Track D-1.2B (LFM2-1.2B)

**Architecture**: LFM2-1.2B dense (16 layers = 10 conv + 6 attn, `d=2048`,
FF=8192, 32Q / 8KV / `head_dim=64`, GQA group=4, conv `k=3`, tied embeddings).
See [`src/configs.py::lfm2_1_2b`](../../src/configs.py).

Vocabulary: GPT-2 BPE padded to 50 304.
Param count with this vocab: **~1.14 B** (official 1.17B is with vocab 65 536).

| Item | Value |
|------|-------|
| Hardware (canonical timing) | 8 × H100 |
| Hardware (cloud dev) | 8 × B200 |
| Data | FineWeb-GPT2 |
| Target val CE on FineWeb | **TBD** (set by R00) |
| Token budget | **TBD** (set by R00) |
| Launch | `TRACK=d1_2b ./run.sh` or `scripts/launch_modal.sh h100x8 d1_2b_R00` |

## Records

| #   | Time | Description | Date | Log | Contributor |
| --- | ---  | ---         | ---  | --- | ---         |
| R00 | —    | AdamW baseline at the LFM2-1.2B shape | pending | — | — |

## Notes

- This is the "headline" dense track: shares HF defaults with LFM2-1.2B, sits
  in the sweet spot for training-on-one-node experiments.
- Memory budget on 8 × H100 (80 GB): with `bf16` activations + `bf16` model,
  ~14 GB weights + activations leave plenty of room for context warmup.
