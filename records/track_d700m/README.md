# Track D-700M (LFM2-700M)

**Architecture**: LFM2-700M dense (16 layers = 10 conv + 6 attn, `d=1536`,
FF=6912, 24Q / 8KV / `head_dim=64`, GQA group=3, conv `k=3`, tied embeddings).
See [`src/configs.py::lfm2_700m`](../../src/configs.py).

Vocabulary: GPT-2 BPE padded to 50 304.
Param count with this vocab: **~719 M** (official 742M is with vocab 65 536).

| Item | Value |
|------|-------|
| Hardware (canonical timing) | 8 × H100 |
| Hardware (cloud dev) | 8 × B200 |
| Data | FineWeb-GPT2 |
| Target val CE on FineWeb | **TBD** (set by R00) |
| Token budget | **TBD** (set by R00) |
| Launch | `TRACK=d700m ./run.sh` or `scripts/launch_modal.sh h100x8 d700m_R00` |

## Records

| #   | Time | Description | Date | Log | Contributor |
| --- | ---  | ---         | ---  | --- | ---         |
| R00 | —    | AdamW baseline at the LFM2-700M shape | pending | — | — |

## Notes

- Same 16-layer attention pattern as D-350M / D-1.2B
  (`attn_idxs = {2, 5, 8, 10, 12, 14}`).
- The bump from 350M is mainly width (1024 → 1536) and head count (16 → 24);
  the FFN expansion ratio stays ~4.5× hidden.
