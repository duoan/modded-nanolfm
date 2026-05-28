"""Download cached, pre-tokenised FineWeb10B shards from HuggingFace.

This script is a direct port of modded-nanogpt's ``data/cached_fineweb10B.py``
so the on-disk binary format is interchangeable. The shards are tokenised with
the GPT-2 BPE tokenizer (vocab 50257; we pad to 50304 inside the model).

Each shard is ~100M tokens stored as ``uint16`` after a 256 × int32 header
(magic ``20240520``, version ``1``, token count). The header is what
``train_lfm.py``'s data loader validates.

Usage::

    # Smoke (~1 GB) -- just the val shard plus 1 train shard.
    uv run python data/cached_fineweb10B.py 1

    # Recommended R00 dataset (~10 GB) -- enough for a single epoch at the
    # baseline token budget.
    uv run python data/cached_fineweb10B.py 9

    # Full 10B-token sweep (~100 GB).
    uv run python data/cached_fineweb10B.py 103
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "kjj0/fineweb10B-gpt2"  # same repo modded-nanogpt uses
NUM_TRAIN_CHUNKS_FULL = 103       # 103 chunks × 100M tokens = full FineWeb-10B sample


def _get(fname: str) -> None:
    local_dir = os.path.join(os.path.dirname(__file__), "fineweb10B")
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type="dataset", local_dir=local_dir)


def main() -> None:
    num_chunks = NUM_TRAIN_CHUNKS_FULL
    if len(sys.argv) >= 2:
        num_chunks = int(sys.argv[1])
    _get(f"fineweb_val_{0:06d}.bin")
    for i in range(1, num_chunks + 1):
        _get(f"fineweb_train_{i:06d}.bin")
    print(f"downloaded validation shard + {num_chunks} train shard(s)")


if __name__ == "__main__":
    main()
