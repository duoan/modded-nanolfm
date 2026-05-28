"""Custom Triton kernels (stubbed for R00).

This module is intentionally empty in the R00 baseline. As speedrun records
land, the trajectory looks roughly like this:

================  =====================================  ==================================
Record (planned)  Kernel                                  Replaces
================  =====================================  ==================================
R??               ``fused_softcap_cross_entropy``        ``F.cross_entropy`` on huge logits
R??               ``fused_lm_head_cross_entropy``        ``lm_head`` + softcap + CE
R??               ``fused_short_conv``                   eager ``nn.Conv1d`` in ShortConv
R??               ``fused_short_conv_with_resets``       above + ``cu_seqlens`` boundaries
R??               ``fused_short_conv_mxfp4``             above + RHT + on-chip MXFP4 (G=32)
================  =====================================  ==================================

Until a kernel lands, attempting to import ``triton`` would fail loudly on
machines without it. We keep the dependency optional by only importing inside
the functions that actually need it.
"""

__all__: list[str] = []
