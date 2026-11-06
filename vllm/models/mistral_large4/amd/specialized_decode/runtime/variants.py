# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer-type mono kernel variants (hosted in aiter)."""

from aiter.ops.flydsl.mono.variants import KernelVariant, load_variant

__all__ = ["KernelVariant", "load_variant"]
