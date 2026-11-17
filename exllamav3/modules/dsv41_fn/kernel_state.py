"""Per-device, per-module availability of the DS4.1 JIT kernel modules
(dsv41_hip / ds41_pf / dsv41_idx / dsv41_fuse / dsv41_moe).

The JIT sources target gfx1151 (WMMA 256b + lane layouts). On another architecture a module either fails
to compile (LLVM: "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32 needs target feature wmma-256b-insts") or
must be assumed unsafe. A multi-arch box (mixed RDNA3.5 + RDNA4, i.e. the layer split with the 9070 as
part0) loads one process on several devices, so the decision cannot be a process-wide flag — and it is
per *module*, not per device: on gfx1201 the decoder chain (dsv41_hip), the fused glue (dsv41_fuse) and
the JIT MoE (dsv41_moe) build and run, while the WMMA prompt attention (ds41_pf) and the WMMA indexer
scoring (dsv41_idx) do not. Each module is probed once at model load
(`serve_ops.probe_ds41_kernels`), and every kernel entry point consults this registry for the tensor's
device and module name and takes its fallback when disabled.

The registry is a set of (device, module) pairs; it is intentionally small and lock-free (populated
during load, read-only afterwards). `clear()` exists for tests/harnesses that reload in one process.
"""
from __future__ import annotations

import torch

MODULES = ("dsv41_hip", "ds41_pf", "dsv41_idx", "dsv41_fuse", "dsv41_moe")

_OFF: set[tuple[str, str]] = set()


def disable(device, module: str) -> None:
    """Mark one JIT module as unusable on a device (probe failure / explicit request)."""
    assert module in MODULES, module
    _OFF.add((str(torch.device(device)), module))


def disabled(device, module: str) -> bool:
    """True when `module` must not run on `device` (its caller takes the torch/ext fallback)."""
    assert module in MODULES, module
    return (str(torch.device(device)), module) in _OFF


def clear() -> None:
    _OFF.clear()
