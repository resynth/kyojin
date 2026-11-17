"""
DSV4 contiguous layer split placement (split roadmap Stage 4).

DeepSeek-V4.1 is not a common-`Model` adapter: `dsv41.pack.load` builds the serving model itself
(`DSV41Serving`), so the split does not go through `SplitPlan`/`ComponentMap`. What it reuses from the
universal split core is `resolve_devices` (device selection), the `SplitSettings` CLI/env contract and the
`Bridge`. This module adds the DSV4 part: which layers land where, and where a boundary is even possible.

DSV4 layers read other layers' compressed-KV / indexer-key pools and the candidate-block selection of an
earlier layer, all as ordinary tensors written and read inside one forward. A boundary is therefore only
safe when every producer a part1 layer needs is itself on part1:

  * pool of kv source s     is used by layers [s, next(kv source) - 1];
  * pool of index source s  is used by layers [s, next(index source) - 1];
  * candidate blocks from layer cs are consumed by every compressed layer above cs.

A usage span that starts on one side of the boundary and ends on the other would need cross-device KV /
index state, which this stage does not implement. With the default config (`kv_source_layer_ids=[2,8,14,20]`,
`index_source_layer_ids=[2,8,14,20,24,28,32,36]`, `candidate_source_layer_id=20`) that leaves
`safe_split_boundaries == [0, 1, 2, 8, 14, 20, 40]`; N = 1 or 2 is the supported target (9070 as part0).

Placement additionally fixes the module/device rules the loader and the serving model follow:

  * trunk layer l        `part0` when l < N else `part1` (the pack's home device, `model.device`);
  * embed / final norm / head ("endcaps") and the MTP / DSpark stages share ONE device: `part0` when
    endcaps are kept, else `part1` — `DSparkHeads` reads `m.head` / `m.embed_w`, so MTP can never leave
    embed/head behind (decision D4);
  * the SSD expert tier lives on exactly one device (`part1` by default; `KYOJIN_DS41_TIER_DEVICE`
    overrides for experiments). Front blocks never register with the store: their experts load resident;
  * vision defaults to `part1` (decision D8; text-only is the Stage 4 gate).

Everything here is pure placement arithmetic plus torch device objects (no kernels), so it is unit-testable
without GPUs by injecting a synthetic `DeviceInfo` inventory into `resolve_devices` — the same pattern as
`tests/test_split_plan_cpu.py`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch

from .. import device_map as _dm
from ..serve_split import DEFAULT_PART0_RESERVE_GB, SplitSettings

DSV41_TIER_DEVICE_ENV = "KYOJIN_DS41_TIER_DEVICE"
DSV41_VISION_DEVICE_ENV = "KYOJIN_DS41_VISION_DEVICE"


def _usage_spans(dcfg) -> list[tuple[int, int, str]]:
    """Contiguous (first, last, kind) index ranges of the layers that touch each shared pool."""
    spans: list[tuple[int, int, str]] = []
    for kind, sources in (("kv", dcfg.kv_source_layer_ids), ("idx", dcfg.index_source_layer_ids)):
        ordered = sorted(s for s in sources if s < dcfg.num_hidden_layers)
        for i, s in enumerate(ordered):
            nxt = ordered[i + 1] if i + 1 < len(ordered) else dcfg.num_hidden_layers
            spans.append((s, nxt - 1, kind))
    return spans


def safe_split_boundaries(dcfg) -> list[int]:
    """Layer counts N (front size) that keep every cross-layer producer on one side of the boundary.

    A pool used by `[s, e]` straddles the boundary iff s < N <= e (the producer s stays on part0 while a
    reader reaches part1). The candidate selection produced by layer `candidate_source_layer_id` is consumed
    by every compressed layer above it, so any N with cs < N <= such a consumer is unsafe. Always includes
    0 (nothing on part0) and the full layer count (everything on part0, no part1 readers left).
    """
    n = dcfg.num_hidden_layers
    spans = _usage_spans(dcfg)
    cs = dcfg.candidate_source_layer_id
    out = []
    for N in range(n + 1):
        ok = True
        for s, e, _kind in spans:
            if s < N <= e:
                ok = False
                break
        if ok and cs is not None and 0 <= cs < n and cs < N and \
                any(l > cs and dcfg.ratio(l) > 0 for l in range(N, n)):
            ok = False                    # part1 consumers would need the candidate blocks of part0
        if ok:
            out.append(N)
    return out


def validate(n, endcaps: bool, dcfg, devices: _dm.Devices = None) -> int:
    """Reject what this stage does not support. Returns the manual N. `n` may be 'auto' (rejected: Stage 5
    owns auto-N); a non-safe N is rejected with the allowed values; `devices` only re-checks part0 != part1."""
    if isinstance(n, str):
        if n == "auto":
            raise RuntimeError("DSV4 layer split needs a manual N for now (--part0-layers N); "
                               "auto-N is a later stage. Safe boundaries for this config: "
                               f"{safe_split_boundaries(dcfg)}")
        raise RuntimeError(f"DSV4 layer split wants an integer N, got {n!r}")
    N = int(n)
    if N < 0:
        raise RuntimeError(f"DSV4 layer split wants N >= 0, got {N}")
    if devices is not None and torch.device(devices.part0) == torch.device(devices.part1):
        raise RuntimeError("DSV4 layer split part0 and part1 are the same device")
    allowed = safe_split_boundaries(dcfg)
    if N not in allowed:
        raise RuntimeError(f"DSV4 layer split: part0_layers={N} would read a compressed-KV / candidate pool "
                           f"across the boundary. Allowed boundaries for this config: {allowed}")
    # D4 is a construction rule (endcaps <-> MTP share one device), not a per-N check; the front budget is
    # enforced at load, not here (no auto-N).
    return N


def tier_allows(config, device) -> bool:
    """The SSD expert tier guard (decision D3), shared by `DSV41Block._tier_register`.

    With no split boundary set (`cfg._expert_tier_device` unset) every device registers as before; with the
    split on, only the tier device does. Front blocks never register — their experts load resident, and the
    store never sees an SSD read from part0. Split off is untouched (the attribute simply does not exist)."""
    tier = getattr(config, "_expert_tier_device", None)
    return tier is None or torch.device(device) == torch.device(tier)


@dataclass
class DSV41Placement:
    """Where every DSV4 module lives, and the rules the loader / serving model follow.

    `cfg` is the effective (possibly layer-subset) `DSV41Config`, not the framework Config. The framework
    Config instead receives `cfg_framework._expert_tier_device` and `cfg_framework._dsv41_placement` in
    `pack.load` so `_tier_register` and any other framework-side code can see the placement.
    """

    devices: _dm.Devices
    cfg: object                                  # effective DSV41Config
    part0_layers: int = 0
    endcaps: bool = True
    tier_device: torch.device | str = "cuda:1"
    vision_device: torch.device | str = "cuda:1"
    part0_reserve_gb: float = DEFAULT_PART0_RESERVE_GB
    bridge: object = None                        # exllamav3.bridge.Bridge, installed by the loader

    # -- device rules ------------------------------------------------------
    @property
    def part1(self) -> torch.device:
        return torch.device(self.devices.part1)

    @property
    def part0(self) -> torch.device:
        return torch.device(self.devices.part0)

    @property
    def endcap_device(self) -> torch.device:
        """embed / final norm / head (and, with them, MTP): part0 when endcaps are kept, else the home device."""
        return self.part0 if self.endcaps else self.part1

    @property
    def mtp_device(self) -> torch.device:
        """MTP / DSpark stages share the endcap device (DSparkHeads reads m.head / m.embed_w; decision D4)."""
        return self.endcap_device

    @property
    def draft_device(self) -> torch.device:
        return self.mtp_device

    @property
    def tier(self) -> torch.device:
        return torch.device(self.tier_device)

    def text_device(self, l: int) -> torch.device:
        """Device of trunk layer l, or of MTP stage l for l >= num_hidden_layers."""
        cfg = self.cfg
        if l >= cfg.num_hidden_layers:
            return self.mtp_device
        return self.part0 if l < self.part0_layers else self.part1

    def front(self, l: int) -> bool:
        return l < self.part0_layers

    def devices_used(self) -> list[torch.device]:
        seen, out = set(), []
        for d in (self.text_device(l) for l in range(self.cfg.num_hidden_layers + self.cfg.n_mtp_layers)):
            d = torch.device(d)
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out

    # -- narration ---------------------------------------------------------
    def describe(self) -> str:
        cfg = self.cfg
        n, N = cfg.num_hidden_layers, self.part0_layers
        d0, d1 = self.part0, self.part1
        front = f"layers 0..{N - 1}" if N else "no layers"
        return (f"[split] DSV4: {front} on {d0} ({d0.type} {getattr(d0, 'index', None)}), "
                f"layers {N}..{n - 1} on {d1}; endcaps {'on' if self.endcaps else 'off'} -> "
                f"embed/norm/head + MTP on {self.endcap_device}; expert tier on {self.tier}; "
                f"bridge part1<->part0; part0 reserve {self.part0_reserve_gb:.2f} GiB")

    # -- budget estimate ---------------------------------------------------
    def front_layers(self) -> list[int]:
        return list(range(min(self.part0_layers, self.cfg.num_hidden_layers)))


def estimate_front_bytes(placement: DSV41Placement, numel_of: Callable[[str], int],
                         bits: float = 2.9, bits_for: Callable[[str], float] = None) -> dict:
    """Startup budget estimate of what the part0 device must hold, in bytes.

    `numel_of(module_key)` gives the pre-load number of weights of a framework module tree (use the loaded
    `Model`'s per-module `weights_numel()`); `bits` / `bits_for(key)` fold the pack's bpw in. This is a
    printed estimate to sanity-check the 9070 budget before the load, not an auto-N solver (Stage 5 owns
    auto-N). Returns per-group bytes plus the reserve and the total.
    """
    cfg = placement.cfg

    def _bytes(key: str) -> int:
        nb = int(numel_of(key))
        bpw = float(bits_for(key)) if bits_for is not None else bits
        return int(nb * bpw / 8.0)

    layers = [f"layers.{l}" for l in placement.front_layers()]
    layer_bytes = sum(_bytes(k) for k in layers)
    endcap_keys = ["embed", "norm", "head", "hc_expand"] if placement.endcaps else []
    endcap_bytes = sum(_bytes(k) for k in endcap_keys)
    mtp_keys = [f"mtp.{j}" for j in range(cfg.n_mtp_layers)] + (["mtp.extra"] if cfg.n_mtp_layers else [])
    mtp_bytes = sum(_bytes(k) for k in mtp_keys) if placement.endcaps else 0
    total = layer_bytes + endcap_bytes + mtp_bytes
    return {
        "layer_bytes": layer_bytes,
        "endcap_bytes": endcap_bytes,
        "mtp_bytes": mtp_bytes,
        "reserve_bytes": int(placement.part0_reserve_gb * (1 << 30)),
        "total_bytes": total,
        "total_gib": total / (1 << 30),
        "layers": placement.front_layers(),
    }


def resolve_placement(settings: SplitSettings, dcfg, *, devices: _dm.Devices = None,
                      tier_device=None, tier_explicit: bool = None, vision_device=None,
                      logger=print) -> DSV41Placement:
    """`SplitSettings` + effective config -> `DSV41Placement` (or None when the split is off).

    `devices` injects a resolved `device_map.Devices` for tests; production callers omit it and
    `resolve_devices` probes the runtime. `tier_device` overrides `KYOJIN_DS41_TIER_DEVICE`; passing
    either marks the choice as explicit (part0 tiering is accepted only then).
    """
    if not settings.enabled:
        return None
    if settings.auto:
        # D5: manual N only in this stage. reject before anything else so an --part0-layers auto run
        # fails with the boundary list instead of a fit chase.
        validate("auto", settings.endcaps, dcfg)
    N = validate(settings.part0_layers, settings.endcaps, dcfg, devices)
    if devices is None:
        devices = _dm.resolve_devices(settings.part0_device, settings.part1_device)
    env_tier = os.environ.get(DSV41_TIER_DEVICE_ENV, "").strip().lower()
    pick = tier_device or env_tier or ""
    explicit = tier_explicit if tier_explicit is not None else bool(tier_device or env_tier)
    if pick in ("", "part1", "auto"):
        tier = torch.device(devices.part1)
    elif pick in ("part0",):
        tier = torch.device(devices.part0)
    else:
        tier = torch.device(pick)
    if tier == torch.device(devices.part0) and not explicit:
        raise RuntimeError(f"expert tier on part0 needs an explicit request "
                           f"({DSV41_TIER_DEVICE_ENV}=part0); default is part1")
    if tier not in (torch.device(devices.part0), torch.device(devices.part1)):
        raise RuntimeError(f"expert tier device {tier} is neither part0 ({devices.part0}) "
                           f"nor part1 ({devices.part1})")
    env_vis = os.environ.get(DSV41_VISION_DEVICE_ENV, "").strip().lower() if vision_device is None else ""
    pick_vis = (vision_device or "") or env_vis
    if pick_vis in ("", "part1", "auto"):
        vision = torch.device(devices.part1)
    elif pick_vis in ("part0",):
        # D8: part0 vision only on request (budget is checked when the tower actually loads)
        vision = torch.device(devices.part0)
    else:
        vision = torch.device(pick_vis)
    placement = DSV41Placement(
        devices=devices,
        cfg=dcfg,
        part0_layers=N,
        endcaps=bool(settings.endcaps),
        tier_device=tier,
        vision_device=vision,
        part0_reserve_gb=float(settings.part0_reserve_gb),
    )
    logger(placement.describe())
    return placement
