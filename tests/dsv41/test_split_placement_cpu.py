"""CPU placement tests of the DSV4 contiguous layer split (dsv41.placement).

Same recipe as tests/test_split_plan_cpu.py: a fake `exllamav3` package (real files, no package __init__)
plus an injected synthetic device inventory, so no GPU and no built extension are needed. The boundary
derivation, the validation, the device rules (endcaps / MTP / tier / vision) and the front-budget estimate
are pure placement arithmetic and torch device objects.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def placement_modules():
    added = []
    if "exllamav3" not in sys.modules:
        pkg = types.ModuleType("exllamav3")
        pkg.__path__ = [str(ROOT / "exllamav3")]
        sys.modules["exllamav3"] = pkg
        added.append("exllamav3")
    if "exllamav3.util" not in sys.modules:
        util = types.ModuleType("exllamav3.util")
        util.__path__ = [str(ROOT / "exllamav3" / "util")]
        sys.modules["exllamav3.util"] = util
        added.append("exllamav3.util")
        s = importlib.util.spec_from_file_location("exllamav3.util", ROOT / "exllamav3" / "util" / "__init__.py")
        m = importlib.util.module_from_spec(s)
        sys.modules["exllamav3.util"] = m
        s.loader.exec_module(m)
    if "exllamav3.util.arch_list" not in sys.modules:
        _load_module("exllamav3.util.arch_list", ROOT / "exllamav3" / "util" / "arch_list.py")
        added.append("exllamav3.util.arch_list")
    dm = _load_module("exllamav3.device_map", ROOT / "exllamav3" / "device_map.py")
    ss = _load_module("exllamav3.serve_split", ROOT / "exllamav3" / "serve_split.py")
    from exllamav3.dsv41 import placement as P          # through the fake package, real files
    from exllamav3.dsv41.config import DSV41Config, tiny_config
    yield dm, ss, P, DSV41Config, tiny_config
    for name in ("exllamav3.dsv41.placement", "exllamav3.serve_split", "exllamav3.device_map"):
        sys.modules.pop(name, None)
    for name in added:
        sys.modules.pop(name, None)


def _devices(dm, archs=("gfx1201", "gfx1151"), index0=0, index1=1):
    infos = [dm.DeviceInfo(index=i, name=f"device-{i}", arch=a, uuid=f"uu{i}")
             for i, a in zip((index0, index1), archs)]
    return dm.Devices(
        part1=torch.device("cuda", infos[0].index if index0 == 1 else infos[1].index),
        part0=torch.device("cuda", infos[0].index if index0 == 0 else infos[1].index),
        part1_arch=infos[0 if index0 == 1 else 1].arch,
        part0_arch=infos[0 if index0 == 0 else 1].arch,
        inventory=tuple(infos),
    )


def _settings(ss, N=1, endcaps=True, part0="gfx1201", part1="gfx1151"):
    return ss.SplitSettings(part0_layers=N, part0_device=part0, part1_device=part1, endcaps=endcaps)


# ---------------------------------------------------------------- boundaries

def test_safe_boundaries_default(placement_modules):
    dm, ss, P, DSV41Config, tiny_config = placement_modules
    assert P.safe_split_boundaries(DSV41Config()) == [0, 1, 2, 8, 14, 20, 40]


def test_safe_boundaries_synthetic_generic(placement_modules):
    dm, ss, P, DSV41Config, tiny_config = placement_modules
    # tiny: kv/idx sources (2,4) and (2,4,5), candidate source 4 -> [0,1,2,4,8]; a different
    # layout must not carry the default numbers over.
    assert P.safe_split_boundaries(tiny_config()) == [0, 1, 2, 4, 8]


def test_safe_boundaries_candidate_straddle(placement_modules):
    dm, ss, P, DSV41Config, tiny_config = placement_modules
    cfg = tiny_config(candidate_source_layer_id=2)      # consumers at layers 5, 6 (ratio 1)
    b = P.safe_split_boundaries(cfg)
    assert 3 not in b and 4 not in b and 5 not in b and 6 not in b
    assert 2 in b and 8 in b                            # N=2 keeps producer + every reader on part1
    # a boundary between the candidate source and any of its consumers needs cross-device candidate
    # blocks; the pool rule already forbids the rest (pool 4 spans [4, 7]).
    assert P.safe_split_boundaries(tiny_config(candidate_source_layer_id=6)) == [0, 1, 2, 4, 8]
    # a candidate source BELOW a pool-safe boundary still cross-device-reads: cs=3 sits on part0 when N=4
    # and layers 5, 6 (its consumers) sit on part1, so only [0,1,2,8] survive.
    assert P.safe_split_boundaries(tiny_config(candidate_source_layer_id=3)) == [0, 1, 2, 8]


def test_validate_rejects_auto_and_unsafe(placement_modules):
    dm, ss, P, DSV41Config, tiny_config = placement_modules
    cfg = tiny_config()
    with pytest.raises(RuntimeError, match="manual N"):
        P.validate("auto", True, cfg)
    for bad in (3, 5, 6, 7, 9, 38):
        with pytest.raises(RuntimeError, match="Allowed boundaries"):
            P.validate(bad, True, cfg)
    for good in (0, 1, 2, 4, 8):
        assert P.validate(good, True, cfg) == good
    cfg_full = DSV41Config()
    with pytest.raises(RuntimeError):
        P.validate(21, True, cfg_full)
    assert P.validate(20, True, cfg_full) == 20


def test_validate_rejects_same_device(placement_modules):
    dm, ss, P, DSV41Config, tiny_config = placement_modules
    cfg = tiny_config()
    devs = dm.Devices(part1=torch.device("cuda", 0), part0=torch.device("cuda", 0),
                      part1_arch="", part0_arch="")  # resolve_devices refuses this; validate re-checks
    with pytest.raises(RuntimeError, match="same device"):
        P.validate(1, True, cfg, devices=devs)


# ---------------------------------------------------------------- resolve / device rules

def test_resolve_endcaps_and_mtp_share_device(placement_modules, monkeypatch):
    dm, ss, P, DSV41Config, tiny_config = placement_modules
    cfg = tiny_config()
    monkeypatch.delenv(P.DSV41_TIER_DEVICE_ENV, raising=False)
    monkeypatch.delenv(P.DSV41_VISION_DEVICE_ENV, raising=False)
    devs = _devices(dm)
    pl = P.resolve_placement(_settings(ss, N=1, endcaps=True), cfg, devices=devs)
    assert pl.part0_layers == 1
    assert pl.text_device(0) == torch.device("cuda", pl.part0.index)        # front layer on part0
    assert pl.text_device(1) == pl.part1                                    # rest on part1
    assert pl.endcap_device == pl.part0                                     # D4: endcaps -> part0
    assert pl.mtp_device == pl.endcap_device == pl.draft_device
    assert pl.tier == pl.part1                                              # D3 default
    assert pl.vision_device == pl.part1                                     # D8 default
    assert pl.front(0) and not pl.front(1)

    pl_off = P.resolve_placement(_settings(ss, N=1, endcaps=False), cfg, devices=devs)
    assert pl_off.endcap_device == pl_off.part1                             # endcaps off: everything on part1
    assert pl_off.mtp_device == pl_off.part1


def test_resolve_rejects_non_pair_devices(placement_modules, monkeypatch):
    dm, ss, P, DSV41Config, tiny_config = placement_modules
    cfg = tiny_config()
    monkeypatch.delenv(P.DSV41_TIER_DEVICE_ENV, raising=False)
    with pytest.raises(RuntimeError):                       # N not in the safe set
        P.resolve_placement(_settings(ss, N=3), cfg, devices=_devices(dm))
    with pytest.raises(RuntimeError, match="neither part0"):  # tier device neither side
        P.resolve_placement(_settings(ss, N=1), cfg, devices=_devices(dm),
                            tier_device="cuda:9", tier_explicit=True)


def test_tier_part0_only_explicit(placement_modules, monkeypatch):
    dm, ss, P, DSV41Config, tiny_config = placement_modules
    cfg = tiny_config()
    devs = _devices(dm)
    with pytest.raises(RuntimeError, match="explicit"):
        P.resolve_placement(_settings(ss, N=1), cfg, devices=devs, tier_device="part0", tier_explicit=False)
    pl = P.resolve_placement(_settings(ss, N=1), cfg, devices=devs, tier_device="part0", tier_explicit=True)
    assert pl.tier == pl.part0                              # an explicit request is accepted (D3)
    monkeypatch.setenv(P.DSV41_TIER_DEVICE_ENV, "part0")
    pl = P.resolve_placement(_settings(ss, N=1), cfg, devices=devs)
    assert pl.tier == pl.part0                              # the env override counts as explicit
    monkeypatch.delenv(P.DSV41_TIER_DEVICE_ENV, raising=False)


def test_auto_rejected_for_dsv41(placement_modules):
    dm, ss, P, DSV41Config, tiny_config = placement_modules
    with pytest.raises(RuntimeError, match="manual N"):     # D5: manual N only in this stage
        P.resolve_placement(_settings(ss, N="auto"), tiny_config(), devices=_devices(dm))


def test_tier_allows_guard(placement_modules):
    dm, ss, P, DSV41Config, tiny_config = placement_modules
    cfg = tiny_config()
    assert P.tier_allows(cfg, "cuda:0") is True             # no boundary set: the original behavior
    cfg._expert_tier_device = "cuda:1"
    assert P.tier_allows(cfg, "cuda:1") is True
    assert P.tier_allows(cfg, torch.device("cuda", 1)) is True
    assert P.tier_allows(cfg, "cuda:0") is False            # front device never registers (D3)


# ---------------------------------------------------------------- budget estimate

def test_estimate_front_bytes_ordering(placement_modules):
    dm, ss, P, DSV41Config, tiny_config = placement_modules
    cfg = tiny_config()
    numel = {"embed": 512 * 64, "norm": 64, "head": 64 * 512, "hc_expand": 0}
    numel.update({f"layers.{l}": 64 * 64 for l in range(cfg.num_hidden_layers)})
    numel.update({f"mtp.{j}": 64 * 64 for j in range(cfg.n_mtp_layers)})
    numel["mtp.extra"] = 64

    def sizes(pl):
        return P.estimate_front_bytes(pl, lambda k: numel.get(k, 0), bits=2.9)

    devs = _devices(dm)
    n1 = sizes(P.resolve_placement(_settings(ss, N=1, endcaps=True), cfg, devices=devs))
    n2 = sizes(P.resolve_placement(_settings(ss, N=2, endcaps=True), cfg, devices=devs))
    n1e = sizes(P.resolve_placement(_settings(ss, N=1, endcaps=False), cfg, devices=devs))
    assert n1["total_bytes"] < n2["total_bytes"]            # more front layers, more bytes; no silent auto
    assert n1e["endcap_bytes"] + n1e["mtp_bytes"] == 0      # endcaps off: no endcap/MTP term
    assert n1["endcap_bytes"] > 0 and n1["mtp_bytes"] > 0
    assert n1["reserve_bytes"] == int(1.2 * (1 << 30))
    assert abs(n1["total_gib"] - n1["total_bytes"] / (1 << 30)) < 1e-9


def test_markov_head_device_rule_note(placement_modules):
    # D4 is enforced by construction (mtp_device == endcap_device); this pins it down.
    dm, ss, P, DSV41Config, tiny_config = placement_modules
    cfg = tiny_config()
    devs = _devices(dm)
    for endcaps in (True, False):
        pl = P.resolve_placement(_settings(ss, N=1, endcaps=endcaps), cfg, devices=devs)
        assert pl.mtp_device == pl.endcap_device
