"""Split-off regression + bridge bounded-hop CPU tests.

Split off (part0_layers = 0 / no env keys) must be the original single-device path: no device resolution,
no env forcing, no bridge, no split marker in the environment (roadmap Stage 4, "split-off regression").
The bridge part pins the bounded `hop_chunk` staging budget and its slice plan (the plan math is pure;
a real two-GPU hop is a box test).
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
def split_modules():
    added = []
    if "exllamav3" not in sys.modules:
        pkg = types.ModuleType("exllamav3")
        pkg.__path__ = [str(ROOT / "exllamav3")]
        sys.modules["exllamav3"] = pkg
        added.append("exllamav3")
    if "exllamav3.util" not in sys.modules:
        s = importlib.util.spec_from_file_location("exllamav3.util", ROOT / "exllamav3" / "util" / "__init__.py")
        util = importlib.util.module_from_spec(s)
        sys.modules["exllamav3.util"] = util
        s.loader.exec_module(util)
        added.append("exllamav3.util")
    if "exllamav3.util.arch_list" not in sys.modules:
        _load_module("exllamav3.util.arch_list", ROOT / "exllamav3" / "util" / "arch_list.py")
        added.append("exllamav3.util.arch_list")
    dm = _load_module("exllamav3.device_map", ROOT / "exllamav3" / "device_map.py")
    bridge = _load_module("exllamav3.bridge", ROOT / "exllamav3" / "bridge.py")
    split = _load_module("exllamav3.serve_split", ROOT / "exllamav3" / "serve_split.py")
    yield dm, bridge, split
    for name in ("exllamav3.serve_split", "exllamav3.bridge", "exllamav3.device_map"):
        sys.modules.pop(name, None)
    for name in added:
        sys.modules.pop(name, None)


class _Cfg:
    pass


# ---------------------------------------------------------------- split off

def test_settings_off_paths(split_modules, monkeypatch):
    dm, bridge, split = split_modules
    monkeypatch.delenv("KYOJIN_PART0_LAYERS", raising=False)
    s = split.settings_from_env()
    assert s.part0_layers == 0 and s.enabled is False
    assert split.apply_split_env(s) == []                # no env writes when off
    assert split.parse_part0_layers("0") == 0 and split.parse_part0_layers("off") == 0
    assert split.parse_part0_layers("2") == 2


def test_apply_split_env_ds41_preset(split_modules, monkeypatch):
    dm, bridge, split = split_modules
    assert split.SPLIT_PRESETS["ds41"] == {"DSV41_DECODE_GRAPH": "0"}   # D6
    s = split.SplitSettings(part0_layers=1, preset="ds41")
    before = dict(os.environ)
    try:
        split.apply_split_env(s, logger=lambda *_: None)
        assert os.environ["DSV41_DECODE_GRAPH"] == "0"
        for k in (
            "EXL3_PF_HC_FUSE", "EXL3_PF_GR_FUSE", "EXL3_GR_PF_FUSE",
            "EXL3_BLOCK_GRAPH", "EXL3_CUDA_GRAPH", "EXL3_HIP_GRAPH",
        ):
            assert os.environ[k] == "0"
        # the forcing only ever turns keys off (never turns a feature on)
        for k, v in os.environ.items():
            if before.get(k) != v:
                assert v == "0"
    finally:
        os.environ.clear()
        os.environ.update(before)


def test_install_bridge_off_leaves_no_trace(split_modules, monkeypatch):
    dm, bridge, split = split_modules
    monkeypatch.delenv("KYOJIN_SPLIT_ACTIVE", raising=False)
    cfg = _Cfg()
    assert bridge.install_bridge(cfg, part1=torch.device("cuda", 0), part0=None) is None
    assert cfg.split_active is False
    assert "KYOJIN_SPLIT_ACTIVE" not in os.environ and not hasattr(cfg, "device_bridge")
    # same device on both sides: no split either
    cfg = _Cfg()
    assert bridge.install_bridge(cfg, part1=torch.device("cuda", 0), part0=torch.device("cuda", 0)) is None
    assert cfg.split_active is False and "KYOJIN_SPLIT_ACTIVE" not in os.environ
    assert bridge.install_bridge(cfg, part1=torch.device("cuda", 0), part0=None) is None \
        or getattr(cfg, "device_bridge", None) is None
    # bridge explicitly disabled by env: no bridge object, split marker set, forced host bounce
    monkeypatch.setenv("KYOJIN_BRIDGE", "0")
    cfg = _Cfg()
    b = bridge.install_bridge(cfg, part1=torch.device("cuda", 0), part0=torch.device("cuda", 1))
    assert b is None and cfg.split_active is True
    monkeypatch.delenv("KYOJIN_BRIDGE", raising=False)


def test_settings_dict_roundtrip_is_off_safe(monkeypatch):
    """`vars(SplitSettings)` (the serve.py -> loader payload) round-trips through _split_settings and an
    off payload stays off. Imports the loader shim, which is stdlib-light by design."""
    sys.path.insert(0, str(ROOT / "tools" / "ds41"))
    try:
        import ds41_model
    except Exception as ex:                              # noqa: BLE001 - heavy optional deps of the loader module
        pytest.skip(f"ds41_model import needs its optional deps ({ex!r})")
    monkeypatch.delenv("KYOJIN_PART0_LAYERS", raising=False)
    from exllamav3.serve_split import SplitSettings
    off = dict(vars(SplitSettings(part0_layers=0)))
    assert ds41_model._split_settings(off).enabled is False
    assert ds41_model._split_settings(None) is None
    on = dict(vars(SplitSettings(part0_layers=1, part0_device="cuda:1", part1_device="cuda:0")))
    s = ds41_model._split_settings(on)
    assert s.enabled is True and s.part0_layers == 1
    sys.path.remove(str(ROOT / "tools" / "ds41"))


# ---------------------------------------------------------------- bridge bounded hop

def test_slice_plan_bounds_and_cover(split_modules):
    dm, bridge, split = split_modules
    B = bridge.Bridge
    # unbounded (0) -> one slice
    assert B.slice_plan(1_000_000, 4, 0) == [(0, 1_000_000)]
    # under budget -> one slice
    assert B.slice_plan(1000, 4, 4096) == [(0, 1000)]
    plan = B.slice_plan(1_000_000, 4, 4096)
    assert plan[0][0] == 0 and plan[-1][1] == 1_000_000
    for (a0, b0), (a1, b1) in zip(plan, plan[1:]):
        assert b0 == a1
    step = plan[0][1] - plan[0][0]
    assert step * 4 <= 4096                                    # every slice within the staging budget
    assert all((b - a) == step for a, b in plan[:-1])           # near-equal: no short tail except the last
    assert (plan[-1][1] - plan[-1][0]) > 0


def test_bridge_default_budget_from_env(split_modules, monkeypatch):
    dm, bridge, split = split_modules
    monkeypatch.setenv("KYOJIN_BRIDGE_CHUNK_MB", "16")
    b = bridge.Bridge(part1=torch.device("cuda", 0), part0=torch.device("cuda", 1))
    assert b.max_chunk_bytes == 16 * (1 << 20)
    monkeypatch.delenv("KYOJIN_BRIDGE_CHUNK_MB")
    b = bridge.Bridge(part1=torch.device("cuda", 0), part0=torch.device("cuda", 1))
    assert b.max_chunk_bytes == 256 * (1 << 20)                # the plan's bounded-hop default


def test_hop_same_device_noop(split_modules):
    dm, bridge, split = split_modules
    b = bridge.Bridge(part1=torch.device("cuda", 0), part0=torch.device("cuda", 1))
    t = torch.arange(8, dtype=torch.float32)
    assert b.hop(t, torch.device("cpu")) is t                  # same device: returned as-is
    assert b.hop_chunk(t, torch.device("cpu")) is t
    assert b.stats["chunk_hops"] == 0 and b.stats["hops"] == 0
    assert t.device.type == "cpu"
