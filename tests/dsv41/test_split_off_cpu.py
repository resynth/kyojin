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


def test_split_zero_is_endcap_only(split_modules, monkeypatch):
    """N=0 is split-off by default; the explicit split_zero flag makes it an endcap/MTP-only split."""
    dm, bridge, split = split_modules
    monkeypatch.delenv("KYOJIN_PART0_LAYERS", raising=False)
    monkeypatch.delenv("KYOJIN_SPLIT_ZERO", raising=False)
    assert split.SplitSettings(part0_layers=0).enabled is False
    assert split.SplitSettings(part0_layers=0, split_zero=True).enabled is True
    assert split.SplitSettings(part0_layers=0, split_zero=True, endcaps=False).enabled is False
    assert split.SplitSettings(part0_layers=1).enabled is True
    assert split.SplitSettings(part0_layers=0, split_zero=True, preset="ds41").enabled is True
    monkeypatch.setenv("KYOJIN_SPLIT_ZERO", "1")
    s = split.settings_from_env()
    assert s.split_zero is True and s.enabled is True


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


def test_bridge_env_parse_falls_back(split_modules, monkeypatch, capsys):
    dm, bridge, split = split_modules
    monkeypatch.setenv("KYOJIN_BRIDGE_CHUNK_MB", "abc")
    b = bridge.Bridge(part1=torch.device("cuda", 0), part0=torch.device("cuda", 1))
    assert b.max_chunk_bytes == 256 * (1 << 20)                # documented default, load instead of crash
    assert "could not parse" in capsys.readouterr().out
    monkeypatch.setenv("KYOJIN_BRIDGE_CHUNK_MB", "1.5")
    assert bridge.Bridge(part1=torch.device("cuda", 0), part0=torch.device("cuda", 1)).max_chunk_bytes \
        == int(1.5 * (1 << 20))                                # a plain MiB value is honored


def test_settings_from_env_guarded_float(split_modules, monkeypatch, capsys):
    dm, bridge, split = split_modules
    monkeypatch.setenv("KYOJIN_PART0_RESERVE_GB", "nonsense")
    s = split.settings_from_env()
    assert s.part0_reserve_gb == 1.2                           # DEFAULT_PART0_RESERVE_GB, warned above
    assert "could not parse" in capsys.readouterr().out


def test_shared_env_builder_matches_namespace(split_modules, monkeypatch):
    """settings_from_env and the env fallbacks of settings_from_namespace must stay one contract."""
    dm, bridge, split = split_modules
    monkeypatch.setenv("KYOJIN_PART0_LAYERS", "2")
    monkeypatch.setenv("KYOJIN_PART0_DEVICE", "cuda:1")
    monkeypatch.setenv("KYOJIN_PART1_DEVICE", "cuda:0")
    monkeypatch.setenv("KYOJIN_ENDCAPS", "0")
    monkeypatch.setenv("KYOJIN_SPLIT_PRESET", "DS41")
    s_env = split.settings_from_env()

    class _Args:                                               # the CLI with every flag unset
        pass

    s_ns = split.settings_from_namespace(_Args())
    assert vars(s_env) == vars(s_ns)
    assert s_env.part0_layers == 2 and s_env.part0_device == "cuda:1" and s_env.part1_device == "cuda:0"
    assert s_env.endcaps is False and s_env.preset == "ds41"   # preset normalized the same way

    _Args.part0_layers = "3"                                   # an explicit N wins, even with env set
    s_ns2 = split.settings_from_namespace(_Args)
    assert s_ns2.part0_layers == 3 and vars(s_ns2)["part0_device"] == "cuda:1"


def test_env_snapshot_restore_roundtrip(split_modules, monkeypatch):
    """A split-off load must see exactly the pre-split env (and device_copy.FORCED), so harness A/B in one
    process keeps the single-device contract."""
    dm, bridge, split = split_modules
    sys.path.insert(0, str(ROOT / "tools" / "ds41"))
    try:
        import ds41_model
    except Exception as ex:                              # noqa: BLE001 - loader module optional deps
        pytest.skip(f"ds41_model import needs its optional deps ({ex!r})")
    monkeypatch.delenv("KYOJIN_SPLIT_ACTIVE", raising=False)
    monkeypatch.setenv("KYOJIN_PART0_LAYERS", "0")
    ds41_model._restore_split_env()                            # nothing to undo: a no-op
    ds41_model._snapshot_split_env()
    before = {k: os.environ.get(k) for k in ds41_model._split_forced_keys()}
    import exllamav3.util.device_copy as dc
    before["dc.FORCED"] = dc.FORCED
    # a split-on run forces keys and flips the device_copy flag
    os.environ["KYOJIN_SPLIT_ACTIVE"] = "1"
    os.environ["DSV41_DECODE_GRAPH"] = "0"
    os.environ["EXL3_BLOCK_GRAPH"] = "0"
    dc.FORCED = True
    ds41_model._restore_split_env()
    assert os.environ.get("KYOJIN_SPLIT_ACTIVE") == before["KYOJIN_SPLIT_ACTIVE"]
    assert dc.FORCED == before["dc.FORCED"]
    assert ds41_model._SPLIT_ENV_DONE is False                 # a later split-on load applies fresh
    # ...and re-arming the marker state is coherent: snapshot again, mutate, restore
    ds41_model._snapshot_split_env()
    os.environ["KYOJIN_SPLIT_ACTIVE"] = "1"
    ds41_model._restore_split_env()
    assert "KYOJIN_SPLIT_ACTIVE" not in os.environ
    sys.path.remove(str(ROOT / "tools" / "ds41"))


def test_kernel_state_registry(split_modules):
    """Per-device DS4.1 JIT disable registry (the split's mixed-arch fallback): the decision is per device,
    so the 9070 can run torch paths while the iGPU keeps its kernels."""
    ks = _load_module("exllamav3.modules.dsv41_fn.kernel_state",
                      ROOT / "exllamav3" / "modules" / "dsv41_fn" / "kernel_state.py")
    ks.clear()
    assert not ks.disabled("cuda:0", "ds41_pf") and not ks.disabled(torch.device("cuda", 1), "dsv41_hip")
    ks.disable("cuda:1", "ds41_pf")
    assert ks.disabled("cuda:1", "ds41_pf") and ks.disabled(torch.device("cuda", 1), "ds41_pf")
    assert not ks.disabled("cuda:1", "dsv41_hip")        # module-local: the decode chain still runs
    assert not ks.disabled("cuda:0", "ds41_pf")          # device-local: the other card keeps its kernels
    ks.disable("cuda:1", "dsv41_idx")
    assert ks.disabled("cuda:1", "dsv41_idx") and ks.disabled("cuda:1", "ds41_pf")
    with pytest.raises(AssertionError):
        ks.disable("cuda:1", "not_a_ds41_module")
    ks.clear()
    assert not ks.disabled("cuda:1", "ds41_pf")
    with pytest.raises(AssertionError):
        ks.disabled("cuda:1", "nope")
    sys.modules.pop("exllamav3.modules.dsv41_fn.kernel_state", None)
