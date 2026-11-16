"""CPU tests of the DSV4 tier guard (decision D3) and its wiring.

The guard decision itself lives in `dsv41.placement.tier_allows` (import-light, no GPU, no built extension);
this file pins the contract and checks that `DSV41Block._tier_register` actually calls it (the block class
itself needs the built extension, so on a plain dev box only the wiring canary runs; the functional path
runs on the box).

Also exercised: the placement level — a front layer must never carry tier state after a split load, and
the store's device must be the tier device (the loader asserts this in pack.load; the rules are mirrored
here so a regression names the decision that broke).
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
def modules():
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
    _load_module("exllamav3.device_map", ROOT / "exllamav3" / "device_map.py")
    from exllamav3.dsv41 import placement as P
    yield P
    for name in ("exllamav3.dsv41.placement", "exllamav3.device_map"):
        sys.modules.pop(name, None)
    for name in added:
        sys.modules.pop(name, None)


def test_guard_semantics(modules):
    P = modules
    cfg = object()                                       # split off: no _expert_tier_device attribute at all
    assert P.tier_allows(cfg, None) is True              # device None is filtered later; the guard is device-agnostic
    assert P.tier_allows(cfg, "cuda:7") is True
    cfg = types.SimpleNamespace(_expert_tier_device="cuda:1")
    assert P.tier_allows(cfg, "cuda:1") is True
    assert P.tier_allows(cfg, torch.device("cuda", 1)) is True
    assert P.tier_allows(cfg, "cuda:0") is False
    cfg = types.SimpleNamespace(_expert_tier_device=torch.device("cuda", 0))
    assert P.tier_allows(cfg, "cuda:0") is True
    assert P.tier_allows(cfg, "cuda:1") is False


def test_tier_register_calls_guard(modules):
    """Canary: `_tier_register` must consult `tier_allows` before any store registration."""
    src = (ROOT / "exllamav3" / "modules" / "dsv41.py").read_text()
    assert "tier_allows" in src
    seg = src.split("def _tier_register", 1)[1]
    assert "tier_allows" in seg.split("def ", 1)[0]


class _FakeStore:
    """`ExpertStore` stand-in: records the devices that register, never touches a GPU."""

    def __init__(self):
        self.device = None
        self.registered = []

    def register(self, module, device, shells: bool = True):
        self.registered.append(torch.device(device))
        if self.device is None:
            self.device = torch.device(device)
        return object()                                  # the TierSet stand-in


def test_tier_register_respects_tier_device(modules, monkeypatch):
    """The real `_tier_register` against a stubbed store. Skips when the built extension (or numpy) is
    missing (dev box); the canary above carries the wiring contract there."""
    _t = types
    try:
        import exllamav3.modules.dsv41 as _M
        import exllamav3.modules.expert_store as _ES
    except Exception as ex:                              # noqa: BLE001 - dev box without the ext build
        pytest.skip(f"needs the built extension to import exllamav3.modules ({ex!r})")
    store = _FakeStore()
    monkeypatch.setattr(_ES, "get_store", lambda config: store, raising=False)
    monkeypatch.setenv("EXL3_EXPERT_BUDGET_GIB", "92")
    cfg = _t.SimpleNamespace(_expert_tier_device="cuda:1")
    fn = _M.DSV41Block._tier_register
    front = _t.SimpleNamespace(config=cfg, mtp=False, pruned=None)
    assert fn(front, "cuda:0") is None                   # front device: no tier, no registration
    assert store.registered == []
    back = _t.SimpleNamespace(config=cfg, mtp=False, pruned=None)
    ts = fn(back, "cuda:1")
    assert ts is not None                                # tier device registers
    assert store.device == torch.device("cuda", 1)       # the store captures the tier device

    cfg_off = _t.SimpleNamespace()                       # split off: the original behavior
    any_dev = _t.SimpleNamespace(config=cfg_off, mtp=False, pruned=None)
    assert fn(any_dev, "cuda:0") is not None
    monkeypatch.delenv("EXL3_EXPERT_BUDGET_GIB")
    assert fn(any_dev, "cuda:0") is None                 # store off entirely: still None
