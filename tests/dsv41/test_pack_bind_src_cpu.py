"""CPU regression: the pack loader's module dispatch (`pack._bind_src`).

`mtp.extra` is a DSV41Raw sidecar (main_norm / markov / confidence heads + main_proj), not a block: it must
take `add_extra`, never `add_block` (which needs `.l`). Caught by the split MTP load on the box.
"""
from __future__ import annotations

from exllamav3.dsv41.pack import _bind_src


class _Src:
    def __init__(self):
        self.calls = []

    def add_block(self, mod):
        self.calls.append(("block", mod.key))

    def add_extra(self, mod, main_proj):
        self.calls.append(("extra", mod.key, main_proj))


class _Mod:
    def __init__(self, key):
        self.key = key


class _MTP:
    main_proj = object()


def test_mtp_extra_routes_to_add_extra():
    src = _Src()
    assert _bind_src(src, _Mod("mtp.extra"), _MTP()) == "mtp.extra"
    assert src.calls == [("extra", "mtp.extra", _MTP.main_proj)]


def test_mtp_and_trunk_blocks_route_to_add_block():
    src = _Src()
    for key in ("mtp.0", "mtp.2", "layers.7"):
        assert _bind_src(src, _Mod(key), _MTP()) == key
    assert src.calls == [("block", "mtp.0"), ("block", "mtp.2"), ("block", "layers.7")]
