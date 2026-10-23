#!/usr/bin/env python
"""Per-plane byte accounting for the GLM-5.3 paged cache, fp16 vs int8 (--cache-bits 8).

Builds the Cache objects (no weight load: the geometry comes from the attention config, so
storage_size() and the plane shapes are valid before Model.load) for the target and the MTP
draft component, in both cache modes, and prints bytes per token and per plane.

usage: python tools/glm/cache_bytes.py [-m ~/models/glm53-exl3-td205] [-c 40960]
"""
import argparse, os

import numpy as np
import torch

from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant


def gtt_gb():
    import glob
    best = 0
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"):
        try:
            best = max(best, int(open(p).read()))
        except Exception:
            pass
    return best / 2**30


def planes(layer):
    """name -> (shape, itemsize) for the cache planes the layer actually carries."""
    out = {}
    for name in ("shape_c", "shape_r", "shape_i", "shape_p", "qshape", "sshape"):
        shp = getattr(layer, name, None)
        if shp is None:
            continue
        dt = torch.int if name == "qshape" else torch.half   # qk is int32, everything else fp16
        out[name] = (tuple(shp), dt.itemsize)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="~/models/glm53-exl3-td205")
    ap.add_argument("-c", "--max-ctx", type=int, default=40960)
    a = ap.parse_args()
    cfg = Config.from_directory(os.path.expanduser(a.model))
    nt = a.max_ctx + 4096
    print(f"model={a.model} max_ctx={a.max_ctx} max_num_tokens={nt} page={256}")
    for comp in (None, "mtp"):
        for cb in (0, 8):
            kw = dict(layer_type=CacheLayer_quant, k_bits=cb, v_bits=cb) if cb else {}
            m = Model.from_config(cfg, component=comp) if comp else Model.from_config(cfg)
            c = Cache(m, max_num_tokens=nt, max_history=2, **kw)
            raw = sum(l.storage_size() for l in c.layers.values())
            per_plane = {}
            for l in c.layers.values():
                for name, (shp, isz) in planes(l).items():
                    per_plane[name] = per_plane.get(name, 0) + int(np.prod(shp)) * isz
            comp_name = comp or "target"
            print(f"  {comp_name:6} cache_bits={cb}: layers={len(c.layers)} raw={raw} "
                  f"B/token={raw / nt:.1f}")
            for name, b in sorted(per_plane.items()):
                print(f"      {name:8} {b / nt:8.2f} B/token   shape={planes(next(iter(c.layers.values())))[name][0]}")
            del c, m


if __name__ == "__main__":
    main()
