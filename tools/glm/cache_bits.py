"""Single source of truth for the quantised paged-cache request kwargs.

`tools/glm/serve.py` builds the MLA cache from these; the measurement harnesses
(`tools/strix_halo/bench_mtp.py`, `tools/glm/cache_bytes.py`) must build it the same way or
their numbers stop describing the server.
"""
from __future__ import annotations


def quant_cache_kwargs(cache_bits: int) -> dict:
    """`Cache(...)` kwargs for a quantised paged cache, or `{}` for `cache_bits` 0 (fp16)."""
    if not cache_bits:
        return {}
    from exllamav3 import CacheLayer_quant
    return dict(layer_type=CacheLayer_quant, k_bits=cache_bits, v_bits=cache_bits)
