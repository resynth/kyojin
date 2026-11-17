"""
Loader of a converted DS4.1 pack (the converter's output: `layers.N.safetensors` ..., `config.json` with
`quantization_config`, EXL3 tensors `<linear>.{trellis,suh,svh,...}`) into the serving model.

The weights go through the engine's framework: `Config.from_directory` (arch `DeepseekV41ForCausalLM`), the module tree of
`architecture/deepseek_v41.py` (`DSV41Block` declares every Linear and raw tensor of a layer under its official name),
`Module.load(device)` (EXL3 trellis -> `LinearEXL3`, plain `.weight` -> `LinearFP16`). `PackSource` then exposes the loaded
modules as a `WeightSource` (callable linears, raw tensors, expert handles) to `DSV41Serving`.

    from exllamav3.dsv41.pack import load
    model = load("/models/ds41-pack", device="cuda:0")          # DSV41Serving
    st = model.new_state(1, 8192)
    logits, _ = model.prefill(ids, st)                         # chunked prefill, then model.forward(tok, pos, st) per token

`load` is also the `serve.py --loader` entry (`tools/ds41/ds41_serving.py:load_model` wraps it into an engine).
"""
from __future__ import annotations
import copy
import os
import torch
import torch.nn.functional as F
from .config import DSV41Config
from .linear import Lin
from .weights import WeightSource
from .experts import ExpertHandle


class PackLin(Lin):
    """Callable view of a framework `Linear`: pads the input to the 128-aligned weight width, applies the fp16 weight
    (`LinearFP16`: float math on CPU, fp16 matmul on GPU) or the EXL3 kernels (`LinearEXL3`), trims padded outputs and
    returns the activation dtype of the caller."""

    def __init__(self, lin):
        self.lin = lin
        self.in_features, self.out_features = lin.in_features_unpadded, lin.out_features_unpadded
        self._wt = None

    def _dense_t(self, inner):
        if self._wt is None:                                  # CPU float path: [out, in] fp32, built once
            self._wt = inner.weight.t().float().contiguous()
        return self._wt

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        lin = self.lin
        xp = x
        if xp.shape[-1] < lin.in_features:
            xp = F.pad(xp, (0, lin.in_features - xp.shape[-1]))
        inner = lin.inner
        if getattr(inner, "quant_type", None) == "fp16":
            if inner.weight.device.type == "cpu":
                y = F.linear(xp.float(), self._dense_t(inner))
            else:
                y = torch.matmul(xp.half(), inner.weight).float()
            if inner.bias is not None:
                y = y + inner.bias.float()
        else:
            lead = xp.shape[:-1]
            if type(inner).__name__ == "CpuLinearEXL3":           # CPU: float activations, like the fp16 path above
                y = inner.forward(xp.float().reshape(-1, xp.shape[-1]), {}, torch.float32).reshape(*lead, -1)
            else:
                y = inner.forward(xp.half().reshape(-1, xp.shape[-1]).contiguous(), {}, torch.half).reshape(*lead, -1)
        if y.shape[-1] != lin.out_features_unpadded:
            y = y[..., :lin.out_features_unpadded]
        return y.to(x.dtype).contiguous()

    def nbytes(self):
        return 0


class PackSource(WeightSource):
    """WeightSource over loaded framework modules (text blocks, mtp blocks, embed / final norm / head)."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.raw, self.lins, self.blocks, self.wo_a = {}, {}, {}, {}
        self.embed = self.norm = None

    def add_block(self, b):
        """b: a loaded `DSV41Block`."""
        key = b.key
        self.blocks[b.l] = b
        for k, v in b.raw.t.items():
            self.raw[k] = v
        a = f"{key}.attn"
        for name, m in (("wq_a", b.wq_a), ("wkv", b.wkv), ("wq_b", b.wq_b), ("wo_b", b.wo_b)):
            self.lins[f"{a}.{name}"] = PackLin(m)
        if b.idx_wq_b is not None:
            self.lins[f"{a}.indexer.wq_b"] = PackLin(b.idx_wq_b)
        for nm, m in (("w1", b.sh_w1), ("w3", b.sh_w3), ("w2", b.sh_w2)):
            self.lins[f"{key}.ffn.shared_experts.{nm}"] = PackLin(m)
        if b.eg_wkv is not None:
            self.lins[f"{key}.engram.wkv"] = PackLin(b.eg_wkv)
        self.wo_a[key] = [PackLin(m) for m in b.wo_a]

    def add_extra(self, extra, main_proj):
        for k, v in extra.t.items():
            self.raw[k] = v
        self.lins["mtp.0.main_proj"] = PackLin(main_proj)

    # ---- WeightSource
    def has(self, name):
        return name in self.lins or name in self.raw or name + ".weight" in self.raw

    def tensor(self, name, dtype=torch.float32, device="cpu"):
        if name.endswith(".wo_a"):
            return torch.empty(0)                               # grouped output projection: wo_a_fn
        if name == "embed":
            t = self.embed
        elif name == "norm":
            t = self.norm
        else:
            t = self.raw.get(name)
            if t is None:
                t = self.raw[name + ".weight"]
        return t.to(dtype).to(device)

    def linear(self, name, dtype=torch.float32, device="cpu"):
        return self.lins[name]

    def wo_a_fn(self, prefix: str, groups: int):
        slices = self.wo_a[prefix]
        assert len(slices) == groups
        fn = lambda o: torch.stack([slices[g](o[:, :, g]) for g in range(groups)], dim=2)
        fn.slices = slices                      # dec2.wo_a: one launch for all groups
        return fn

    def expert(self, layer, e, dtype=torch.float32, device="cpu"):
        b = self.blocks[layer]
        if e in b.pruned:
            raise RuntimeError(f"layers.{layer}: expert {e} was pruned by the converter (K = 0) but the router selected it; "
                               "the pack's router bias must mask pruned experts")
        return ExpertHandle(layer, e, PackLin(b.ex_w1[e]), PackLin(b.ex_w3[e]), PackLin(b.ex_w2[e]))


def subset_config(dcfg: DSV41Config, layers: int = None, mtp: bool = True) -> DSV41Config:
    """Config of the first `layers` trunk layers (smoke tests) and/or without the dspark stages."""
    c = copy.deepcopy(dcfg)
    if layers is not None and layers < c.num_hidden_layers:
        c.num_hidden_layers = layers
        c.dspark_target_layer_ids = []
        c.kv_source_layer_ids = [i for i in c.kv_source_layer_ids if i < layers]
        c.index_source_layer_ids = [i for i in c.index_source_layer_ids if i < layers]
        if c.candidate_source_layer_id >= layers:
            c.candidate_source_layer_id = -1
        ne = list(c.engram_num_embeddings or [])
        keep = [i for i, e in enumerate(c.engram_layer_ids) if e < layers]
        if len(ne) == len(c.engram_layer_ids):
            c.engram_num_embeddings = [ne[i] for i in keep]
        c.engram_layer_ids = [c.engram_layer_ids[i] for i in keep]
        c.num_nextn_predict_layers = 0
    if not mtp:
        c.num_nextn_predict_layers = 0
    return c


def _bind_src(src, mod, text_mtp) -> str:
    """Wire one loaded module into the PackSource; the single dispatch shared by the split and split-off
    load branches so a module key is handled in exactly one place. Returns the key. Trunk and MTP block
    modules get src.add_block here; whether they join the `blocks` list is the caller's business (MTP
    stages are walked via mtp_mod separately)."""
    key = mod.key
    if key == "embed":
        src.embed = mod.embedding.weight.data
    elif key == "norm":
        src.norm = mod.w
    elif key == "head":
        src.lins["head"] = PackLin(mod.head)
    elif key == "mtp.extra":
        src.add_extra(mod, text_mtp.main_proj)
    elif key.startswith("layers.") or key.startswith("mtp."):
        src.add_block(mod)
    return key


def load(model_dir: str, device="cuda:0", dtype=torch.float16, layers: int = None, mtp: bool = True,
         moe: str = "auto", prefill_chunk: int = 2048, engram_store=None, log=None, placement=None, **kw):
    """Load the pack in `model_dir` into a `DSV41Serving` on `device`.

    layers  only the first N trunk layers (+ embed / final norm / head): the smoke test of a real pack on one layer
    mtp     load the dspark stages when the pack has them
    moe     "auto": mixed-K grouped kernels where the device and the tensors allow (else per-expert loop), "loop": always loop
    engram_store  an `EngramRowStore` (default: the pack's shards, or $EXL3_DSV41_ENGRAM_DIR)
    placement  a `dsv41.placement.DSV41Placement` for the contiguous layer split (dsv41.placement); None = split off,
               which loads exactly as before (single device, no bridge, no env changes)
    """
    from ..model.config import Config
    from ..model.model import Model
    from ..modules.moe_mixedk import parse_expert_bits, MixedKUnsupported
    from .serve_model import DSV41Serving
    from .serve_moe import GroupedMoE, LoopMoE, grouped_available
    log = log or (lambda s: None)
    if layers is not None and placement is not None:
        raise ValueError("pack.load: the --layers smoke subset and the layer split are mutually exclusive")
    split = placement is not None and (placement.part0_layers > 0 or placement.endcaps)
    if split:
        # D6: graphs and the split's host-staged crossing never mix. Forced before the serving model reads
        # the env (DSV41Serving.__init__ captures decode_graph), regardless of the tier.
        os.environ["DSV41_DECODE_GRAPH"] = "0"
    dev = torch.device(device)
    cfg = Config.from_directory(model_dir, **({"dsv41_engram_store": engram_store} if engram_store else {}))
    full = cfg.dsv41
    dcfg = subset_config(full, layers, mtp and cfg.dsv41_n_mtp > 0)
    n_load = dcfg.num_hidden_layers
    qcfg = cfg.config_dict.get("quantization_config") or {}
    if split:
        if placement.cfg is not dcfg:
            import dataclasses
            placement = dataclasses.replace(placement, cfg=dcfg)
        cfg._expert_tier_device = placement.tier          # the tier guard (modules/dsv41.py) reads this
        cfg._dsv41_placement = placement
        from ..bridge import install_bridge
        placement.bridge = install_bridge(cfg, part1=placement.part1, part0=placement.part0)
        bridge_desc = placement.bridge.describe() if placement.bridge else "disabled (KYOJIN_BRIDGE=0): direct/host copies"
        log(f"[split] bridge: {bridge_desc}")

    text = Model.from_config(cfg)
    src = PackSource(full.num_hidden_layers)
    blocks = []
    per_dev_bytes: dict = {}
    if split:
        d_end, d0 = placement.endcap_device, placement.part0
        plan = []
        for mod in text.modules:
            key = mod.key
            if key.startswith("layers.") and int(key.split(".")[1]) >= n_load:
                continue
            if key.startswith("layers."):
                dv = placement.text_device(int(key.split(".")[1]))
            elif key in ("embed", "hc_expand", "norm", "head"):
                dv = d_end
            else:
                dv = d_end                                # any stub module rides with the endcaps
            plan.append((mod, dv))
        if dcfg.n_mtp_layers:
            mtp_mod_plan = Model.from_config(cfg, component="mtp")
            plan += [(mod, placement.mtp_device) for mod in mtp_mod_plan.modules]
            text_mtp = mtp_mod_plan
        else:
            text_mtp = None
        # startup budget estimate to sanity-check the 9070 front budget before the load (no auto-N)
        try:
            from .placement import estimate_front_bytes
            def _numel(key, _t=text, _m=text_mtp):
                try:
                    mod = _t.find_module(key)
                except (KeyError, AttributeError):
                    try:
                        mod = _m.find_module(key) if _m is not None else None
                    except (KeyError, AttributeError):
                        mod = None
                return mod.weights_numel() if mod is not None else 0
            try:
                qbits = float(qcfg.get("bits", 2.9))
            except (TypeError, ValueError):
                qbits = 2.9
            est = estimate_front_bytes(placement, _numel, bits=qbits)
            log(f"[split] front budget estimate: {est['total_gib']:.2f} GiB of weights on part0 "
                f"(layers {est['layers']}, endcaps {est['endcap_bytes'] / 2**30:.2f} GiB, "
                f"MTP {est['mtp_bytes'] / 2**30:.2f} GiB) + {placement.part0_reserve_gb:.2f} GiB reserve")
        except Exception as ex:                                   # noqa: BLE001 - informative only
            log(f"[split] front budget estimate unavailable ({ex!r})")
        # part0 reserve slab: held while the front (and, later, the endcap / MTP) modules load, released
        # just before the first part1 module so the front load always has its headroom on the 9070.
        reserve_bytes = int(placement.part0_reserve_gb * (1 << 30))
        slab = None
        for mod, dv in plan:
            if dv != d0:
                if slab is not None:
                    slab = None                           # release before the first part1 module
                    log("split: part0 reserve released")
            elif slab is None and reserve_bytes:
                try:
                    slab = torch.empty(reserve_bytes, dtype=torch.uint8, device=d0)
                except RuntimeError as ex:
                    log(f"split: part0 reserve of {placement.part0_reserve_gb:.2f} GiB not engageable "
                        f"({ex}); the end-of-load check still applies")
                    slab = None
            before = torch.cuda.memory_allocated(dv)
            mod.load(dv)
            per_dev_bytes[dv] = per_dev_bytes.get(dv, 0) + (torch.cuda.memory_allocated(dv) - before)
            key = _bind_src(src, mod, text_mtp)
            log(f"loaded {key} on {dv}")
            if key.startswith("layers."):
                blocks.append(mod)
        slab = None
        mtp_mod = text_mtp
    else:
        mtp_mod = None
        for mod in text.modules:
            key = mod.key
            if key.startswith("layers.") and int(key.split(".")[1]) >= n_load:
                continue
            mod.load(dev)
            log(f"loaded {key}")
            key = _bind_src(src, mod, mtp_mod)          # text modules never hit the mtp.* keys
            if key.startswith("layers."):
                blocks.append(mod)
        if dcfg.n_mtp_layers:
            mtp_mod = Model.from_config(cfg, component="mtp")
            for mod in mtp_mod.modules:
                mod.load(dev)
                log(f"loaded {mod.key}")
                _bind_src(src, mod, mtp_mod)

    store = tm = mult = None
    if dcfg.engram_layer_ids:
        eg = cfg.engram_runtime()
        tm, mult = eg["token_map"], eg["multipliers"]
        if not dcfg.engram_compressed_vocab_size:
            dcfg.engram_compressed_vocab_size = eg["compressed_vocab"]
        store = engram_store or cfg.engram_store()
    tiered = [b for b in blocks if getattr(b, "tier", None) is not None]
    tier_dev = placement.tier if split else dev
    if tiered:
        # SSD expert tier (EXL3_EXPERT_BUDGET_GIB): the kernels address experts through the store's slot tables; the decode
        # graph would capture stale table contents and the host sync of the store cannot run inside a capture
        os.environ["DSV41_DECODE_GRAPH"] = "0"
        if moe == "loop" or not grouped_available(tier_dev):
            raise RuntimeError("EXL3_EXPERT_BUDGET_GIB needs the grouped MoE kernels (moe=auto on a GPU with the extension)")
        if split:
            bad = [b.key for b in tiered if torch.device(b.device) != tier_dev]
            if bad:
                raise RuntimeError(f"split: tiered layers {bad} are not on the tier device {tier_dev}")
    model = DSV41Serving(dcfg, src, None, store, token_map=tm, multipliers=mult, dtype=dtype, device=dev,
                         prefill_chunk=prefill_chunk, ablit_dir=kw.pop("ablit_dir", cfg.directory),
                         placement=placement, **kw)
    model.pack_modules = (text, mtp_mod)                      # keep the module trees (and their tensors) alive
    if moe != "loop":
        for b in blocks + [m for m in (mtp_mod.modules if mtp_mod else []) if m.key != "mtp.extra"]:
            bdv = torch.device(b.device)
            if not grouped_available(bdv):
                if getattr(b, "tier", None) is not None:
                    raise RuntimeError(f"{b.key}: tiered layer needs the grouped MoE kernels on {bdv}")
                continue                        # this device cannot host the grouped kernels: per-expert loop
            if b.pruned:
                log(f"{b.key}: {len(b.pruned)} pruned experts; per-expert loop")
                continue
            try:
                eb = parse_expert_bits(qcfg, f"{b.key}.ffn", b.n_exp)
                be = GroupedMoE(b.ex_w1, b.ex_w3, b.ex_w2, dcfg, bdv, eb,
                                fallback=model.loop_moe_for(bdv), layer=b.l)
                if getattr(b, "tier", None) is not None:
                    if not (be.decode_ok and be.wmma_ok):
                        raise RuntimeError(f"{b.key}: tiered layer needs the grouped decode and prefill kernels")
                    b.tier.bind_mixed(be.store)          # K-group pointer tables become views of the store's group buffers
                    be.tier = b.tier
                model.moe_backends[b.l] = be
            except (MixedKUnsupported, AssertionError, ValueError) as ex:
                if getattr(b, "tier", None) is not None:
                    raise RuntimeError(f"{b.key}: tiered layer cannot use the grouped MoE ({ex})") from ex
                log(f"{b.key}: grouped MoE unavailable ({ex}); per-expert loop")
    if tiered:
        from ..modules.expert_store import get_store
        st_ = get_store(cfg)
        st_.finalize()                                     # allocate the slot pools after every other allocation of the load
        if split:
            if torch.device(st_.device) != tier_dev:
                raise RuntimeError(f"split: expert store device {st_.device} != tier device {tier_dev}")
            front = [b.key for b in blocks if placement.front(b.l) and getattr(b, "tier", None) is not None]
            if front:
                raise RuntimeError(f"split: front layers {front} registered with the expert tier")
            log(f"[split] expert tier on {tier_dev}; front layer(s) resident, no SSD reads")
    if split:
        # split contract for every split load, tier or not: the load table and the part0 headroom check
        # fail at load time instead of as a mid-prefill OOM
        lines = ", ".join(f"{dv}: {nb / 2**30:.2f} GiB" for dv, nb in per_dev_bytes.items())
        log(f"[split] load table (weights): {lines} (endcaps on {placement.endcap_device}, MTP on "
            f"{placement.mtp_device}, expert tier on {tier_dev})")
        free = torch.cuda.mem_get_info(placement.part0)[0]
        need = int(placement.part0_reserve_gb * (1 << 30))
        log(f"[split] part0 free after load: {free / 2**30:.2f} GiB (reserve {placement.part0_reserve_gb:.2f} GiB)")
        if free < need:
            raise RuntimeError(f"split: part0 ({placement.part0}) has only {free / 2**30:.2f} GiB free after "
                               f"the load but the reserve wants {placement.part0_reserve_gb:.2f} GiB. "
                               "Reduce --part0-layers, use --endcaps off, or run with --num-draft 0.")
    return model
