# Contiguous layer split

The layer split places the first N transformer layers of a model on one GPU (`part0`) and
the remaining layers on another (`part1`) in a single process. It is for models that do not
fit on one device, or for pairing a small fast card with a large unified-memory device.

Status: GLM-5.3-Flash is measured and supported by the GLM server. Qwen3.8-Flash-Next and
DeepSeek-V4.1-Flash adapters are staged; Flint and Mimo use the generic model path. See
[Model support](#model-support).

## How it works

    part0: layers 0..N-1 + endcaps          part1: layers N..last
        \______________________________________/
                    host-staged crossings

`part0` and `part1` are positions in the layer order, not device indices. Either physical
device can serve either part.

Every tensor that crosses the split goes through an asynchronous host-staged bridge:

    src tensor --D2H--> pinned slot ring --H2D--> dst tensor

Both legs are enqueued on the two devices' compute streams, so the host does not wait. The
destination is a fresh allocation. `KYOJIN_BRIDGE=0` disables the bridge; cross-device
moves then use `to_device` with a blocking host bounce instead of probing or using a direct
peer copy. An explicit `EXLLAMA_NO_P2P_COPY` still wins.

Measured configuration: an RX 9070 (`gfx1201`, 15.92 GiB) over USB4 as `part0` and a
118 GiB Strix Halo iGPU (`gfx1151`) as `part1`. The numbers below are measurements from
that pair, not assumptions of the interface.

## Quick start

The GLM server and the split harnesses accept the flags below. The environment-only path
is useful for wrappers that cannot pass CLI flags:

```sh
KYOJIN_PART0_LAYERS=6 \
KYOJIN_PART0_DEVICE=gfx1201 \
KYOJIN_PART1_DEVICE=gfx1151 \
  python tools/glm/serve.py --model /path/to/model
```

## Interface

| Flag | Environment fallback | Meaning |
| --- | --- | --- |
| `--part0-layers auto\|N` | `KYOJIN_PART0_LAYERS` | first N transformer layers on `part0`; 0 = split off |
| `--part1-device auto\|cuda:N\|gfxXXXX\|uuid\|name` | `KYOJIN_PART1_DEVICE` | `part1` device (`auto` = `cuda:0`) |
| `--part0-device auto\|cuda:N\|gfxXXXX\|uuid\|name` | `KYOJIN_PART0_DEVICE` | `part0` device (`auto` = the only other device when two are visible) |
| `--endcaps on\|off` | `KYOJIN_ENDCAPS` | keep text endcaps and whole MTP/vision components on `part0` (default on) |
| `--part0-reserve-gb` | `KYOJIN_PART0_RESERVE_GB` | VRAM held as a load-time slab on `part0` for workspace (default 1.2) |
| `--split-preset auto\|glm\|qwen\|ds41\|none` | `KYOJIN_SPLIT_PRESET` | per-model unsafe-environment preset; GLM needs none today |
| `--split-graphs auto\|on\|off` | - | reserved for per-segment capture; split currently runs graphs off |
| - | `KYOJIN_DS41_TIER_DEVICE` | DSV4 only: which side hosts the SSD expert tier (`part1` default; `part0` only on explicit request) |
| `KYOJIN_BRIDGE_CHUNK_MB` | - | DSV4: the pinned-staging budget the prefill bridge slices against (default 256) |

`cuda:N` is the torch device index **after** `HIP_VISIBLE_DEVICES`, not the `rocm-smi`
index. Explicit selectors win in either position: `part0=cuda:1/part1=cuda:0` and
`part0=cuda:0/part1=cuda:1` are both valid. With exactly two visible devices, a part0-only
selector still works and `part1=auto` takes the other device. Two selectors that resolve to
the same device fail, and a one-device split fails with a clear error. With more than two
visible devices, `auto` requires an explicit selector and prints the inventory.

`KYOJIN_PART0_LAYERS` and `KYOJIN_PART1_DEVICE` also work in `model_init.init` paths
(examples and evaluation tools) without CLI flags. The legacy `EXL3_LOAD_SPLIT` budget knob
is not part of this interface and must not be combined with the new map.

## What gets placed

| Where | Contents |
| --- | --- |
| `part0` | first N text layers; with `--endcaps on`, the text allowlist (norm/hc/lm_head-like) and whole non-CPU MTP/vision components |
| `part1` | remaining text layers; MTP/vision when endcaps are off; unknown non-layer modules |
| CPU | the shared `embed_tokens` (`prefer_cpu`) stays host-side by design |
| cache | each KV/recurrent layer follows its owning layer; a placement audit prints after load |
| bridge | one bridge per config, covering the target, drafter and vision |

Text endcaps are selected by a class/key allowlist, not by per-model placement in the core.
“Whole component” means all non-CPU modules of that component.

## Memory and capacity

- The `part0` reserve is an inert load-time slab, not a process-wide allocator cap. It
  removes bytes from the pool while the model loads and is released at the first `part1`
  module. Size it to the part0 card.
- Split mode runs eager. Whole-forward and per-block graphs are disabled; per-segment
  capture is not implemented yet.
- Use `util/memguard.py` when loading near the limit.  It help save you from the dreaded OOM but slight adverse 
  performance effects have been noted (to be investigated further.)

## Model support

| Model | Status | Notes |
| --- | --- | --- |
| GLM-5.3-Flash | measured | text, MTP and vision; on a 16 GiB `gfx12` `part0`, N=6 fits short context (MTP off), N=4 is the 128K MTP setting, and N=5 fits 500K |
| Qwen3.8-Flash-Next | staged, not load-tested | QSA/GDN/PLE components; separate dflash drafter config; QSA `gfx12` caveat below |
| DeepSeek-V4.1-Flash | implemented (text first), not load-tested | own adapter (`tools/ds41/serve.py`); see the DSV4 section below |
| Flint, Mimo | best effort | generic `Model` path; no dedicated adapter |

Kernel support is per architecture. The split owns placement and transport, not kernel
ports; a missing kernel fallback on a new architecture is a model/kernel issue, not a split
defect.

### DeepSeek-V4.1-Flash

The DSV4 engine builds its serving model itself (`exllamav3/dsv41/pack.py`), so the split
adapter is DSV4-local placement (`exllamav3/dsv41/placement.py`) on top of the shared
device selection, settings and bridge. Differences from the generic map:

- **Boundaries are structural.** DSV4 layers read other layers' compressed-KV and
  indexer-key pools and the candidate-block selection produced by
  `candidate_source_layer_id`. A boundary is valid only when every producer a `part1`
  layer needs is itself on `part1`. For the default 40-layer config the allowed
  `--part0-layers` values are therefore `0, 1, 2, 8, 14, 20, 40` (computed from the
  config, never hardcoded); any other N fails at startup with this list. `N=1` or `N=2`
  is the intended target (the small card holds the front).
- **`auto` is rejected for DSV4 in this stage** — use a manual N.
- **Endcaps carry the MTP.** DSpark reads `embed`/`head`, so endcaps and the MTP/draft
  stages always share one device: `part0` with `--endcaps on`, `part1` with
  `--endcaps off`. If that does not fit the part0 budget, the load fails with a clear
  message; reduce N or use `--endcaps off` / fewer draft tokens.
- **SSD expert tier on one device.** With `EXL3_EXPERT_BUDGET_GIB` on, the tier streams
  onto exactly one device — `part1` by default, `KYOJIN_DS41_TIER_DEVICE=part0` only on
  explicit request. Front layers never register with the store: their experts are
  resident, and the front never reads the tier.
- **Graphs off** (`DSV41_DECODE_GRAPH=0` via the `ds41` preset). The JIT modules are
  probed **per device and per module** at load, not as one process-wide switch: on a
  `gfx12` `part0` the decode chain, fused glue and JIT MoE (`dsv41_hip`, `dsv41_fuse`,
  `dsv41_moe`) build and run there, while the WMMA prompt-attention and indexer modules
  (`ds41_pf`, `dsv41_idx`) cannot compile and take their torch fallbacks on that device
  only. The iGPU keeps every fast path. `DSV41_SPLIT_HIP=0` disables every module on
  every split device, for A/B against the torch path.
- **Bounded bridge staging.** A 32K `part0`→`part1` crossing is a 2.69 GB fp32 tensor;
  `hop_chunk` slices such crossings so the pinned staging stays near
  `KYOJIN_BRIDGE_CHUNK_MB` (default 256 MiB) ×2 slots instead of two full chunks.
- **What fits a 16 GiB front.** `N=1` with endcaps on fits with headroom and is the measured
  setting: decode is faster than split-off and prefill is at parity (the row-blocked torch
  prompt pass keeps the 32K single chunk on the card). A 94 GiB expert pool is the best
  all-round budget (32K and 128K both safe); the split does **not** buy a materially larger
  pool: the ~5 GiB of weights it moves to VRAM are largely offset by the bridge's pinned
  staging and per-device state. The gain grows with context (decode +5 % at 32K, +9 % at
  128K against split-off at the same/best pool), while a cold 128K prefill costs ~5 % extra.
  `N=2` does not fit at 32K (with or without the endcaps on the front), and DSpark does not
  fit next to layer 0 + endcaps; for MTP use `--endcaps off` (the head and the MTP stages
  move to `part1`, the front keeps layer 0) or a front card with more memory.

Example (the working orientation is the small card as `part0`):

```sh
KYOJIN_PART0_LAYERS=1 \
KYOJIN_PART0_DEVICE=gfx1201 \
KYOJIN_PART1_DEVICE=gfx1151 \
KYOJIN_ENDCAPS=on \
  python tools/ds41/serve.py --model /path/to/ds41-pack --num-draft 0 --engine-opt prefill_chunk=32768
```

The reverse orientation is selectable but not fit-feasible for this pack: a safe tail
boundary would have to sit at or above the candidate source layer, which puts the tail
layers' resident experts on the small card.

## Verification

- Split **off** must remain byte-identical to the parent branch, including outputs and the
  serve environment block. It must not enumerate devices or mutate split settings.
- Split **on** is judged by run-to-run determinism with plain greedy decoding, KLD/task
  metrics and the needle probe. A mixed-architecture pair is not expected to be bitwise.
- `tools/strix_halo/split_eval.py` runs the battery: `--baseline`, `--part0-layers`,
  `--part1-device`/`--part0-device`, `--needle-depth`, `--bench-depth`,
  `--bench-prompt passkey|summary`, `--bench-acceptance`, `--num-draft`,
  `--perf-tokens`/`--perf-reps`, `--vision-image PATH` and `--compare a.json b.json`.
  Its context-dependent part0
  default is harness tuning, not a split property; override it with `--part0-layers` or the
  `--long-ctx-part0-layers`/`--short-ctx-part0-layers` flags. `--bench-acceptance` uses
  long-form summary prompts and reports the drafted-token sample plus a 95% interval;
  the passkey prompt is too short for an acceptance comparison. `--vision-image` encodes
  one image through the split vision tower; the harness sets `EXL3_CONV_GEMM=1` as the
  safe gfx12 spatial-merge fallback.
- `--kld-corpus PATH --kld-slices N --kld-len L --kld-out FILE` teacher-forces corpus
  slices and `--kld-compare BASE SPLIT` reports KLD both ways, top-1 agreement and PPL.
  The short battery also reports programmatic task checks. Use these for split-on quality:
  a mixed-architecture pair is not bitwise.
- `tools/strix_halo/split_sessions_smoke.py` sends N concurrent requests at a running
  server and checks that every response succeeds and that the requests overlap.
- `tools/strix_halo/split_startup_health.py` polls `/health` during load and checks the
  stage trace; run it with and without `--no-vision` and with `--num-draft 0`.

## Known limitations

- Split-on is deterministic run-to-run, but it is **not bitwise** equal to a single
  device on a mixed-architecture pair. Measured on GLM, N=4: 4/6 short greedy prompts
  identical, task checks 5/6 vs 4/6, PPL 9.374 vs 9.401 and KLD 2.08e-2 with 94.52%
  top-1 agreement over 1022 positions. The distributions and task accuracy are
  unchanged; a few near-tie argmax positions flip. Use KLD/task metrics for split-on.
- At 500K context the split fits the measured pair (`part0` 10.75 GiB, `part1` 100.2 GiB
  load-time peak). A single 118 GiB device did not have headroom for the same context
  there; size for capacity, not only speed.
- Concurrent serving is smoke-tested at 4 sessions with MTP on (4/4 responses overlap,
  no queue or slot error). A heavier mixed chat/image/tool workload is future work.
- MTP on the split is deterministic at 8K/32K/128K with N=4 in the measured
  configuration (two independent runs: 6/6 greedy prompts and identical long-prompt ids).
  The earlier N=6 128K battery used short prompts; N=6 long prefill OOMs a 16 GB part0,
  so N=4 is the 128K MTP configuration and the 500K capacity run used N=5. Use
  `--num-draft 0` for configs outside this measured set.
- Per-segment graphs are not implemented. Split runs eager; measured GLM decode is -3.4%
  at 8K and +1.3% to +8.1% at 32K/128K versus single device with MTP on.
- Auto-N subtracts the MTP drafter and vision tower when the adapter predeclares them
  before load (the GLM server does). The default 2 GiB safety covers the measured 32K
  MTP+vision warm-up (1 GiB still chose N=6 and OOMed; 1.5/2/3 GiB chose N=5), but it is
  a fixed stopgap. Stage 5 sizes the forward workspace from the context and cache so
  short context can use more part0 layers and long context reserves more;
  `KYOJIN_AUTON_SAFETY_GB` remains the override.
- Qwen QSA sparse attention has an illegal-access report in the `gfx12` top-k path on
  `gfx1201`; keep QSA cache pages off `part0` until that kernel is fixed.
- Reverse orientation with MTP on is validated at 32K (N=42, endcaps off, MTP on
  `part1`) on the measured configuration.
- Same-architecture and RDNA3/3.5 pairs are selection-tested only until such hardware is
  available.
