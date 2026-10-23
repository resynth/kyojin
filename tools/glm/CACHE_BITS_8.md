# `--cache-bits 8`: measurement log, crash notes and remaining work

Status doc + partial deliverable for the plan `.kilo/plans/1792712478833-glm-cache-bits-8-measure.md`.

**State:** steps 0, 1, 2, 3, 6 complete and measured. Steps 4, 5, 7, 8 and the pytest
validation were **not** run. The measurement box (`m@10.10.10.2`) **crashed** while starting the
first step-4 `bench_mtp` smoke run (see *Crash notes*). Everything below is copied verbatim from the
runs; nothing here is estimated except the explicitly-labelled linear extrapolations.

---

## Crash notes

- Time: 2026-10-23 ~01:29–01:41 UTC.
- The box was healthy through all of steps 0–3 and 6. Its last good state:
  - an int8 server had loaded the NicSir pack: `serve: mem[loaded] ... torch alloc 109.8 reserved
    109.8 GB` at `-c 40960 --no-vision`.
  - slot checks then ran (two `slot_check`/`cross_slot` cycles, ~110.5 GB each) and finished.
- **What I was doing when it crashed:** launching the step-4 validation smoke run,
  `bench_mtp.py -m ~/models/glm53-exl3-td205 --cache-bits 8 -c 4096 -n 40 -g`, in the background,
  and polling its log. The first two attempts returned empty output; on the third poll the SSH
  session was closed by the remote host, then the box stopped answering (`ping` fail ×3,
  `ssh ... "No route to host"`).
- **Likely cause:** memory exhaustion / crash on the box. It runs 124 GB RAM with a 126 GB GTT
  pool; the server already sat at ~110 GB alloc for target+draft+vision-less load at `-c 40960`,
  and a fresh `bench_mtp` load (target + MTP draft + cache) plus Triton/graph warm-up can exceed
  what remains. This matches the user's warning that the box is "close to OOMing".
- **Not a thunderbolt-data cause:** no laptop↔box bulk transfer was in flight (only SSH traffic;
  one 4.6 KB script and a 1.8 KB grid script were sent earlier).
- **Recovery needed:** physical/power check of the box; confirm it boots and `rocm-smi` sees the
  GPU, then re-verify `~/models/glm53-exl3-td205` symlink and `~/cache/llama-slots`.

---

## Environment deviations from the plan (record these in the final report)

1. **Pack:** `~/models/glm53-exl3-td205` did **not** exist (`~/models` was empty). Per the user,
   the plan measures the **NicSir `GLM-5.3-Flash-EXL3-2.72bpw`** pack, symlinked as
   `~/models/glm53-exl3-td205`:
   ```
   ln -sfn /data/hf/hub/models--NicSir--GLM-5.3-Flash-EXL3-2.72bpw/snapshots/db672fc5e61f1d61bdf6d641fb63538b80bb7613 \
           ~/models/glm53-exl3-td205
   ```
   RED-SNOW 2.49bpw exists on the box but the user says it does not work. No models downloaded.
2. **Repo path:** the clone is `~/engines/kyojin` (the plan says `~/kyojin`). Remote is
   `https://github.com/resynth/kyojin.git`.
3. **HEAD:** the plan's `c105ba7` is docs-only and had to be fetched; the harness commit built on
   top is **`9eb14ce`** (see *Commits*). Box was reset to it before the server/slot runs.
4. **PPL corpus:** `~/bench/ppl/wiki.test.raw` was absent; generated deterministically with
   `tools/glm/make_corpus.py ~/bench/ppl/wiki.test.raw 400000` (no download).
5. **GTT sysfs:** the used-memory file is `/sys/class/drm/card0/device/mem_info_gtt_used` on this
   box (the plan says `card1`). `serve.py`'s `mem_line` prints `gtt nan` here regardless.
6. **MTP eh_proj sidecar absent:** server logs `serve: MTP eh_proj sidecar: none ... draft
   acceptance is lower` — no `mtp_eh_proj.st` in the NicSir pack folder and no
   `~/models/glm53-mtp-eh-proj-bf16.safetensors`. Step-4 acceptance numbers will be lower than a
   pack with the sidecar; note it in any comparison.
7. **Slot dir:** `~/cache/llama-slots` did not exist and had to be created
   (`mkdir -p ~/cache/llama-slots`) before any `/slots/0?action=save` works.

---

## Commits

| commit | what | where |
|---|---|---|
| `32951fa` | "Add --cache-bits 8" (feature) | pre-existing |
| `18cdd3c` | parent (pre-change), used by step 8 | pre-existing |
| `c105ba7` | "Tidy comments" (deleted the caveat docs) | pre-existing |
| `9eb14ce` | harness: `bench_mtp --cache-bits/--temp` + `tools/glm/niah.py` | pushed to `origin/feat/glm-kv-q8` |

`9eb14ce` contents:
- `tools/strix_halo/bench_mtp.py`: added `--cache-bits {0,8}` (builds the packed-latent cache for
  **both** the target and MTP draft cache, as `tools/glm/serve.py` does) and `--temp FLOAT`
  (`0` = greedy via `CategoricalSampler(0)`; unset keeps the previous sampler). Defaults preserve
  today's fp16 behaviour.
- `tools/glm/niah.py`: new HTTP needle-in-a-haystack driver (lorem haystack, reference value at a
  fixed depth, greedy answer, exact-value recall per depth per trial).

---

## Step 0 — config dims (box: `m@10.10.10.2`, NicSir pack)

```
num_hidden_layers 45
kv_lora_rank 512
qk_rope_head_dim 0            # NoPE
index_head_dim 128
index_kpool 4
index_n_heads 32
index_topk 2048
num_nextn_predict_layers 1
```
`layer_types`: `deepseek_sparse_attention` at indices 3,7,11,15,19,23,27,31,35,39,43 (11 MLA
layers); the rest are `linear_attention` (KDA, separate recurrent state). `indexer_types` is
`full` for **all 45** entries, so **all 11 MLA layers are "full"** → every one carries the DSA
indexer plane and the k-pool plane.

---

## Step 1 — byte accounting

### 1.1 Analytic (per token, per **MLA layer**, per plane)

`idx_plane_dim = index_head_dim*2 = 256` (because `index_kpool != 0`).

| plane | fp16 B/token | int8 B/token | notes |
|---|---|---|---|
| latent `k` / packed `qk`+`sk` | `2*512 = 1024` | `(512/32)*8*4 + (512/32)*2 = 512 + 32 = 544` | the **only** quantised plane |
| rope key `v` | `2*qk_rope_head_dim = 0` | `0` | NoPE; empty plane, stays fp16 (nothing) |
| DSA indexer `k_idx` | `2*256 = 512` | `512` | stays fp16 (DSA) |
| k-pool `k_pool` | `2*(128/4) = 64` | `64` | stays fp16 (k-pool) |
| **per MLA layer** | **1600** | **1120** | ratio 1.43× |
| **target (11 MLA layers)** | **17600** | **12320** | |
| **MTP draft (1 MLA layer)** | **1600** | **1120** | |
| **target + draft** | **19200** | **13440** | saving 5760 B/tok = 30.0% |

### 1.2 Empirical (`storage_size()` sums; no weights loaded — cache geometry comes from the
attention config)

```
comp None CQ 0 cs 16384 nt 20480 nlayers 11 {'CacheLayer_MLA_fp16': 11}  raw 360448000  B/tok 17600.0
comp None CQ 8 cs 16384 nt 20480 nlayers 11 {'CacheLayer_MLA_quant': 11} raw 252313600  B/tok 12320.0
comp None CQ 0 cs 40960 nt 45056 nlayers 11 {'CacheLayer_MLA_fp16': 11}  raw 792985600  B/tok 17600.0
comp None CQ 8 cs 40960 nt 45056 nlayers 11 {'CacheLayer_MLA_quant': 11} raw 555089920  B/tok 12320.0
comp mtp  CQ 0 nt 45056 nlayers 1  raw 72089600 B/tok 1600.0
comp mtp  CQ 8 nt 45056 nlayers 1  raw 50462720 B/tok 1120.0
```
Measured planes (one layer, `nt = 45056`, `pages = 176`):

| mode | tensors (shape, dtype) |
|---|---|
| fp16 | `k (176,256,1,512) f16`, `v (176,256,1,0) f16`, `k_idx (176,256,256) f16`, `k_pool (176,64,128) f16` |
| int8 | `qk (176,256,128) int32`, `sk (176,256,16) f16`, `v (176,256,1,0) f16`, `k_idx (176,256,256) f16`, `k_pool (176,64,128) f16` |

**Reconciliation: measured == analytic exactly** (no gap). Reasons: `max_num_tokens = ctx+4096` is
always a multiple of `PAGE_SIZE` (256), so there is no page rounding; and every MLA layer is a
"full" indexer, so there is no full/shared mix to explain.

### 1.3 Linear extrapolation (cache-only, `max_num_tokens = ctx + 4096`, target+draft)

| ctx | tokens | fp16 | int8 | int8 saves |
|---|---|---|---|---|
| 16384 | 20480 | 375.0 MiB | 262.5 MiB | 112.5 MiB |
| 40960 (max run) | 45056 | 825.0 MiB | 577.5 MiB | 247.5 MiB |
| 131072 *(extrapolated)* | 135168 | 2.417 GiB | 1.692 GiB | 0.725 GiB |
| 524288 *(extrapolated)* | 528384 | 9.448 GiB | 6.614 GiB | 2.834 GiB |

The 131072/524288 rows are linear extrapolations, not measured (`-c` capped at 40960).

---

## Step 2 — PPL (cacheless; expected and measured delta 0)

Two identical runs (the protocol has no cache flag — "both modes" is the same command):

```
python tools/glm/ppl7.py --out /tmp/cb8/ppl_fp16.json     # run 1
python tools/glm/ppl7.py --out /tmp/cb8/ppl_second.json   # run 2
```
Result: **ppl = 2.868039** in both runs, `per_row_ppl` equal, **delta 0.0** (abs and relative
0.000000 / 0.00%). Protocol: first 10 non-overlapping rows of 4096, score positions 1..4095,
forward with `attn_mode="flash_attn_nc"`, which asserts `cache is None`
(`exllamav3/modules/mla_attn.py:1467`) — it cannot exercise the latent quantiser at all, so the
identical output is by construction.

Corpus note: `~/bench/ppl/wiki.test.raw` was generated by `tools/glm/make_corpus.py` (400000
chars, seed 20251007) because the plan's wikitext file was absent; absolute PPL is therefore not
comparable to earlier wikitext numbers, only the fp16↔int8 delta is meaningful (and is 0).

---

## Step 3 — retrieval (needle-in-a-haystack), server-only

Driver: `tools/glm/niah.py` (commit `9eb14ce`). Server flags (both modes identical except
`--cache-bits`):

```
python tools/glm/serve.py --model ~/models/glm53-exl3-td205 --host 127.0.0.1 --port 18080 \
  -c 40960 --num-draft 1 --no-vision --cache-bits {0|8}
```
`--tokens 40000`, depths `0.1,0.5,0.9`, 3 trials each, greedy, seed 1234.

| mode | depth 10% | depth 50% | depth 90% | total |
|---|---|---|---|---|
| fp16 | 3/3 | 3/3 | 3/3 | **9/9** |
| int8 | 3/3 | 3/3 | 3/3 | **9/9** |

`prompt_tokens` = 40083 every request; cold wall ≈ 80–85 s, resumed ≈ 18 s.

---

## Step 6 — behaviour checks (int8 unless stated)

Server flags unless noted: `-c 40960 --num-draft 1 --no-vision --cache-bits 8`.

1. **Same prompt twice (greedy, MTP on)** → **identical text**.
2. **MTP on vs off (greedy)** → diverge at **token 25 of 62** (prefix 40.3%); ON has
   "This text is a **long** series …", OFF has "This text is a series …".
3. **`--sessions 2`, two identical prompts concurrently** → the two concurrent rows are
   **identical to each other** but **both differ from the solo run** at char 130 of 345 (~38%).
   Confirms the row-invariance caveat for `--sessions > 1` (the int8 paged attention is not
   row-invariant across batched rows).
4. **`--exact-prefix-cache` + int8**: cold `cache_n = 0`, `prompt_ms = 10779`; resumed
   `cache_n = 4729`, `prompt_ms = 5.23`; **identical text**.
5. **Slots**:
   - same-mode int8, `tools/slot_check.py` → **PASS** (`n_saved 4608 == n_copied 4608`,
     `warm_cache_n 4608`, `turn2_cache_n 4736`, `same_text true`, cold 12.1 s / warm 3.26 s).
   - cross-mode **both directions** (fp16 file → int8 server, int8 file → fp16 server) →
     **HTTP 400** `{"error": {"code": 400, "message": "ValueError: slot file cache layout does not
     match the loaded cache", ...}}`. Fails **loudly**; so step 7.1 needs **no** SlotStore change,
     only docs.

---

## Remaining work

### Step 4 — speed / MTP grid (**REVISED per user: drop 16K; 3 runs total per cell**)

Cells: `(ctx 4096 / 40960) × (greedy / T=1.0) × (MTP on / off)` = **8 cells**.
Per cell, 3 runs total in the plan's alternating order trimmed to its first three,
`fp16, int8, int8` (2× int8, 1× fp16) → **24 `bench_mtp` invocations**, one at a time.
(Writes raw rows to `bench_grid.jsonl`; report decode tok/s and acceptance, and the fp16↔int8
delta, per cell. Small n — label medians as such.)

Driver prepared but not yet sent/run: `/tmp/kilo/bench_grid.py` (edit `CTXS = [4096, 40960]` and
`ORDER = [0, 8, 8]`). Equivalent shell per cell:

```bash
M=~/models/glm53-exl3-td205
for CB in 0 8 8; do
  python tools/strix_halo/bench_mtp.py -m $M -c <4096|40960> -n 200 -g        --cache-bits $CB   # greedy
  python tools/strix_halo/bench_mtp.py -m $M -c <4096|40960> -n 200 --temp 1.0 --cache-bits $CB  # T=1.0
  # plus the --no-mtp variants for the MTP-off cells
done
```
Record the MTP eh_proj sidecar absence (deviation 6) alongside acceptance.

### Step 5 — knob sweep (conditional on step 4)
Only if int8 decode is slower than fp16 beyond run-to-run noise: at 16K/40K sweep
`EXL3_MLA_QC_BN ∈ {32,64,128,256}` and `EXL3_MLA_QC_TRANS ∈ {unset,0,1}` (export before each
server/bench start); keep engine defaults unless a win reproduces. (User dropped 16K from the
grid; if step 5 still runs, use 40960 only.)

### Step 7 — fixes, docs, report (small separate commits)
1. **Slot mode guard:** confirmed loud (step 6.5) → **no code change**; document the HTTP 400
   behaviour. (Do **not** add `cache_bits` to `SlotStore`; not needed.)
2. **Row-invariance caveat (docs):** applies with `--sessions > 1` (step 6.3 showed batched rows
   differ from solo). Remove the old "does not apply to GLM" wording.
3. **`--cache-bits` help + startup log** (`tools/glm/serve.py:1409` help, `:1460` startup line):
   mention slot files do not cross cache modes.
4. **Optional warning** (`tools/glm/serve.py:1428-1432`): when `--cache-bits 8` and
   `--sessions > 1`, add one startup warning next to the existing `--sessions` note.
5. **Docs + report:** re-add the deleted paragraph to `tools/glm/SERVE.md` and the GLM line to
   `doc/models.md` **with the measured numbers above**; finish this file as
   `tools/glm/CACHE_BITS_8.md`.

`c105ba7` deleted exactly two doc fragments to restore (from `git show c105ba7`):
- `doc/models.md`: the "GLM: `--cache-bits 8` stores the MLA latent cache pages packed int8 …"
  paragraph.
- `tools/glm/SERVE.md`: the `--cache-bits` table row's "slot files do not cross modes" clause and
  the standalone paragraph ("… measured memory, PPL, MTP accept rate and decode tok/s deltas …
  are pending … that caveat does not apply to GLM's current serving shape").

### Step 8 — fp16 regression guard
At HEAD and at `18cdd3c`, fixed prompt, greedy, MTP off (`--num-draft 0`), `-c 16384`, fp16
default, through `serve.py`; compare `/v1/completions` text byte-for-byte. `fp16` is unchanged by
construction (`qkw = {}` when `cache_bits == 0`); the byte-equal text is the empirical
confirmation.

```bash
git worktree add /tmp/kyojin-base 18cdd3c
cp ~/engines/kyojin/exllamav3_ext*.so /tmp/kyojin-base/     # built ext is gitignored
```

### Validation
```bash
python -m pytest -q tests/test_mla.py tests/test_mla_dsa.py tests/test_kv_quant.py tests/test_cache8_roundtrip_cpu.py
python -m pytest -q tools/glm/test_serve.py tools/test_serve_compat.py tools/test_startup_health.py
```
(GPU quant cases in `tests/test_mla.py -k quant` run on the box; the server tests are CPU/fake.)

---

## Not run (explicit list)

- **Step 4 grid** (speed / MTP acceptance): not run — box crashed during the smoke run.
- **Step 5 knob sweep**: not run (conditional on step 4).
- **Step 8 fp16 regression**: not run.
- **Step 7 source/doc edits**: not applied yet.
- **pytest validation**: not run.
- **Empirical cache sizes at 131072 / 524288**: extrapolated only (`-c` capped at 40960).
- **MTP acceptance with the eh_proj sidecar**: sidecar absent on this box.
- **RED-SNOW 2.49bpw pack**: user reports it does not work; not measured.

---

## Reproduce (exact commands, box `m@10.10.10.2`)

```bash
# environment
cd ~/engines/kyojin && source tools/strix_halo/env.sh
git fetch origin && git reset --hard 9eb14ce        # harness commit
ln -sfn /data/hf/hub/models--NicSir--GLM-5.3-Flash-EXL3-2.72bpw/snapshots/db672fc5e61f1d61bdf6d641fb63538b80bb7613 \
        ~/models/glm53-exl3-td205
mkdir -p ~/cache/llama-slots /tmp/cb8
python tools/glm/make_corpus.py ~/bench/ppl/wiki.test.raw 400000

# step 2 (PPL)
python tools/glm/ppl7.py --out /tmp/cb8/ppl_fp16.json

# step 3 (NIAH) — start server, then:
python tools/glm/niah.py --base http://127.0.0.1:18080/v1 --model glm-5.3-exl3 \
  --model-path ~/models/glm53-exl3-td205 --tokens 40000 --out /tmp/cb8/niah_int8.json
```
