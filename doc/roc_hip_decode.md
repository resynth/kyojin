# Experimental gfx12 ROCm decode backend

This page documents the implemented Heterogeneous-compute Interface for Portability (HIP) decode path. It accelerates EXL3 matrix-vector multiplication (GEMV) on RDNA4 and provides correctness fallbacks outside that envelope.

## What runs on gfx12

The runtime gate in `exllamav3.modules.quant.exl3.LinearEXL3` sends a layer to HIP GEMV only when:

- ROCm is active and `ext.exl3_gemv_supported(device)` is true;
- the GPU is `gfx1200` or `gfx1201`;
- the decode batch is `rows <= 16` for the HIP GEMV route, subject to the existing K/codebook constraints; rows 9..16 are only for the supported native MMODE-2 variants (K3/mul1, K4/mcg or mul1, K5/mul1, K6/mcg or mul1), while K2 and all other codebook variants retain the rows<=8 limit; unsupported combinations fall back;
- `in_features` and `out_features` are both multiples of 128;
- `K` is from 2 through 6;
- the codebook flags match an instantiated family: every `K` in 2..6 has the plain
  procedural codebook (`cb=0`), MCG (`cb=1`) and MUL1 (`cb=2`) variants for
  `rows <= 8`. The 9..16-row MMODE-2 variants are the native subset only
  (K3/mul1, K4/mcg or mul1, K5/mul1, K6/mcg or mul1); all other combinations
  retain the `rows <= 8` limit and fall back above it.

On those gfx12 devices, `m == 1` now uses the RDNA fdot2 dot core
(`rocm/quant/exl3_gemv_rdna.cu`): one warp per 16-wide output tile, decoded
through `V_DOT2_F32_F16`, with the hand-laid tiles decoder. Rows 2..16 keep the
`v_wmma_f32_16x16x16_f16` kernel with fp32 accumulation; that is still the only
WMMA path here, not a general matrix fused multiply-add (MFMA) backend.

## What falls back

Anything outside the envelope above uses the existing reconstruct path:

- rows `> 16`
- unsupported bitrate / codebook combinations (K outside 2..6, half-integer
  bitrates without MUL1, rows 9..16 for a non-MMODE-2 K/codebook pair)
- `EXL3_GEMV=0`
- non-gfx12 ROCm devices

The fallback is `reconstruct + hgemm`. That path is correctness-first. It is not a fused performance kernel, and it does not try to match CUDA decode throughput.

The rest of the ROCm shims in `ext_fallbacks.py` follow the same rule: they preserve behavior with PyTorch implementations for correctness, not speed.

## Why the HIP launches are split

HIP does not use the CUDA cooperative-grid decode path.

Rows 2..16 run three stream-ordered launches: input Hadamard, WMMA GEMV, output
Hadamard. For `m == 1` on gfx1200/gfx1201 the middle launch is the RDNA fdot2
dot core and the output Hadamard is fused into its store when
`out_features / 128 <= 1024`; the 248,320-wide lm_head keeps the separate output
launch. The earlier cooperative-launch experiment was removed after the
Heterogeneous System Architecture (HSA) runtime crashed during shutdown. CUDA
keeps its separate cooperative path with `grid.sync()`.

## Why LDS staging is mandatory on gfx12

The gfx12 implementation stages decoded trellis words and wave matrix multiply-accumulate (WMMA) operands through warp-private local data share (LDS).

That is not an optimization choice. On gfx12, feeding WMMA from the shuffle path caused hardware exceptions, so the HIP kernel disables that path.

## What is still deferred

Out of scope for this backend slice:

- large-throughput matrix-matrix multiplication (GEMM, `rows > 144`)
- int8 GEMV
- CDNA MFMA
- the quantizer
- CUDA graphs / `bc_*`

## Validation snapshot

Current local verification on gfx1201 includes:

- 98 GEMV matrix, routing, codebook, K5, and K6 tests per model fixture
- 88 cache / reconstruct tests
- 61 multi-head latent attention (MLA) / DeepSeek sparse attention (DSA) tests
- 34 native ROCm hyperconnection route, numerical, stream, validation, and fallback tests
- 63 grouped K3 MoE route, numerical, multirow, safety, LoRA, and fallback tests, plus opt-in full Flash and MTP oracles
- 44 native ROCm shared-gate fusion, stream, validation, and fallback tests
- 25 gfx12 batch-one router numerical, tie, stream, validation, fallback, and real-layer tests
- forced-identical-context 16-step logits oracles for mul1, MCG, and plain-codebook models

The Qwen3.8-27B mul1 oracle reports:

- top-1 agreement: 16/16
- top-5 overlap: at least 4/5 tokens at every step
- observed max absolute logit delta: 0.488281
- worst-step mean delta: 0.046886
- overall mean delta: 0.027793
- fallback-only `-inf` positions carry <1e-6 softmax mass per step

The Qwen3.5-9B MCG oracle reports:

- top-1 agreement: 16/16
- exact top-5 agreement: 16/16 steps
- max absolute logit delta: 0.234375
- worst-step mean delta: 0.017942
- overall mean delta: 0.009152

## Plain procedural codebook (cb0)

Packs that store no `mcg`/`mul1` tensors (the OraQuant 9B fixture is one) use the plain
procedural codebook. K2/K3/K5/K6 now have `cb=0` instantiations for the
`rows <= 8` configs (narrow and wide, fp16 and fp32 output), so their projections no
longer reconstruct the weight every token. `exl3_gemv_cfg` and
`exl3_gemv_try_launch` keep the `cb=0` restriction only on the CUDA branch; the
Python route requires `2 <= K <= 6` and keeps the K/rows/shape checks.

The Qwen3.5-9B Ora oracle (all dense projections through the GEMV) reports:

- route census on a 32-token decode: 6,629 `exl3_gemv` calls and zero
  `reconstruct`/`reconstruct_slice`, i.e. ~207 GEMV launches per token; before the
  expansion the same run made ~6,100 reconstruct calls
- 16 steps: top-1 16/16, top-5 minimum overlap 4/5, max absolute logit delta
  0.058594, worst-step mean 0.008850, overall mean 0.003390;
  route proof `gemv=3413, reconstruct=0, reconstruct_slice=0`
- 128 steps: top-1 128/128 on gfx1201 with max delta 8.113281, worst-step mean
  1.214331, overall mean 0.060761 (and 128/128 on gfx1151, whose route is still
  WMMA). Raw logit deltas grow with the trace because the hybrid GDN recurrence
  feeds the numerically different linears back into its cache state; the same
  growth appears on gfx11.5, so it is the fixture rather than the gfx12 path. The
  forced-token top-1 identity is the hard signal; the tight numeric bounds stay on
  the 16-step arm.

Warmed decode on the same pack and prompt, three trials after warmup:

| Device | All-reconstruct baseline | Plain-codebook GEMV | Uplift |
| --- | ---: | ---: | ---: |
| gfx1201 (RX 9070), 128 tokens | 16.88 tok/s | 76.8 tok/s | 4.55x |
| gfx1151 (8060S), 64 tokens | 5.09 tok/s | 37.4 tok/s | 7.35x |

The same pack on upstream exllamav3 v1.6.6 (its `BC_LinearEXL3` HIP path) measured
89.8 tok/s on the same RX 9070. The RDNA fdot2 port closed the kernel gap: a
streaming pass over all 201 dense linears takes 10.2-10.3 ms here (260 GB/s),
versus 10.8 ms for upstream's direct `exl3_gemv` and 9.93 ms for its `exl3_gemm`
route. The remaining ~1.5 ms/token is host-side: upstream executes whole blocks
through its C++ `BC_*` graph modules, while this tree still runs the dense Python
modules eagerly (about 14.6 ms CPU per token against 12.9 ms wall). Closing it
needs a BC-block port or a graph-safe dense-block path, not another GEMV config.

A cfg sweep was performed with a temporary in-kernel `EXL3_GEMV_CFG` hook because
`ext.exl3_gemv` passes `force=true` and pins mode 2; the hook was removed before
merge.

Chat-path quality (8 prompts: factual, arithmetic, code, instruction-following,
long-form; greedy and default sampling; `tools/strix_halo/coherence_suite.py`):
all 16 runs reached EOS, no loop-guard fire, no 20-gram repeated more than twice.
The raw-completion loop seen with `load_qwen.py` does not reproduce through the
chat template; use `tools/qwen/serve.py` (template + stop conditions + loop guard)
as the supported chat path.

Large-pack coverage was checked from the safetensors headers (neither fits the
16 GB card, so this is eligibility only, not a run): the 102 GB Qwen3.8-Flash-Next
pack is mul1 with dense K4/K5/K6 (K8 tensors stay on reconstruct), and the 92 GB
GLM-5.3 pack is entirely mul1 with K2/K3/K4/K5/K6. Both are covered by the existing
mul1 instantiations; the plain-codebook expansion changes nothing for them.

Warmed decode measurements on gfx1201 (MCG / mul1 fixtures; these are not the Ora
plain-codebook pack above and must not be used as its baseline):

| Model | Trial size | K6 reconstruct median | Direct K6 median | Uplift |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 64 tokens | 14.569 tok/s | 16.250 tok/s | 11.5% |
| Qwen3.5-9B | 128 tokens | 36.522 tok/s | 47.311 tok/s | 29.5% |

Each median is three trials after a 32-token warmup, using one model, prompt, and GPU. The K6-reconstruct arm preserves direct GEMV for K2-K4 and falls back only for K6.

Reusing each assembled WMMA A fragment across adjacent N tiles provides another measured gain:

| Model | Before A reuse | With A reuse | Uplift |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B | 16.315 tok/s | 17.325 tok/s | 6.2% |
| Qwen3.5-9B | 47.315 tok/s | 51.917 tok/s | 9.7% |

Specializing the m=1 reduction for its single valid output row raises Qwen3.8-27B from 17.285 to 18.115 tok/s (4.8%). Qwen3.5-9B rises from 51.880 to 54.411 tok/s (4.9%). This reduces kernel LDS use from 28,672 to 14,336 bytes and raises occupancy from two to four blocks per compute unit.

Compared with the earlier all-reconstruct baseline, the current backend measures 3.94x faster on Qwen3.8-27B (4.603 to 18.115 tok/s). Qwen3.5-9B measures 3.66x faster (14.847 to 54.411 tok/s). These cross-run ratios show the cumulative backend improvement; the tables and percentages above are controlled incremental comparisons.

Steady-state profiles after prefill show the K6 vocabulary head changing from reconstruct plus rocBLAS hgemm to one direct kernel:

| Model | Steps | K6 reconstruct + hgemm | Direct K6 | Total self GPU time change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 8 | 80.543 ms | 18.791 ms | 483.397 ms → 423.119 ms |
| Qwen3.5-9B | 12 | 97.071 ms | 21.174 ms | 292.792 ms → 217.324 ms |

A-fragment reuse reduces total self GPU time from 423.119 to 400.882 ms for the eight-step Qwen3.8 profile. The dominant K4 mul1/fp16 wide kernel falls from 163.596 to 154.410 ms. For the twelve-step Qwen3.5 profile, time falls from 217.324 to 197.155 ms. Its K4 MCG/fp32 narrow kernel falls from 67.561 to 57.072 ms.

The one-row reduction lowers the Qwen3.8 wide K4 kernel again, from 154.413 to 140.053 ms in a controlled profile. The corresponding Qwen3.5 wide K4 kernel falls from 51.207 to 42.012 ms; its narrow kernel is effectively unchanged.

## Qwen3.8-Flash-Next target

The `turboderp/Qwen3.8-Flash-Next-exl3` 3.05-bpw checkpoint runs target-only with a layer split across two gfx1201 GPUs. Its 32.64-GB PLE n-gram table stays file-backed and streams only selected rows. A balanced 27/27-GB load budget allocated approximately 28.6 and 24.4 GB on the two GPUs with a 32K cache.

A short greedy smoke produced `Paris. Paris is the capital` and executed the direct EXL3 route. Adding K5 support moves the model's high-quality shared-expert and other K5 projections off reconstruct+hgemm:

- warmed target-only decode: 6.588 to 7.687 tok/s, a 16.7% gain;
- four-token self-device profile: 875.654 to 752.568 ms, a 14.1% reduction;
- K5 reconstruction: 1,232 launches to zero;
- real shared-expert gate/up/down K5 projections match reconstruct+hgemm on both gfx1201 devices.

The existing fused GatedResidual and hyperconnection kernels are also available on wave32 ROCm devices. They replace the decode reference path's small matrix multiplications and elementwise chains with two `gr_mix` launches and one in-place `hc_apply` launch per residual site. Unsupported wave sizes retain the PyTorch path. In a controlled same-process comparison, fusion improved the 32-token median from 7.787 to 14.085 tok/s. Over four tokens, it reduced `aten::mm` from 1,552 to 392 calls and removed 7,736 HIP launches. A separate fresh-process run measured a 13.553 tok/s median.

Keeping the PLE table in RAM did not improve that fresh-process result: disk streaming measured 13.553 tok/s versus 13.410 tok/s from RAM. RAM residency also increased model load time from 17.0 to 30.9 seconds. The streamed path remains the recommended default and avoids reserving 32.64 GB of host memory.

This is a foundation result, not the final serving profile. Qwen4Exp currently uses layer split rather than tensor parallelism. MTP has not been enabled, and sparse QSA beyond the short-context dense threshold still needs ROCm qualification. The remaining short-context bottlenecks are launch count and K3 expert GEMV. Prefill and long-context performance need separate measurements before comparison with sparse QSA serving results.

## MCG compatibility

The Qwen3.5-9B fixture exposed an incorrect portable translation of the original PTX `lop3` expression. The fix restores the exact LUT `0x6a` semantics for plain and MCG procedural codebooks. A zero-state reconstruction test now checks the codebook value independently of both GEMV and the reconstruct implementation.

## Narrow skinny GEMM fix

The ROCm `hgemm` skinny path (m <= 32, n <= 128) sized its grid for two adjacent
columns per lane and then dispatched the one-column padded kernels for m=9..32.
At n=128 that left columns 64..127 unwritten, and because `reconstruct_hgemm`
passes an uninitialized output, repeated reconstruct calls differed. `hgemm.cu`
now restricts the two-column form to `m <= 8`, so rows 9..32 use the matching
one-column grid. Verified at m=9/n=128 (all 128 columns written) and by the
MMODE-2/K5 synthetic tests at n=128.

## Build and test

```sh
cd exllamav3
export HIPCXX=hipcc PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=4
pip install . --no-build-isolation
python -c "from exllamav3 import ext; assert ext.exllamav3_ext.exl3_gemv_supported(0)"

EXL3_TEST_MODEL=/path/to/Qwen3.8-27B-exl3 \
  python -m pytest tests/test_hip_gemv_decode.py -q

EXL3_TEST_MODEL=/path/to/Qwen3.8-27B-exl3 \
  python tests/hip_gemv_logits_oracle.py

# Repeat with an MCG fixture.
EXL3_TEST_MODEL=/path/to/Qwen3.5-9B-exl3 \
  python tests/hip_gemv_logits_oracle.py

```