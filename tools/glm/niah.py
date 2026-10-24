#!/usr/bin/env python
"""Needle-in-a-haystack recall driver for the GLM-5.3 EXL3 server.

Only the server accepts --cache-bits, so quantised-cache retrieval has to be driven over HTTP.
For every (depth, trial) this builds a ~target-token haystack, plants a random reference value at
that depth, asks what the value is (greedy), and checks whether it comes back.

The haystack builder follows examples/chat_util.py:make_haystack_prompt (lorem filler, a
"reference value" needle, an answer prompt) but fixes the needle depth and the seed per trial, so
depths are exact and runs are reproducible. HTTP uses tools/glm/serve_accept.py:post.

usage:
  python tools/glm/niah.py --base http://127.0.0.1:18080/v1 --model glm-5.3-exl3 \
      --model-path ~/models/glm53-exl3-td205 --tokens 40000 \
      --depths 0.1,0.5,0.9 --trials 3 --seed 1234 --out /tmp/cb8/niah_fp16.json
"""
import argparse, json, random, string, time
from pathlib import Path

LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, "
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
    "consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse "
    "cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non "
    "proident, sunt in culpa qui officia deserunt mollit anim id est laborum.\n\n"
)


def make_prompt(tok, target_tokens, depth, ref):
    """Binary-search the character count whose tokenized length is nearest target_tokens."""
    def build(nchars):
        hay = (LOREM * (nchars // len(LOREM) + 1))[:nchars]
        pos = max(0, min(len(hay), int(len(hay) * depth)))
        return (
            hay[:pos] +
            f"\n\nThe reference value is: {ref}\n\n" +
            hay[pos:] +
            "\n\nWhat is the reference value mentioned in the text? "
            "Reply with only the value."
        )

    def ntok(p):
        return int(tok.encode(p, add_bos=False, encode_special_tokens=True).shape[-1])

    lo, hi = 0, max(1, target_tokens * 8)
    best, bestn = build(0), ntok(build(0))
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = build(mid)
        n = ntok(cand)
        if abs(n - target_tokens) < abs(bestn - target_tokens):
            best, bestn = cand, n
        if n < target_tokens:
            lo = mid + 1
        elif n > target_tokens:
            hi = mid - 1
        else:
            break
    return best, bestn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18080/v1")
    ap.add_argument("--model", default="glm-5.3-exl3")
    ap.add_argument("--model-path", default="~/models/glm53-exl3-td205",
                    help="tokenizer source (no GPU used)")
    ap.add_argument("--tokens", type=int, default=40000)
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import os
    from exllamav3 import Config, Tokenizer
    cfg = Config.from_directory(os.path.expanduser(a.model_path))
    tok = Tokenizer.from_config(cfg)

    from serve_accept import post
    rng = random.Random(a.seed)
    alpha = string.ascii_letters + string.digits
    res = {"tokens_target": a.tokens, "depths": a.depths, "trials": a.trials,
           "seed": a.seed, "cells": []}
    for depth in [float(x) for x in a.depths.split(",")]:
        hits = 0
        for trial in range(a.trials):
            ref = "".join(rng.choices(alpha, k=16))
            prompt, ntok = make_prompt(tok, a.tokens, depth, ref)
            body = {"model": a.model, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": a.max_tokens, "temperature": 0}
            t0 = time.perf_counter()
            d = json.load(post(a.base, body))
            wall = time.perf_counter() - t0
            ch = (d.get("choices") or [{}])[0].get("message") or {}
            text = (ch.get("content") or "") + (ch.get("reasoning_content") or "")
            hit = ref.lower() in text.lower()
            hits += int(hit)
            res["cells"].append({"depth": depth, "trial": trial, "ref": ref, "hit": hit,
                                 "prompt_tokens": d.get("usage", {}).get("prompt_tokens"),
                                 "chars": len(prompt), "tok_est": ntok, "wall_s": round(wall, 2),
                                 "reply": text[:200]})
            print(f"[niah] depth={depth} trial={trial} hit={hit} "
                  f"prompt_tokens={d.get('usage', {}).get('prompt_tokens')} wall={wall:.1f}s", flush=True)
        res.setdefault("recall", {})[str(depth)] = f"{hits}/{a.trials}"
        print(f"[niah] depth={depth} recall {hits}/{a.trials}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
