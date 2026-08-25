"""Run the two retrieval tasks against a real model and cache the scores.

Separated from ``generate_results.py`` on purpose. This script needs PyTorch,
Transformers and half a gigabyte of weights; the results generator needs
neither, and CI runs the generator on every push. Putting inference in the
drift gate would mean downloading a model during a lint job and, worse, would
make the committed results file depend on whether the runner's download
succeeded.

What is cached and why it is portable
-------------------------------------
The artifact holds **accuracies and token counts**, not timings. Under greedy
decoding (``do_sample=False``, float32) the model is a deterministic function
of the prompt, and the prompts are deterministic given the seed, so the
accuracy of each cell is a property of the model and the task -- not of the
machine. Two runs on different hardware agree on the scores. Wall-clock
latency does not transfer, so it is written to the gitignored raw file and
never enters the committed artifact.

Float32 rather than bfloat16 is a deliberate cost. On CPU, bf16 matmul
reductions vary with the kernel the runtime picks, which makes a borderline
greedy argmax flip between machines. Paying ~2x in time buys an artifact that
actually regenerates.

Run:  python scripts/run_eval.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from context_extension_and_long_context_eval import multihop, niah  # noqa: E402
from context_extension_and_long_context_eval.io import write_lf  # noqa: E402
from context_extension_and_long_context_eval.runner import (  # noqa: E402
    DEFAULT_MODEL,
    Runner,
    calibrate_filler,
)

RESULTS = ROOT / "results"
CACHE = RESULTS / "eval-cache.json"          # committed: accuracies only
RAW = RESULTS / "eval-raw.md"                # gitignored: timings

# Capped at 8k. The brief allows it and CPU prefill is the binding constraint:
# 16k would roughly quadruple the cost of the sweep for a point that the trend
# through 8k already establishes.
LENGTHS = (1024, 2048, 4096, 8192)
DEPTHS = (0.1, 0.5, 0.9)
N_PER_DEPTH = 4          # NIAH: 3 depths x 4 = 12 samples per length
N_MULTIHOP = 12          # matched sample count, so the CIs are comparable
N_DISTRACTORS = 4
SEED = 0


def save_cache(info, rows: list[dict], lengths_done: list[int]) -> None:
    """Write the committed artifact. Called after every length, not just at the end.

    The first version of this script wrote the cache once, after the whole
    sweep. An 8k tier interrupted near the end therefore destroyed four
    completed tiers of work, and the only surviving record was a truncated
    stdout log -- from which partial data could have been reconstructed by
    hand, which is exactly how a results file ends up with numbers nobody can
    regenerate. Checkpointing per length means the artifact is always a
    complete description of the lengths it claims to cover.

    ``lengths_done`` is recorded rather than ``LENGTHS`` so the config never
    advertises a length whose rows are absent.
    """
    RESULTS.mkdir(exist_ok=True)
    # Committed artifact: scores only, latency stripped.
    portable = [{k: v for k, v in r.items() if k != "latency_s"} for r in rows]
    payload = {
        "schema": 1,
        "model": asdict(info),
        "config": {
            "lengths": lengths_done,
            "depths": list(DEPTHS),
            "n_per_depth": N_PER_DEPTH,
            "n_multihop": N_MULTIHOP,
            "n_distractors": N_DISTRACTORS,
            "seed": SEED,
            "decoding": "greedy",
            "max_new_tokens": 12,
        },
        "rows": portable,
    }
    write_lf(CACHE, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()

    print(f"loading {args.model} ...", flush=True)
    runner = Runner(args.model, threads=args.threads)
    info = runner.info()
    print(f"advertised context: {info.advertised_context}, rope_theta {info.rope_theta}")

    rows: list[dict] = []
    done: list[int] = []
    for target in LENGTHS:
        # Calibrate each task separately: the two prompts have different
        # boilerplate and different planted lines, so equal filler counts
        # would give unequal token lengths and the curves would no longer be
        # sampled at the same x.
        n_niah = calibrate_filler(
            target,
            lambda n: niah.build_sample(n_filler=n, depth=0.5, seed=SEED).prompt,
            runner.n_tokens,
        )
        n_multi = calibrate_filler(
            target,
            lambda n: multihop.build_sample(
                n_filler=n, n_distractors=N_DISTRACTORS, seed=SEED
            ).prompt,
            runner.n_tokens,
        )
        print(f"[{target}] filler lines: niah={n_niah} multihop={n_multi}", flush=True)

        for s in niah.build_suite(
            n_filler=n_niah, depths=DEPTHS, n_per_depth=N_PER_DEPTH, seed=SEED
        ):
            text, n_in, secs = runner.generate(s.prompt)
            rows.append(
                dict(
                    task="niah",
                    target_tokens=target,
                    actual_tokens=n_in,
                    depth=s.depth,
                    correct=niah.score(text, s),
                    bridge_found=0.0,
                    latency_s=round(secs, 3),
                )
            )
            print(f"  niah  d={s.depth} tok={n_in} -> {rows[-1]['correct']}", flush=True)

        for s in multihop.build_suite(
            n_filler=n_multi, n_samples=N_MULTIHOP, n_distractors=N_DISTRACTORS, seed=SEED
        ):
            text, n_in, secs = runner.generate(s.prompt)
            rows.append(
                dict(
                    task="multihop",
                    target_tokens=target,
                    actual_tokens=n_in,
                    depth=-1.0,
                    correct=multihop.score(text, s),
                    bridge_found=multihop.scores_bridge_only(text, s),
                    latency_s=round(secs, 3),
                )
            )
            print(f"  multi tok={n_in} -> {rows[-1]['correct']}", flush=True)

        # Checkpoint. A kill during a later tier now costs only that tier.
        done.append(target)
        save_cache(info, rows, done)
        print(f"[{target}] checkpointed {len(rows)} rows", flush=True)

    print(f"wrote {CACHE}")

    lines = [
        "# Raw evaluation timings (gitignored)",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Machine: {platform.platform()}",
        f"- Processor: {platform.processor()}",
        f"- Model: {info.name}",
        "",
        "| task | target | actual tokens | latency s |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r['task']} | {r['target_tokens']} | {r['actual_tokens']} | {r['latency_s']:.2f} |"
        for r in rows
    ]
    write_lf(RAW, "\n".join(lines) + "\n")
    print(f"wrote {RAW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
