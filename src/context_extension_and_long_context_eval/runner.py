"""Model execution for the two retrieval tasks. Optional, CPU-only, greedy.

Kept in its own module and behind a lazy import so that ``rope.py``,
``scaling.py``, ``niah.py`` and ``multihop.py`` -- and every test over them --
stay importable with nothing but NumPy. The CI runner has no model cache and
no reason to acquire one: the committed results file is produced from the
closed-form analysis plus a cached measurement artifact, never by downloading
half a gigabyte of weights during a lint job.

Decoding is greedy (``do_sample=False``) so a run is reproducible from the
prompt alone. Sampling would put a temperature-shaped confidence interval
around every cell of the results table for no benefit -- the task has one
correct four-digit string.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 12


@dataclass(frozen=True)
class ModelInfo:
    """Provenance for the results file. Everything needed to reproduce a run."""

    name: str
    revision: str
    advertised_context: int
    rope_theta: float
    head_dim: int
    n_layers: int
    dtype: str


@dataclass(frozen=True)
class Result:
    """One scored sample."""

    task: str
    target_tokens: int
    actual_tokens: int
    depth: float
    correct: float
    bridge_found: float
    latency_s: float


class Runner:
    """Thin wrapper over a Hugging Face causal LM. Constructed once per run."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, threads: int = 24) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        self._torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()

    def info(self) -> ModelInfo:
        cfg = self.model.config
        return ModelInfo(
            name=self.model_name,
            revision=str(getattr(cfg, "_commit_hash", "unknown")),
            advertised_context=int(cfg.max_position_embeddings),
            rope_theta=float(getattr(cfg, "rope_theta", 10000.0)),
            head_dim=int(cfg.hidden_size // cfg.num_attention_heads),
            n_layers=int(cfg.num_hidden_layers),
            dtype="float32",
        )

    def n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text).input_ids)

    def generate(self, prompt: str) -> tuple[str, int, float]:
        """Greedy decode. Returns (completion, prompt_tokens, seconds).

        The prompt goes through the chat template because the model is an
        instruct checkpoint: feeding a raw string to a chat-tuned model
        measures the template mismatch as much as the context length, which
        would be a confound in exactly the direction that makes long context
        look worse than it is.
        """
        torch = self._torch
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer(text, return_tensors="pt")
        n_in = int(enc.input_ids.shape[1])

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        completion = self.tokenizer.decode(out[0, n_in:], skip_special_tokens=True)
        return completion, n_in, elapsed


def calibrate_filler(
    target_tokens: int,
    build_prompt: Any,
    count_tokens: Any,
    *,
    lo: int = 1,
    hi: int = 40_000,
) -> int:
    """Binary-search the filler-line count that hits ``target_tokens``.

    Necessary because the x-axis of the results table must be *tokens*, not
    lines. Filler sentences differ in token length, so a fixed line count
    gives a different context length for each task and the two curves would no
    longer be measured at comparable points -- which is the one thing this
    repo's headline claim depends on.

    Deterministic: same target in, same count out, on any machine, because it
    only calls the tokenizer.
    """
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(build_prompt(mid)) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return lo


def result_rows(results: list[Result]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]
