"""Rotary position embedding (RoPE), implemented from the rotation formulation.

This is the *positions* half of attention. The sibling repository
``attention-kv-cache-from-scratch`` implements softmax(QK^T/sqrt(D))V, the
MHA/MQA/GQA head layouts, and the KV cache; none of that is repeated here.
What is added here is the map applied to q and k *before* the dot product,
and what happens to it when you ask for positions the model never trained on.

The construction
----------------
Split a head-dimension-``D`` vector into ``D/2`` adjacent pairs and treat each
pair as a point in a plane. Pair ``i`` is rotated by angle ``m * theta_i``,
where ``m`` is the absolute token position and

    theta_i = base ** (-2i / D),    i = 0 .. D/2 - 1

so pair 0 turns once per token (fast) and the last pair turns once per
``2*pi*base`` tokens (slow). ``base`` (also called ``theta``) is 10000 in the
original paper and in Llama/Qwen.

Why this specific form. Write the rotation as complex multiplication: pair
``i`` of ``q`` at position ``m`` becomes ``q_i * e^{i*m*theta_i}``. Then

    <R(m) q_i, R(n) k_i> = Re[ q_i * conj(k_i) * e^{i*(m-n)*theta_i} ]

The absolute positions cancel and only ``m - n`` survives. That is the whole
point: an *absolute* transform applied independently to q and k produces a
*relative* bias in the score, with no S x S bias matrix to materialise and
nothing to add to the attention kernel.

Why it extrapolates poorly
--------------------------
The rotation is exactly periodic, so nothing numerically breaks past the
training length. What breaks is that the fast pairs have wrapped many times:
at ``m - n`` beyond the trained window, ``(m-n) * theta_0`` lands the low
dimensions at phase offsets whose *combination* across pairs never occurred
during training. The model has seen every individual angle, but not that
joint configuration, and the learned readout of it is undefined. This module
provides the measurement (``attention_score_by_offset``) rather than the
claim; ``scaling.py`` provides the three standard repairs.

Conventions. Arrays are NumPy, real, and use the *interleaved-pair* layout
(dims 2i, 2i+1 form pair i), which is the formulation in the RoPE paper.
Hugging Face implementations use a mathematically equivalent split-half
layout; ``rotate_half_style`` documents the difference.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

DEFAULT_BASE = 10000.0


def inv_freq(head_dim: int, *, base: float = DEFAULT_BASE) -> Array:
    """The ``D/2`` angular frequencies ``theta_i = base ** (-2i/D)``.

    Returned in descending order (fastest pair first), matching the pair
    ordering of the input vector. ``theta_0`` is always exactly 1.0 radian per
    token regardless of base or head dimension -- a useful anchor when
    hand-checking fixtures.
    """
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be positive and even, got {head_dim}")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    i = np.arange(head_dim // 2, dtype=np.float64)
    return base ** (-2.0 * i / head_dim)


def wavelengths(head_dim: int, *, base: float = DEFAULT_BASE) -> Array:
    """Tokens per full turn for each pair: ``2*pi / theta_i``.

    This is the quantity the scaling methods actually reason about. A pair
    whose wavelength exceeds the training context has never completed a turn
    during training, so it encodes absolute-ish position and interpolating it
    is safe. A pair whose wavelength is far shorter has wrapped thousands of
    times and only its phase matters. YaRN's whole design is a rule for
    treating those two regimes differently.
    """
    return 2.0 * np.pi / inv_freq(head_dim, base=base)


def angles(positions: Array | list[int], head_dim: int, *, base: float = DEFAULT_BASE) -> Array:
    """Rotation angle per (position, pair): shape ``(len(positions), D/2)``."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    return np.outer(pos, inv_freq(head_dim, base=base))


def rope_cos_sin(
    positions: Array | list[int],
    head_dim: int,
    *,
    base: float = DEFAULT_BASE,
    inv_freq_override: Array | None = None,
    attention_scale: float = 1.0,
) -> tuple[Array, Array]:
    """Precomputed ``(cos, sin)`` tables of shape ``(T, D/2)``.

    ``inv_freq_override`` lets ``scaling.py`` substitute a modified frequency
    vector without this module knowing anything about interpolation. That is
    the seam the whole comparison hangs on: every scaling method in this repo
    is exactly "a different ``inv_freq``, plus possibly a scalar on the
    logits". Keeping that seam narrow is what makes the methods comparable at
    all rather than three separate reimplementations.

    ``attention_scale`` is YaRN's ``1/t`` temperature factor, folded into the
    cos/sin tables so it multiplies both q and k. Applying it here rather than
    to the logits means it costs nothing at attention time -- the same trick
    the YaRN reference implementation uses.
    """
    freqs = inv_freq(head_dim, base=base) if inv_freq_override is None else inv_freq_override
    freqs = np.asarray(freqs, dtype=np.float64)
    if freqs.shape != (head_dim // 2,):
        raise ValueError(f"inv_freq must have shape ({head_dim // 2},), got {freqs.shape}")
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    ang = np.outer(pos, freqs)
    return np.cos(ang) * attention_scale, np.sin(ang) * attention_scale


def apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    """Rotate ``x`` (..., T, D) by the given tables. Interleaved-pair layout.

    Per pair ``(a, b)`` at position ``m``:

        a' = a*cos(m*theta) - b*sin(m*theta)
        b' = a*sin(m*theta) + b*cos(m*theta)

    which is the standard 2x2 rotation matrix, applied ``D/2`` times
    block-diagonally. Nothing mixes across pairs, so this is O(D) per token
    and needs no matrix multiply.
    """
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"head_dim must be even, got {d}")
    t = x.shape[-2]
    if cos.shape[-2:] != (t, d // 2) or sin.shape[-2:] != (t, d // 2):
        raise ValueError(
            f"cos/sin must be ({t}, {d // 2}); got {cos.shape} and {sin.shape}"
        )

    a = x[..., 0::2]
    b = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = a * sin + b * cos
    return out


def rope(
    x: Array,
    positions: Array | list[int],
    *,
    base: float = DEFAULT_BASE,
    inv_freq_override: Array | None = None,
    attention_scale: float = 1.0,
) -> Array:
    """Convenience wrapper: build the tables and apply them."""
    cos, sin = rope_cos_sin(
        positions,
        x.shape[-1],
        base=base,
        inv_freq_override=inv_freq_override,
        attention_scale=attention_scale,
    )
    return apply_rope(x, cos, sin)


def rotate_half_style(x: Array, cos: Array, sin: Array) -> Array:
    """The Hugging Face ``rotate_half`` layout, for reference.

    HF pairs dimension ``i`` with ``i + D/2`` instead of ``2i`` with ``2i+1``,
    and duplicates the cos/sin tables to width ``D``. The two conventions are
    related by a fixed permutation of the head dimension, so a model trained
    under one and served under the other is *not* broken -- the permutation is
    absorbed into the learned q and k projections. They are however not
    interchangeable mid-model, which is a real source of silent bugs when
    porting weights. This exists so the test suite can prove the equivalence
    rather than assert it in a comment.
    """
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    half = d // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    cos_full = np.concatenate([cos, cos], axis=-1)
    sin_full = np.concatenate([sin, sin], axis=-1)
    return x * cos_full + rotated * sin_full


def interleave_to_half_perm(head_dim: int) -> Array:
    """Index permutation mapping interleaved-pair layout to HF split-half."""
    half = head_dim // 2
    perm = np.empty(head_dim, dtype=np.int64)
    perm[:half] = np.arange(0, head_dim, 2)
    perm[half:] = np.arange(1, head_dim, 2)
    return perm


def attention_score_by_offset(
    q: Array,
    k: Array,
    offsets: Array | list[int],
    *,
    anchor: int = 0,
    base: float = DEFAULT_BASE,
    inv_freq_override: Array | None = None,
    attention_scale: float = 1.0,
) -> Array:
    """Unnormalised score ``<R(anchor+off) q, R(anchor) k>`` for each offset.

    Single query vector against a single key vector, no softmax and no
    ``1/sqrt(D)``: the object of interest is the raw dot product as a function
    of *relative* distance, which is exactly what RoPE claims to control. Under
    unmodified RoPE this is independent of ``anchor``, and the test suite
    checks that to machine precision. Under a scaling method it is still
    independent of ``anchor``, because every method here only edits
    ``inv_freq`` -- so the anchor sweep is a genuine regression test on the
    implementations, not a tautology.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    k = np.asarray(k, dtype=np.float64).reshape(-1)
    if q.shape != k.shape:
        raise ValueError(f"q and k must match: {q.shape} vs {k.shape}")
    d = q.shape[-1]
    offs = np.asarray(offsets, dtype=np.int64).reshape(-1)

    kwargs = dict(base=base, inv_freq_override=inv_freq_override,
                  attention_scale=attention_scale)
    k_rot = rope(k.reshape(1, d), [anchor], **kwargs)[0]
    q_rot = rope(q.reshape(-1, d).repeat(len(offs), axis=0), anchor + offs, **kwargs)
    return q_rot @ k_rot


def random_qk(head_dim: int, *, seed: int = 0) -> tuple[Array, Array]:
    """A fixed unit-norm q/k pair. Seeded so results files are reproducible."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(head_dim)
    k = rng.standard_normal(head_dim)
    return q / np.linalg.norm(q), k / np.linalg.norm(k)
