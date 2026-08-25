"""The other position encodings, and why RoPE won the argument.

This module exists to make one contrast precise: RoPE needs scaling methods
because of a specific design choice, and the alternatives show what the
choice bought. Each is implemented far enough to expose its extrapolation
behaviour, which is the axis this repo cares about. None is a full
implementation, and none is used by the evaluation -- Qwen2.5 is a RoPE model.

    learned      A trainable embedding per position, added to the token
                 embedding. GPT-2 does this. Extrapolation is not merely poor,
                 it is *undefined*: there is no row in the table for position
                 2049 in a 2048-trained model. Any answer requires inventing a
                 vector. This is the cleanest illustration that "context
                 length" can be a hard architectural boundary rather than a
                 quality gradient.

    sinusoidal   Fixed sin/cos of position at geometrically spaced
                 frequencies, added to the embedding. The original
                 Transformer. Defined at every position, so it extrapolates in
                 the sense of not crashing -- but the *sum* of a position
                 vector and a token vector is entangled, and the network's
                 learned readout of that sum has never seen the far-position
                 values. In practice it degrades badly past the training
                 length. RoPE's key improvement is that it *rotates* rather
                 than *adds*, which is why relative offsets survive the dot
                 product cleanly.

    alibi        No position vectors at all. A linear penalty ``-m * |i - j|``
                 is added directly to the attention logits, with a fixed
                 per-head slope ``m``. Extrapolates by construction: the
                 penalty is defined for any distance, and because it is
                 monotone in distance the *ordering* of scores by recency is
                 preserved at any length. What it gives up is the ability to
                 attend far away at all -- the penalty grows without bound, so
                 a head with a steep slope has an effective window regardless
                 of the advertised one. ALiBi trades peak long-range capacity
                 for graceful degradation, which is the opposite of the trade
                 RoPE makes.

    nope         No positional information whatsoever in a decoder. Not the
                 absurdity it sounds like: causal masking alone breaks the
                 permutation symmetry, because token ``i`` sees ``i`` previous
                 tokens and token ``j`` sees ``j``, so the *count* of visible
                 tokens encodes position implicitly. Small decoder-only models
                 trained this way are competitive and reported to extrapolate
                 better than learned encodings. It is the strongest evidence
                 that explicit position encoding is a convenience the
                 architecture exploits rather than a logical necessity.

The recurring theme is that extrapolation and long-range capacity pull against
each other, and every encoding picks a point on that trade-off. RoPE picks
"maximum capacity, poor extrapolation", which is precisely why an ecosystem of
scaling methods grew around it and not around ALiBi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def sinusoidal_encoding(n_positions: int, d_model: int, *, base: float = 10000.0) -> Array:
    """The original Transformer's additive table. Shape ``(n_positions, d_model)``.

    ``PE[pos, 2i] = sin(pos / base^(2i/d))`` and ``PE[pos, 2i+1] = cos(...)``.

    Note the frequency ladder is the same one RoPE uses. The difference is
    entirely in how it is applied: added to the embedding here, used as a
    rotation angle there. That single change is what moves the encoding from
    absolute to relative.
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")
    pos = np.arange(n_positions, dtype=np.float64)[:, None]
    i = np.arange(d_model // 2, dtype=np.float64)[None, :]
    ang = pos * base ** (-2.0 * i / d_model)
    out = np.empty((n_positions, d_model), dtype=np.float64)
    out[:, 0::2] = np.sin(ang)
    out[:, 1::2] = np.cos(ang)
    return out


def alibi_slopes(n_heads: int) -> Array:
    """ALiBi's per-head slopes: a geometric sequence starting at ``2^(-8/n)``.

    For a power-of-two head count this is exactly ``2^-1, 2^-2, ...`` scaled so
    the sequence spans a useful range of effective windows. Steep-slope heads
    become strongly local; shallow-slope heads retain long reach. The set of
    slopes is what gives the model a spread of receptive fields for free,
    without any of them being learned.
    """
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}")
    start = 2.0 ** (-8.0 / n_heads)
    return start ** np.arange(1, n_heads + 1, dtype=np.float64)


def alibi_bias(n_heads: int, q_len: int, k_len: int | None = None) -> Array:
    """Additive attention bias, shape ``(n_heads, q_len, k_len)``.

    ``bias[h, i, j] = -slope[h] * (i - j)`` for ``j <= i``. Because this is
    added to the logits rather than mixed into q and k, it needs no change to
    the attention maths at all -- which is why ALiBi was easy to adopt and why
    it composes with any attention kernel.
    """
    if k_len is None:
        k_len = q_len
    slopes = alibi_slopes(n_heads)[:, None, None]
    i = np.arange(q_len, dtype=np.float64)[None, :, None] + (k_len - q_len)
    j = np.arange(k_len, dtype=np.float64)[None, None, :]
    return -slopes * np.maximum(i - j, 0.0)


def alibi_effective_window(slope: float, *, logit_budget: float = 8.0) -> float:
    """Distance at which ALiBi's penalty exhausts a given logit budget.

    A crude but honest proxy for a head's reach: once the penalty exceeds the
    spread of the content logits, the softmax weight on that position is
    negligible no matter how well the content matches. ``logit_budget`` is the
    assumed spread. The number to take away is not the value but the *scaling*
    -- the window is a fixed multiple of ``1/slope`` and does not grow when you
    serve the model at a longer context. That is exactly the property RoPE
    lacks and ALiBi guarantees.
    """
    if slope <= 0:
        raise ValueError(f"slope must be positive, got {slope}")
    return logit_budget / slope


def nope_visible_counts(n_positions: int) -> Array:
    """Number of tokens each position can attend to under a causal mask.

    ``[1, 2, 3, ...]``. This is the entire positional signal available to a
    NoPE decoder, and it is strictly monotone -- which is enough for the
    network to recover ordering. Included because the argument is much more
    convincing as one line of code than as a paragraph.
    """
    return np.arange(1, n_positions + 1, dtype=np.float64)
