"""Position-scaling methods, as edits to the RoPE frequency vector.

Every method here is the same shape of intervention: take the trained
``inv_freq`` and return a modified one, optionally with a scalar temperature
on the logits. That framing is not a simplification for teaching -- it is
what the reference implementations do -- and it is what makes the four
methods numerically comparable in one table instead of four.

The problem being solved
------------------------
A model trained on ``L`` tokens has only ever seen relative offsets in
``[0, L)``. Serve it at ``s*L`` and the fast RoPE pairs are asked for phases
in a joint configuration that never occurred in training. Perplexity does not
degrade gracefully; it explodes. The three repairs differ in *which* pairs
they are willing to distort.

    none  (extrapolation)  change nothing. Positions beyond L are out of
                           distribution for every pair at once.

    pi    (position interp) divide every position by s -- equivalently divide
                           every theta by s. Position 8192 now presents the
                           angles the model learned for position 2048, so
                           nothing is out of distribution. The cost is
                           uniform: the fastest pair, which carried adjacent-
                           token resolution, is now s times blunter. Local
                           precision is traded for global range. Requires
                           fine-tuning to recover; the Chen et al. paper
                           reports ~1000 steps.

    ntk   (NTK-aware)      raise the base instead: base' = base * s^(D/(D-2)).
                           This is *not* uniform. Because theta_i is a power
                           law in the base, changing the base barely moves the
                           fast pairs (theta_0 = 1 exactly, at any base) while
                           moving the slow pairs by nearly the full factor s.
                           High-frequency resolution is preserved and the
                           interpolation is spent where wavelengths are long
                           anyway. The exponent D/(D-2) rather than 1 is
                           chosen so the *slowest* pair is scaled by exactly
                           s; with exponent 1 it would fall short. Usable
                           without fine-tuning, which is why it spread
                           through the open-weights community first.

    yarn  (YaRN)           make the fast/slow split explicit rather than
                           implicit. Classify each pair by how many turns its
                           wavelength completes inside the training context:
                           r_i = L / wavelength_i.
                             r_i > beta (=32): many turns, high frequency ->
                                 do not interpolate at all (extrapolate).
                             r_i < alpha (=1): under one turn, low frequency
                                 -> interpolate fully, like PI.
                             between: ramp linearly between the two.
                           Then multiply the logits by 1/t with
                           t = 0.1*ln(s) + 1, because spreading positions over
                           a wider range lowers the expected magnitude of the
                           attention logits and the softmax gets flatter. That
                           temperature term is the part people drop when
                           reimplementing YaRN, and it is a measurable
                           fraction of the benefit.

LongRoPE goes one step further -- a per-dimension rescale found by
evolutionary search rather than a closed form. It is out of scope here
because it needs a search budget and a validation corpus, and this repo is
deliberately CPU-only and closed-form. It is named so the omission is a
choice rather than an oversight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .rope import DEFAULT_BASE, inv_freq

Array = NDArray[np.float64]

METHODS = ("none", "pi", "ntk", "yarn")

YARN_ALPHA = 1.0
YARN_BETA = 32.0


@dataclass(frozen=True)
class ScaledRope:
    """A scaling method's full output: modified frequencies plus temperature.

    ``attention_scale`` is ``sqrt(1/t)`` and is applied to *both* q and k, so
    the product on the logit is ``1/t``. Only YaRN sets it away from 1.0.
    """

    method: str
    scale: float
    inv_freq: Array
    attention_scale: float
    base: float
    head_dim: int
    train_ctx: int

    @property
    def wavelengths(self) -> Array:
        return 2.0 * np.pi / self.inv_freq

    @property
    def effective_positions(self) -> Array:
        """Per pair, the position that now presents the trained angle.

        ``inv_freq_scaled / inv_freq_base``. A value of 1.0 means the pair was
        left alone (pure extrapolation); ``1/s`` means it was fully
        interpolated. Reading this column is the fastest way to see what a
        method actually did, and it is the column that separates NTK and YaRN
        from PI at a glance.
        """
        return self.inv_freq / inv_freq(self.head_dim, base=self.base)


def scaled_rope(
    method: str,
    *,
    scale: float,
    head_dim: int,
    train_ctx: int,
    base: float = DEFAULT_BASE,
    alpha: float = YARN_ALPHA,
    beta: float = YARN_BETA,
) -> ScaledRope:
    """Build the modified frequency vector for one method at one scale."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if scale < 1.0:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if train_ctx <= 0:
        raise ValueError(f"train_ctx must be positive, got {train_ctx}")

    base_freq = inv_freq(head_dim, base=base)
    att_scale = 1.0

    if method == "none":
        freq = base_freq.copy()
    elif method == "pi":
        freq = base_freq / scale
    elif method == "ntk":
        # base * s^(D/(D-2)); see module docstring for why the exponent.
        if head_dim <= 2:
            raise ValueError("NTK-aware scaling needs head_dim > 2")
        new_base = base * scale ** (head_dim / (head_dim - 2))
        freq = inv_freq(head_dim, base=new_base)
    else:  # yarn
        freq = _yarn_inv_freq(
            base_freq, scale=scale, train_ctx=train_ctx, alpha=alpha, beta=beta
        )
        att_scale = math.sqrt(1.0 / yarn_temperature(scale))

    return ScaledRope(
        method=method,
        scale=float(scale),
        inv_freq=freq,
        attention_scale=att_scale,
        base=base,
        head_dim=head_dim,
        train_ctx=train_ctx,
    )


def yarn_temperature(scale: float) -> float:
    """YaRN's ``t = 0.1*ln(s) + 1``. Logits are multiplied by ``1/t``.

    At s=1 this is exactly 1.0, so YaRN degenerates to plain RoPE with no
    temperature discontinuity -- which is the property that lets a served
    model switch scales without a separate code path.
    """
    if scale < 1.0:
        raise ValueError(f"scale must be >= 1, got {scale}")
    return 0.1 * math.log(scale) + 1.0


def yarn_ramp(
    head_dim: int,
    *,
    train_ctx: int,
    base: float = DEFAULT_BASE,
    alpha: float = YARN_ALPHA,
    beta: float = YARN_BETA,
) -> Array:
    """Per-pair interpolation weight in [0, 1]. 0 = extrapolate, 1 = interpolate.

    ``r_i = train_ctx / wavelength_i`` is the number of full turns pair ``i``
    completes inside the training window. The ramp is linear in ``r`` between
    ``alpha`` and ``beta`` and clamped outside.
    """
    if not 0 < alpha < beta:
        raise ValueError(f"need 0 < alpha < beta, got alpha={alpha}, beta={beta}")
    freq = inv_freq(head_dim, base=base)
    turns = train_ctx * freq / (2.0 * np.pi)
    ramp = (beta - turns) / (beta - alpha)
    return np.clip(ramp, 0.0, 1.0)


def _yarn_inv_freq(
    base_freq: Array,
    *,
    scale: float,
    train_ctx: int,
    alpha: float,
    beta: float,
) -> Array:
    head_dim = 2 * base_freq.shape[0]
    ramp = yarn_ramp(head_dim, train_ctx=train_ctx, base=_base_from(base_freq),
                     alpha=alpha, beta=beta)
    interpolated = base_freq / scale
    return ramp * interpolated + (1.0 - ramp) * base_freq


def _base_from(base_freq: Array) -> float:
    """Recover ``base`` from a frequency vector: theta_1 = base^(-2/D)."""
    head_dim = 2 * base_freq.shape[0]
    if base_freq.shape[0] < 2:
        return DEFAULT_BASE
    return float(base_freq[1] ** (-head_dim / 2.0))


def spectrum_table(
    *,
    head_dim: int,
    train_ctx: int,
    scale: float,
    base: float = DEFAULT_BASE,
) -> dict[str, Array]:
    """``effective_positions`` for every method, for the frequency-spectrum plot.

    One row per method, one column per RoPE pair. This is the numerical form
    of "what each method does to the frequency spectrum": PI is a flat line at
    ``1/s``, NTK is a curve from ~1 down to ``1/s``, YaRN is a clamped ramp
    with genuinely flat regions at both ends.
    """
    return {
        m: scaled_rope(m, scale=scale, head_dim=head_dim, train_ctx=train_ctx,
                       base=base).effective_positions
        for m in METHODS
    }


def high_frequency_preservation(sr: ScaledRope) -> float:
    """Fraction of the fastest pair's frequency retained. 1.0 = untouched.

    The single number that most cleanly separates the methods: PI gives
    exactly ``1/s`` here by construction, NTK gives ~1.0, YaRN gives exactly
    1.0 whenever the fastest pair sits above ``beta`` turns. It is a property
    of the closed form, not of any machine, so it is safe to commit.
    """
    return float(sr.effective_positions[0])


def low_frequency_preservation(sr: ScaledRope) -> float:
    """Same for the slowest pair. This is where all methods must interpolate."""
    return float(sr.effective_positions[-1])
