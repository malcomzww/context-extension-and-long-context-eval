"""The alternative encodings, checked on the axis this repo cares about.

Each test targets the extrapolation property that distinguishes the encoding
from RoPE, not the encoding's general correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from context_extension_and_long_context_eval.alternatives import (
    alibi_bias,
    alibi_effective_window,
    alibi_slopes,
    nope_visible_counts,
    sinusoidal_encoding,
)

# --- sinusoidal -------------------------------------------------------


def test_sinusoidal_position_zero_is_alternating_zeros_and_ones():
    """sin(0)=0, cos(0)=1 for every frequency. Hand-checkable."""
    pe = sinusoidal_encoding(1, 8)
    assert np.allclose(pe[0, 0::2], 0.0, atol=1e-15)
    assert np.allclose(pe[0, 1::2], 1.0, atol=1e-15)


def test_sinusoidal_is_bounded_at_any_position():
    """Defined and bounded far past training -- unlike a learned table.

    It does not crash. It also does not work, and the distinction between
    those two is the point of including it.
    """
    pe = sinusoidal_encoding(4, 16)
    far = sinusoidal_encoding(100_001, 16)[100_000]
    assert np.all(np.abs(pe) <= 1.0 + 1e-12)
    assert np.all(np.abs(far) <= 1.0 + 1e-12)


def test_sinusoidal_shares_ropes_frequency_ladder():
    """Same theta ladder, applied additively instead of as a rotation.

    Asserting the shared ladder makes the actual difference legible: it is the
    application, not the frequencies.
    """
    from context_extension_and_long_context_eval.rope import inv_freq

    d = 32
    pe = sinusoidal_encoding(2, d)
    expected = np.sin(inv_freq(d))
    assert np.allclose(pe[1, 0::2], expected, atol=1e-12)


def test_sinusoidal_rejects_odd_dimension():
    with pytest.raises(ValueError):
        sinusoidal_encoding(4, 7)


# --- ALiBi ------------------------------------------------------------


def test_alibi_slopes_are_positive_and_decreasing():
    s = alibi_slopes(8)
    assert s.shape == (8,)
    assert np.all(s > 0)
    assert np.all(np.diff(s) < 0)


def test_alibi_slopes_for_eight_heads_are_powers_of_two():
    """With n_heads=8, start = 2^(-8/8) = 1/2, so slopes are 2^-1 .. 2^-8."""
    s = alibi_slopes(8)
    assert np.allclose(s, [2.0**-k for k in range(1, 9)], rtol=1e-12)


def test_alibi_bias_is_zero_on_the_diagonal_and_negative_below():
    b = alibi_bias(4, 6)
    assert np.allclose(np.diagonal(b, axis1=1, axis2=2), 0.0)
    for h in range(4):
        assert b[h, 5, 0] < b[h, 5, 4] < 0


def test_alibi_penalty_is_exactly_linear_in_distance():
    """-m*(i-j). Doubling the distance doubles the penalty, at any length.

    This is the property that makes ALiBi extrapolate: it is defined and
    behaves identically at distance 10 and distance 10 million.
    """
    b = alibi_bias(2, 1, k_len=1000)
    slopes = alibi_slopes(2)
    for h in range(2):
        assert b[h, 0, 999 - 100] == pytest.approx(-slopes[h] * 100, rel=1e-12)
        assert b[h, 0, 999 - 200] == pytest.approx(-slopes[h] * 200, rel=1e-12)


def test_alibi_ordering_by_recency_holds_at_any_length():
    """Nearer keys always outrank farther ones. True at 100 and at 100k.

    RoPE has no equivalent guarantee -- its score oscillates with offset -- and
    that is the whole difference in extrapolation behaviour.
    """
    for k_len in (100, 100_000):
        b = alibi_bias(1, 1, k_len=k_len)[0, 0]
        assert np.all(np.diff(b) > 0)


def test_alibi_effective_window_scales_as_one_over_slope():
    """Halving the slope doubles the reach. The window is fixed by the head.

    So a steep-slope head has the same effective window whether the model is
    served at 2k or 200k. Extrapolation is free precisely because reach was
    capped in the first place.
    """
    assert alibi_effective_window(0.25) == pytest.approx(2 * alibi_effective_window(0.5))
    assert alibi_effective_window(1.0, logit_budget=8.0) == pytest.approx(8.0)


def test_alibi_rejects_invalid_input():
    with pytest.raises(ValueError):
        alibi_slopes(0)
    with pytest.raises(ValueError):
        alibi_effective_window(0.0)


# --- NoPE -------------------------------------------------------------


def test_nope_visible_count_is_strictly_monotone():
    """Causal masking alone gives a strictly increasing signal.

    Enough to recover ordering without any position encoding at all, which is
    why NoPE is not the absurdity it sounds like.
    """
    c = nope_visible_counts(10)
    assert np.all(np.diff(c) == 1.0)
    assert c[0] == 1.0
