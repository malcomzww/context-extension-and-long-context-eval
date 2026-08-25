"""Position-scaling methods: closed-form properties, hand-checkable.

None of these assertions depend on a model, a dataset, or a machine. Every
one follows from the algebra in ``scaling.py``, which is exactly why this file
can be strict -- tolerances here are 1e-12, not 1e-2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from context_extension_and_long_context_eval.rope import (
    attention_score_by_offset,
    inv_freq,
    random_qk,
)
from context_extension_and_long_context_eval.scaling import (
    METHODS,
    high_frequency_preservation,
    low_frequency_preservation,
    scaled_rope,
    spectrum_table,
    yarn_ramp,
    yarn_temperature,
)

D = 64
CTX = 2048


def build(method: str, scale: float = 8.0, head_dim: int = D):
    return scaled_rope(method, scale=scale, head_dim=head_dim, train_ctx=CTX)


# --- identity at scale 1 ----------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_scale_one_is_a_no_op_for_every_method(method):
    """s=1 must leave RoPE exactly unchanged, including the temperature.

    A method that only degenerates approximately at s=1 forces a branch in the
    serving path; all three here degenerate exactly.
    """
    sr = build(method, scale=1.0)
    assert np.allclose(sr.inv_freq, inv_freq(D), rtol=1e-14, atol=0.0)
    assert sr.attention_scale == pytest.approx(1.0, abs=1e-15)


# --- position interpolation -------------------------------------------


def test_pi_divides_every_frequency_by_exactly_s():
    """PI is uniform by definition: the effective-position column is flat."""
    sr = build("pi", scale=8.0)
    assert np.allclose(sr.effective_positions, 1 / 8.0, rtol=1e-14)
    assert high_frequency_preservation(sr) == pytest.approx(0.125, rel=1e-14)
    assert low_frequency_preservation(sr) == pytest.approx(0.125, rel=1e-14)


def test_pi_maps_extended_position_onto_a_trained_angle():
    """Under PI at s=4, position 8192 presents the angle trained at 2048.

    Stated as an equality on the actual attention score, which is the thing
    the model consumes, rather than on the frequency vector.
    """
    q, k = random_qk(D, seed=2)
    sr = build("pi", scale=4.0)
    extended = attention_score_by_offset(q, k, [8192], inv_freq_override=sr.inv_freq)
    trained = attention_score_by_offset(q, k, [2048])
    assert extended[0] == pytest.approx(trained[0], abs=1e-11)


# --- NTK-aware --------------------------------------------------------


def test_ntk_leaves_the_fastest_pair_untouched():
    """theta_0 = base^0 = 1 for any base, so raising the base cannot move it.

    This is the mechanism behind NTK's headline property, and it is exact.
    """
    for s in (2.0, 8.0, 32.0):
        assert high_frequency_preservation(build("ntk", scale=s)) == pytest.approx(1.0, abs=1e-15)


def test_ntk_scales_the_slowest_pair_by_exactly_s():
    """The D/(D-2) exponent is chosen precisely so this comes out to 1/s.

    Hand-derivable: theta_last = base^(-(D-2)/D), so with base' = base*s^(D/(D-2))
    the ratio is s^(-(D/(D-2)) * ((D-2)/D)) = s^-1.
    """
    for s in (2.0, 8.0, 32.0):
        got = low_frequency_preservation(build("ntk", scale=s))
        assert got == pytest.approx(1.0 / s, rel=1e-12)


def test_ntk_is_monotone_between_its_two_endpoints():
    """Every intermediate pair is scaled between 1 and 1/s, monotonically."""
    sr = build("ntk", scale=8.0)
    eff = sr.effective_positions
    assert np.all(np.diff(eff) < 0)
    assert eff[0] == pytest.approx(1.0, abs=1e-15)
    assert eff[-1] == pytest.approx(0.125, rel=1e-12)


def test_ntk_preserves_more_high_frequency_than_pi_at_every_pair():
    """The comparison that motivates NTK over PI, asserted pairwise."""
    ntk = build("ntk", scale=8.0).effective_positions
    pi = build("pi", scale=8.0).effective_positions
    assert np.all(ntk >= pi - 1e-12)
    assert ntk[0] > pi[0] * 4


# --- YaRN -------------------------------------------------------------


def test_yarn_ramp_is_zero_for_fast_pairs_and_one_for_slow_pairs():
    """Below alpha turns -> full interpolation; above beta turns -> none.

    With D=64, ctx=2048 the fastest nine pairs complete more than 32 turns and
    are left alone entirely, while the slowest pairs complete under one turn
    and are interpolated like PI.
    """
    ramp = yarn_ramp(D, train_ctx=CTX)
    assert ramp[0] == 0.0
    assert ramp[-1] == 1.0
    assert np.all(np.diff(ramp) >= -1e-15)
    assert 0 < np.sum((ramp > 0) & (ramp < 1)) < D // 2


def test_yarn_ramp_boundaries_match_the_turn_count_definition():
    """A pair sits at ramp 0 exactly when it completes >= beta turns."""
    ramp = yarn_ramp(D, train_ctx=CTX)
    turns = CTX * inv_freq(D) / (2 * math.pi)
    assert np.all(ramp[turns >= 32.0] == 0.0)
    assert np.all(ramp[turns <= 1.0] == 1.0)


def test_yarn_matches_pi_on_the_pairs_it_fully_interpolates():
    """Where ramp == 1, YaRN's frequency is exactly PI's. No approximation."""
    yarn = build("yarn", scale=8.0).inv_freq
    pi = build("pi", scale=8.0).inv_freq
    full = yarn_ramp(D, train_ctx=CTX) == 1.0
    assert full.any()
    assert np.allclose(yarn[full], pi[full], rtol=1e-14)


def test_yarn_matches_plain_rope_on_the_pairs_it_extrapolates():
    yarn = build("yarn", scale=8.0).inv_freq
    none = build("none", scale=8.0).inv_freq
    zero = yarn_ramp(D, train_ctx=CTX) == 0.0
    assert zero.any()
    assert np.allclose(yarn[zero], none[zero], rtol=1e-14)


def test_yarn_temperature_is_one_at_scale_one_and_grows_with_log_s():
    assert yarn_temperature(1.0) == pytest.approx(1.0, abs=0.0)
    assert yarn_temperature(math.e) == pytest.approx(1.1, rel=1e-12)
    assert yarn_temperature(32.0) > yarn_temperature(8.0) > yarn_temperature(2.0)


def test_yarn_is_the_only_method_that_touches_the_logit_scale():
    """The term reimplementations drop. Asserting it keeps it from being dropped."""
    for m in ("none", "pi", "ntk"):
        assert build(m, scale=8.0).attention_scale == 1.0
    assert build("yarn", scale=8.0).attention_scale < 1.0


def test_yarn_attention_scale_squares_to_one_over_t():
    sr = build("yarn", scale=8.0)
    assert sr.attention_scale**2 == pytest.approx(1 / yarn_temperature(8.0), rel=1e-12)


# --- properties every method must keep --------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_relative_invariance_survives_every_scaling_method(method):
    """Scaling must not break the property RoPE exists for.

    A method that made the score depend on the absolute anchor would have
    destroyed relative encoding to buy range -- a bad trade nobody would
    accept, and one this test would catch.
    """
    q, k = random_qk(D, seed=4)
    sr = build(method, scale=8.0)
    kw = dict(inv_freq_override=sr.inv_freq, attention_scale=sr.attention_scale)
    ref = attention_score_by_offset(q, k, [0, 64, 4096], anchor=0, **kw)
    for anchor in (13, 2048, 30_000):
        got = attention_score_by_offset(q, k, [0, 64, 4096], anchor=anchor, **kw)
        assert np.allclose(got, ref, atol=1e-11)


@pytest.mark.parametrize("method", METHODS)
def test_frequencies_stay_positive_and_ordered(method):
    sr = build(method, scale=16.0)
    assert np.all(sr.inv_freq > 0)
    assert np.all(np.diff(sr.inv_freq) < 0)


def test_spectrum_table_covers_every_method_with_matching_shapes():
    tab = spectrum_table(head_dim=D, train_ctx=CTX, scale=8.0)
    assert set(tab) == set(METHODS)
    assert all(v.shape == (D // 2,) for v in tab.values())


def test_method_ordering_on_high_frequency_preservation_is_stable():
    """The ordering committed to results/: none == ntk == yarn > pi.

    An ordering rather than four numbers, because an ordering is what
    transfers to another machine -- and here it also happens to be exact.
    """
    for s in (2.0, 4.0, 8.0, 16.0, 32.0):
        hi = {m: high_frequency_preservation(build(m, scale=s)) for m in METHODS}
        assert hi["none"] == pytest.approx(1.0, abs=1e-15)
        assert hi["ntk"] == pytest.approx(1.0, abs=1e-15)
        assert hi["yarn"] == pytest.approx(1.0, abs=1e-15)
        assert hi["pi"] == pytest.approx(1.0 / s, rel=1e-14)
        assert hi["pi"] < hi["ntk"]


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        scaled_rope("longrope", scale=2.0, head_dim=D, train_ctx=CTX)
    with pytest.raises(ValueError):
        scaled_rope("pi", scale=0.5, head_dim=D, train_ctx=CTX)
    with pytest.raises(ValueError):
        scaled_rope("pi", scale=2.0, head_dim=D, train_ctx=0)
    with pytest.raises(ValueError):
        yarn_temperature(0.5)
    with pytest.raises(ValueError):
        yarn_ramp(D, train_ctx=CTX, alpha=32.0, beta=1.0)
