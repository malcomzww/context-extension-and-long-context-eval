"""RoPE properties, with fixtures small enough to check by hand.

The head dimension is 4 wherever a fixture is hand-computed. With D=4 there
are exactly two pairs and, at base 10000, theta = (1.0, 0.01) -- both exact in
binary floating point for theta_0 and close enough to be written down for
theta_1. That makes the expected rotation a 2x2 matrix anyone can verify with
a calculator, which is the point: a test whose expected value came out of the
implementation proves only that the implementation is deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from context_extension_and_long_context_eval.rope import (
    DEFAULT_BASE,
    angles,
    apply_rope,
    attention_score_by_offset,
    interleave_to_half_perm,
    inv_freq,
    random_qk,
    rope,
    rope_cos_sin,
    rotate_half_style,
    wavelengths,
)

# --- the frequency ladder ---------------------------------------------


def test_inv_freq_hand_computed_d4():
    """D=4, base=10000: theta_i = 10000^(-2i/4) = (10000^0, 10000^-0.5)."""
    got = inv_freq(4)
    assert got.shape == (2,)
    assert got[0] == pytest.approx(1.0, abs=0.0)
    assert got[1] == pytest.approx(0.01, rel=1e-12)


def test_theta_zero_is_exactly_one_for_any_base_or_dim():
    """theta_0 = base^0 = 1 always. The fastest pair turns one radian/token.

    This is why NTK-aware scaling cannot damage the fastest pair no matter how
    large it makes the base -- the exponent there is zero.
    """
    for d in (2, 4, 16, 128):
        for b in (100.0, 10000.0, 1_000_000.0):
            assert inv_freq(d, base=b)[0] == 1.0


def test_wavelength_of_slowest_pair_approaches_two_pi_base():
    """The last pair's wavelength is ~2*pi*base -- 62.8k tokens at base 10000.

    So a 4k-trained model never completes one turn of its slowest pair. That
    fact is the entire justification for treating low frequencies differently,
    and both NTK and YaRN depend on it.
    """
    w = wavelengths(128)
    assert w[-1] == pytest.approx(2 * math.pi * DEFAULT_BASE * 10000 ** (-2 / 128), rel=1e-12)
    assert w[-1] > 40_000


def test_inv_freq_is_strictly_decreasing():
    f = inv_freq(64)
    assert np.all(np.diff(f) < 0)


@pytest.mark.parametrize("bad", [0, -2, 3, 7])
def test_inv_freq_rejects_odd_or_nonpositive(bad):
    with pytest.raises(ValueError):
        inv_freq(bad)


# --- the three properties the brief names ------------------------------


def test_position_zero_is_the_identity():
    """m=0 gives angle 0 for every pair, so the rotation matrix is I.

    Exact, not approximate: cos(0) and sin(0) are exactly 1.0 and 0.0.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 1, 64))
    out = rope(x, [0])
    assert np.array_equal(out, x)


def test_relative_position_determines_the_dot_product():
    """<R(m)q, R(n)k> depends only on m-n.

    Checked across a wide range of absolute anchors, including anchors far
    past any plausible training length. This is the property that makes RoPE a
    relative encoding despite being applied absolutely.
    """
    q, k = random_qk(64, seed=3)
    offsets = [0, 1, 2, 17, 256, 4095]
    reference = attention_score_by_offset(q, k, offsets, anchor=0)
    for anchor in (1, 37, 1024, 100_000):
        got = attention_score_by_offset(q, k, offsets, anchor=anchor)
        assert np.allclose(got, reference, atol=1e-12)


def test_same_relative_offset_gives_same_score_regardless_of_absolute_position():
    """The brief's third property, stated directly on rotated vectors.

    Distinct from the test above in that it rotates full q and k stacks rather
    than using the convenience helper, so a bug in the helper cannot hide a
    bug in ``apply_rope``.
    """
    rng = np.random.default_rng(7)
    d = 32
    q = rng.standard_normal(d)
    k = rng.standard_normal(d)
    offset = 13
    scores = []
    for m in (0, 5, 500, 50_000):
        qr = rope(q.reshape(1, d), [m + offset])[0]
        kr = rope(k.reshape(1, d), [m])[0]
        scores.append(float(qr @ kr))
    assert max(scores) - min(scores) < 1e-11


# --- the rotation itself ----------------------------------------------


def test_apply_rope_matches_hand_written_2x2_rotation():
    """D=4, position 1. theta = (1.0, 0.01), so the angles are 1 and 0.01 rad."""
    x = np.array([[1.0, 0.0, 0.0, 1.0]])
    got = apply_rope(x, *rope_cos_sin([1], 4))
    # pair 0 = (1, 0) rotated by 1 rad -> (cos 1, sin 1)
    # pair 1 = (0, 1) rotated by 0.01 rad -> (-sin 0.01, cos 0.01)
    expected = np.array([[math.cos(1.0), math.sin(1.0), -math.sin(0.01), math.cos(0.01)]])
    assert np.allclose(got, expected, atol=1e-15)


def test_rotation_preserves_the_norm_of_every_pair():
    """A rotation is orthogonal, so |x| is invariant -- at any position.

    Worth asserting because it is the reason RoPE can be applied at an
    arbitrary position without the logits blowing up. Nothing numerically
    diverges past the training length; what fails is semantic, not numeric,
    and this test pins down that distinction.
    """
    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, 5, 64))
    for m in (0, 1, 4096, 10**6):
        out = rope(x, [m] * 5)
        assert np.allclose(np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-12)


def test_rotations_compose_additively():
    """R(m) R(n) = R(m+n). Rotation angles add."""
    rng = np.random.default_rng(13)
    x = rng.standard_normal((1, 64))
    once = rope(rope(x, [40]), [60])
    direct = rope(x, [100])
    assert np.allclose(once, direct, atol=1e-12)


def test_rotation_is_periodic_in_two_pi_over_theta():
    """Pair 0 has theta=1, so position 2*pi returns it to where it started."""
    x = np.array([[1.0, 0.0]])
    back = rope(x, [2 * math.pi])
    assert np.allclose(back, x, atol=1e-12)


def test_angles_table_shape_and_values():
    a = angles([0, 1, 2], 4)
    assert a.shape == (3, 2)
    assert np.allclose(a[2], [2.0, 0.02], atol=1e-12)


# --- layout equivalence -----------------------------------------------


def test_interleaved_and_rotate_half_agree_under_the_permutation():
    """HF's split-half layout equals ours after a fixed index permutation.

    Proving it rather than asserting it in prose, because getting this wrong
    when porting weights produces a model that is subtly wrong rather than
    obviously broken.
    """
    rng = np.random.default_rng(17)
    d = 16
    x = rng.standard_normal((4, d))
    cos, sin = rope_cos_sin([0, 1, 2, 3], d)

    ours = apply_rope(x, cos, sin)
    perm = interleave_to_half_perm(d)
    theirs = rotate_half_style(x[:, perm], cos, sin)
    assert np.allclose(ours[:, perm], theirs, atol=1e-13)


def test_permutation_is_a_bijection():
    perm = interleave_to_half_perm(64)
    assert sorted(perm.tolist()) == list(range(64))


# --- extrapolation behaviour ------------------------------------------


def test_score_decays_then_oscillates_with_offset():
    """Near offsets keep a strong score; far offsets oscillate around zero.

    Not a claim about any trained model -- it is a property of the rotation
    with random q, k. It matters because it shows the failure mode is *not*
    divergence: the far-offset scores stay bounded and simply lose the
    monotone structure the model learned to read.
    """
    q, k = random_qk(64, seed=5)
    near = attention_score_by_offset(q, k, list(range(0, 8)))
    far = attention_score_by_offset(q, k, list(range(100_000, 100_008)))
    assert abs(near[0]) > abs(far).mean()
    assert np.all(np.abs(far) < 2.0)


def test_attention_scale_multiplies_both_q_and_k():
    """A scale ``a`` in the tables produces ``a^2`` on the logit."""
    q, k = random_qk(32, seed=9)
    plain = attention_score_by_offset(q, k, [10])
    scaled = attention_score_by_offset(q, k, [10], attention_scale=0.5)
    assert scaled[0] == pytest.approx(0.25 * plain[0], rel=1e-12)


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        apply_rope(np.zeros((2, 5)), *rope_cos_sin([0, 1], 4))
    with pytest.raises(ValueError):
        rope_cos_sin([0], 8, inv_freq_override=np.ones(3))
