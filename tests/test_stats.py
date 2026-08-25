"""The Wilson interval used for every confidence bound in results/.

Tested here rather than trusted because the disjointness of two intervals is
the entire evidential basis for the headline claim. An interval that is too
narrow would manufacture a finding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_results import wilson  # noqa: E402


def test_all_successes_gives_an_upper_bound_of_one_and_a_sane_lower():
    """12/12 must not produce [1, 1].

    The reason Wilson is used instead of a percentile bootstrap: bootstrapping
    an all-ones sample resamples ones forever and returns the degenerate
    interval [1, 1], which would let the results file claim certainty from 12
    samples. Wilson's lower bound near 0.76 is the honest statement.
    """
    lo, hi = wilson(12, 12)
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert 0.70 < lo < 0.80


def test_zero_successes_gives_a_nonzero_upper_bound():
    """0/12 must not produce [0, 0].

    The upper bound here is load-bearing: the claim "multi-hop fails" rests on
    its *upper* bound sitting below NIAH's lower bound, and a bootstrap would
    report 0 and make the comparison vacuous.
    """
    lo, hi = wilson(0, 12)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert 0.15 < hi < 0.30


def test_interval_brackets_the_point_estimate():
    for k in range(13):
        lo, hi = wilson(k, 12)
        assert lo <= k / 12 <= hi


def test_interval_is_symmetric_under_swapping_successes_and_failures():
    """Wilson is symmetric: CI(k, n) mirrors CI(n-k, n) about 0.5."""
    for k in range(13):
        lo_a, hi_a = wilson(k, 12)
        lo_b, hi_b = wilson(12 - k, 12)
        assert lo_a == pytest.approx(1 - hi_b, abs=1e-12)
        assert hi_a == pytest.approx(1 - lo_b, abs=1e-12)


def test_interval_narrows_as_the_sample_grows():
    """More samples, tighter bound. Guards against an n-independent formula."""
    widths = [hi - lo for lo, hi in (wilson(n // 2, n) for n in (12, 48, 192, 768))]
    assert widths == sorted(widths, reverse=True)
    assert widths[-1] < widths[0] / 4


def test_half_successes_centres_near_one_half():
    lo, hi = wilson(6, 12)
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=1e-9)


def test_bounds_stay_inside_zero_one():
    for n in (1, 3, 12, 100):
        for k in range(n + 1):
            lo, hi = wilson(k, n)
            assert 0.0 <= lo <= hi <= 1.0


def test_empty_sample_returns_a_degenerate_interval_rather_than_dividing_by_zero():
    assert wilson(0, 0) == (0.0, 0.0)


def test_the_headline_comparison_is_actually_disjoint():
    """12/12 versus 2/12 -- the cells the finding rests on.

    Asserting the disjointness directly, so that a change to the interval
    maths cannot silently invalidate the claim in results/ while the results
    script still reports it as disjoint.
    """
    niah_lo, _ = wilson(12, 12)
    _, multi_hi = wilson(2, 12)
    assert multi_hi < niah_lo


def test_a_near_comparison_is_correctly_not_disjoint():
    """7/12 versus 5/12 must overlap. Guards against intervals that are too narrow."""
    a_lo, a_hi = wilson(7, 12)
    b_lo, b_hi = wilson(5, 12)
    assert not (b_hi < a_lo or a_hi < b_lo)
