"""NIAH and multi-hop construction, determinism, and scoring.

No model is loaded anywhere in this file. These tests cover the parts that
must be right *before* inference is worth running: if the needle is not where
the depth says, or the scorer accepts a substring, or the suite changes
between machines, then the accuracies downstream are measuring the harness.
"""

from __future__ import annotations

import pytest

from context_extension_and_long_context_eval import multihop, niah
from context_extension_and_long_context_eval.runner import calibrate_filler

# --- NIAH construction ------------------------------------------------


def test_needle_appears_exactly_once():
    s = niah.build_sample(n_filler=200, depth=0.5, seed=0)
    assert s.prompt.count(s.needle) == 1


def test_answer_value_appears_only_in_the_needle():
    """Filler must never accidentally contain the answer string.

    If it did, a model that ignored the document entirely could score above
    zero, and the floor of the benchmark would not be 0%.
    """
    for seed in range(40):
        s = niah.build_sample(n_filler=150, depth=0.5, seed=seed)
        body = s.prompt.replace(s.needle, "")
        assert s.answer not in body


@pytest.mark.parametrize("depth", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_depth_places_the_needle_where_it_says(depth):
    """Fractional position of the needle matches the requested depth.

    Checked in line space with a one-line tolerance for rounding. Sloppiness
    here would smear the lost-in-the-middle signal, which is measured by
    comparing depths against each other.
    """
    n = 400
    s = niah.build_sample(n_filler=n, depth=depth, seed=1)
    body = s.prompt.split("--- DOCUMENT ---\n")[1].split("\n--- END DOCUMENT ---")[0]
    lines = body.split("\n")
    idx = next(i for i, ln in enumerate(lines) if ln == s.needle)
    assert abs(idx - round(depth * n)) <= 1


def test_question_names_the_needle_city():
    s = niah.build_sample(n_filler=50, depth=0.5, seed=2)
    assert f"magic number for {s.city}?" in s.prompt


def test_niah_is_deterministic_across_calls():
    a = niah.build_sample(n_filler=100, depth=0.3, seed=7)
    b = niah.build_sample(n_filler=100, depth=0.3, seed=7)
    assert a == b


def test_niah_different_seeds_give_different_items():
    items = {niah.build_sample(n_filler=50, depth=0.5, seed=s).answer for s in range(20)}
    assert len(items) > 10


def test_suite_size_and_seed_independence():
    """Growing n_per_depth must not renumber the existing samples.

    Otherwise every results row shifts when the sample count changes, and the
    diff of a results file stops being readable.
    """
    small = niah.build_suite(n_filler=50, depths=(0.0, 0.5), n_per_depth=2, seed=0)
    large = niah.build_suite(n_filler=50, depths=(0.0, 0.5), n_per_depth=5, seed=0)
    assert len(small) == 4 and len(large) == 10
    assert small[:2] == large[:2]
    assert small[2:] == large[5:7]


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_niah_rejects_out_of_range_depth(bad):
    with pytest.raises(ValueError):
        niah.build_sample(n_filler=10, depth=bad, seed=0)


# --- NIAH scoring -----------------------------------------------------


def test_score_accepts_bare_and_embedded_answers():
    s = niah.build_sample(n_filler=10, depth=0.5, seed=0)
    assert niah.score(s.answer, s) == 1.0
    assert niah.score(f"The magic number is {s.answer}.", s) == 1.0


def test_score_rejects_wrong_and_superstring_numbers():
    """Word-boundary matching: 7412 must not be scored a hit by 74123.

    Plain ``in`` would accept it. This is the most common way a retrieval eval
    quietly inflates its own numbers.
    """
    s = niah.build_sample(n_filler=10, depth=0.5, seed=0)
    assert niah.score("0000", s) == 0.0
    assert niah.score(s.answer + "9", s) == 0.0
    assert niah.score("9" + s.answer, s) == 0.0
    assert niah.score("", s) == 0.0


# --- multi-hop construction -------------------------------------------


def test_both_hops_present_and_the_answer_is_not_directly_stated():
    """The question's own phrasing must not appear next to the answer.

    If a single line said "the badge number of the coordinator for Zurich is
    7412", the task would be single-hop and the comparison meaningless.
    """
    s = multihop.build_sample(n_filler=200, seed=0)
    assert multihop.coordinator_line(s.city, s.bridge) in s.prompt
    assert multihop.badge_line(s.bridge, s.answer) in s.prompt
    assert f"badge number of the regional coordinator for {s.city} is" not in s.prompt


def test_the_answer_badge_is_unique_among_all_badges():
    """Distractor badges never collide with the answer.

    A collision would let a wrong-hop retrieval score as correct, and the
    multi-hop number would be biased upward -- against the direction of this
    repo's headline, but wrong either way.
    """
    for seed in range(30):
        s = multihop.build_sample(n_filler=100, n_distractors=6, seed=seed)
        badges = [
            ln.split("badge number ")[1].rstrip(".")
            for ln in s.prompt.split("\n")
            if "badge number " in ln and "Agent" in ln
        ]
        assert badges.count(s.answer) == 1


def test_distractors_provide_complete_competing_chains():
    """Each distractor contributes both a coordinator line and a badge line.

    Half a chain is not a distractor: a coordinator with no badge cannot be
    mistakenly followed to a wrong number.
    """
    n = 5
    s = multihop.build_sample(n_filler=100, n_distractors=n, seed=3)
    # Count inside the document only. The question line also contains the
    # phrase "regional coordinator for", which is the point of the task, not a
    # planted fact -- counting it would report one chain too many.
    body = s.prompt.split("--- DOCUMENT ---\n")[1].split("\n--- END DOCUMENT ---")[0]
    coord = sum(1 for ln in body.split("\n") if "regional coordinator for" in ln)
    badge = sum(1 for ln in body.split("\n") if "was assigned badge number" in ln)
    assert coord == n + 1
    assert badge == n + 1


def test_the_answer_number_appears_nowhere_but_the_second_hop():
    """No route to the answer that skips hop two.

    Stronger than badge uniqueness: the four-digit string must not appear
    anywhere else in the prompt at all, filler included. If it did, the scorer
    would credit a model that never composed anything, and the floor of the
    multi-hop task would not be 0%.
    """
    import re

    for seed in range(60):
        s = multihop.build_sample(n_filler=400, n_distractors=4, seed=seed)
        rest = s.prompt.replace(multihop.badge_line(s.bridge, s.answer), "")
        assert not re.search(rf"(?<!\d){s.answer}(?!\d)", rest)


def test_the_bridge_entity_appears_exactly_twice():
    """Once in each hop, and nowhere else.

    A third mention would give the model a second route to the badge and
    weaken the two-hop structure without it being visible in any score.
    """
    for seed in range(60):
        s = multihop.build_sample(n_filler=200, n_distractors=6, seed=seed)
        assert s.prompt.count(f"Agent {s.bridge}") == 2


def test_all_badge_numbers_in_the_document_are_distinct():
    """Distractor badges never collide with each other either.

    Two agents sharing a badge would make the document internally
    inconsistent, which is a different task from the one being measured.
    """
    for seed in range(60):
        s = multihop.build_sample(n_filler=100, n_distractors=6, seed=seed)
        body = s.prompt.split("--- DOCUMENT ---")[1]
        badges = [
            ln.split("badge number ")[1].rstrip(".")
            for ln in body.split("\n")
            if "badge number " in ln
        ]
        assert len(badges) == len(set(badges))


def test_the_two_hops_are_separated_in_the_document():
    """Hops at 0.25 and 0.75 must actually land far apart.

    Adjacent hops collapse the task to a local read; the separation is the
    reason this measures long-range composition.
    """
    n = 400
    s = multihop.build_sample(n_filler=n, depths=(0.25, 0.75), seed=4)
    lines = s.prompt.split("\n")
    i1 = next(i for i, ln in enumerate(lines) if ln == multihop.coordinator_line(s.city, s.bridge))
    i2 = next(i for i, ln in enumerate(lines) if ln == multihop.badge_line(s.bridge, s.answer))
    assert i2 - i1 > n * 0.3


def test_multihop_is_deterministic():
    assert multihop.build_sample(n_filler=80, seed=11) == multihop.build_sample(
        n_filler=80, seed=11
    )


def test_multihop_rejects_too_many_distractors():
    with pytest.raises(ValueError):
        multihop.build_sample(n_filler=10, n_distractors=99, seed=0)


# --- shared scorer ----------------------------------------------------


def test_both_tasks_use_an_identical_scoring_rule():
    """The gap must not be an artefact of one grader being stricter.

    Asserted by scoring the same response strings under both scorers and
    requiring agreement on every case.
    """
    ns = niah.build_sample(n_filler=10, depth=0.5, seed=0)
    ms = multihop.build_sample(n_filler=10, seed=0)
    for value, sample, scorer in ((ns.answer, ns, niah.score), (ms.answer, ms, multihop.score)):
        assert scorer(value, sample) == 1.0
        assert scorer(f"Answer: {value}", sample) == 1.0
        assert scorer(f"{value}1", sample) == 0.0
        assert scorer("no idea", sample) == 0.0


def test_bridge_diagnostic_separates_the_two_failure_modes():
    s = multihop.build_sample(n_filler=10, seed=0)
    assert multihop.scores_bridge_only(f"Agent {s.bridge}", s) == 1.0
    assert multihop.scores_bridge_only(s.bridge.lower(), s) == 1.0
    assert multihop.scores_bridge_only("Agent Nobody", s) == 0.0


# --- calibration ------------------------------------------------------


def test_calibrate_finds_the_largest_count_within_budget():
    """Binary search returns the largest n whose prompt fits the target."""
    count = calibrate_filler(100, lambda n: "x " * n, lambda s: len(s.split()), hi=1000)
    assert count == 100


def test_calibrate_is_monotone_in_the_target():
    counts = [
        calibrate_filler(t, lambda n: "x " * n, lambda s: len(s.split()), hi=5000)
        for t in (50, 100, 400)
    ]
    assert counts == sorted(counts)
