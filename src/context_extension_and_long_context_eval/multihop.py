"""Multi-hop retrieval: two facts, far apart, that must be composed.

The task
--------
Two needles are planted in the same haystack::

    The regional coordinator for Zurich is Agent Vermillion.
    Agent Vermillion was assigned badge number 7412.

and the question asks: *what is the badge number of the regional coordinator
for Zurich?* Neither line answers it. The model must retrieve the first,
carry the intermediate entity ("Agent Vermillion") forward, and use it as the
key for a second retrieval.

Why this is the right harder task
---------------------------------
It is the *minimal* strengthening of NIAH. Same filler, same document
structure, same answer format, same scorer, same lengths. The only thing added
is one hop. That control is deliberate: if the two curves separate, the
separation cannot be attributed to a different prompt style, a harder answer
format, or a more lenient grader, because those are all held fixed. It is the
hop, and only the hop.

Why one hop is enough to break it
---------------------------------
Single-hop lexical retrieval is close to a solved primitive -- induction heads
match the token and copy what follows. Two-hop retrieval is not one primitive
twice. The first hop's output has to be held in the residual stream and used
as a *query* for the second, which means attention must resolve a key that was
not present in the prompt. When positional information degrades, single-hop
degrades gracefully (the match is still lexically unambiguous) while two-hop
degrades sharply, because an error in hop one makes hop two retrieve the wrong
span entirely rather than a slightly worse one. Errors compose; they do not
average.

Distractors
-----------
The document contains other coordinator lines and other badge lines for other
cities and agents. Without them a model could skip hop one entirely: if only
one badge number exists in the whole document, answering "the badge number"
is single-hop again. Distractors are what make the two-hop structure load-
bearing rather than decorative, and ``n_distractors`` is reported alongside
every score so the difficulty is legible.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .niah import CITIES, FILLER

AGENTS = (
    "Vermillion", "Cobalt", "Sorrel", "Marigold", "Fenwick", "Halcyon",
    "Quillon", "Ravenna", "Thistle", "Umber", "Vantablack", "Wren",
)


@dataclass(frozen=True)
class MultihopSample:
    """One two-hop item. ``bridge`` is the intermediate entity."""

    prompt: str
    answer: str
    city: str
    bridge: str
    depths: tuple[float, float]
    n_filler: int
    n_distractors: int


def coordinator_line(city: str, agent: str) -> str:
    return f"The regional coordinator for {city} is Agent {agent}."


def badge_line(agent: str, badge: str) -> str:
    return f"Agent {agent} was assigned badge number {badge}."


def build_sample(
    *,
    n_filler: int,
    depths: tuple[float, float] = (0.25, 0.75),
    n_distractors: int = 4,
    seed: int = 0,
) -> MultihopSample:
    """Build one two-hop item with the hops at the two given depths.

    The hops are placed at *different* depths on purpose. Adjacent hops are a
    much easier task -- one attention window covers both and the composition
    collapses into a single lookup. Separating them is what makes the task
    test long-range composition rather than local reading.
    """
    for d in depths:
        if not 0.0 <= d <= 1.0:
            raise ValueError(f"depths must be in [0, 1], got {depths}")
    if n_distractors < 0:
        raise ValueError(f"n_distractors must be non-negative, got {n_distractors}")
    if n_distractors > len(CITIES) - 1:
        raise ValueError(f"at most {len(CITIES) - 1} distractors available")

    rng = random.Random(seed)
    cities = list(CITIES)
    agents = list(AGENTS)
    rng.shuffle(cities)
    rng.shuffle(agents)

    city, bridge = cities[0], agents[0]
    badge = str(rng.randrange(1000, 10000))

    lines = [rng.choice(FILLER) for _ in range(n_filler)]

    # Distractors: complete coordinator+badge chains for other cities, so
    # every surface pattern in the question also occurs for the wrong answer.
    used = {badge}
    for i in range(n_distractors):
        d_city, d_agent = cities[i + 1], agents[i + 1]
        while (d_badge := str(rng.randrange(1000, 10000))) in used:
            pass
        used.add(d_badge)
        _insert(lines, coordinator_line(d_city, d_agent), rng.random())
        _insert(lines, badge_line(d_agent, d_badge), rng.random())

    # The two real hops last, so their depths are exact rather than perturbed
    # by distractor insertions happening after them.
    _insert(lines, coordinator_line(city, bridge), depths[0])
    _insert(lines, badge_line(bridge, badge), depths[1])

    body = "\n".join(lines)
    prompt = (
        "Read the document below and answer the question using only the "
        "document.\n\n"
        f"--- DOCUMENT ---\n{body}\n--- END DOCUMENT ---\n\n"
        f"Question: What is the badge number of the regional coordinator "
        f"for {city}?\n"
        "Answer with the number only."
    )
    return MultihopSample(
        prompt=prompt,
        answer=badge,
        city=city,
        bridge=bridge,
        depths=depths,
        n_filler=n_filler,
        n_distractors=n_distractors,
    )


def _insert(lines: list[str], text: str, depth: float) -> None:
    lines.insert(round(depth * len(lines)), text)


def score(response: str, sample: MultihopSample) -> float:
    """Identical scorer to NIAH: exact number on a word boundary.

    Sharing the scorer is load-bearing for the comparison. If NIAH were graded
    leniently and multi-hop strictly, the gap this repo reports would be an
    artefact of the graders rather than of the tasks.
    """
    return 1.0 if re.search(rf"(?<!\d){re.escape(sample.answer)}(?!\d)", response) else 0.0


def scores_bridge_only(response: str, sample: MultihopSample) -> float:
    """Did the model name the intermediate entity, even if it missed the badge?

    The diagnostic that separates the two failure modes. A model that says
    "Agent Vermillion" but the wrong number completed hop one and failed hop
    two. A model that names neither failed at hop one. Reporting only the
    final accuracy conflates these, and they call for different conclusions:
    the first is a composition failure, the second is a retrieval failure.
    """
    return 1.0 if re.search(rf"\b{re.escape(sample.bridge)}\b", response, re.I) else 0.0


def build_suite(
    *,
    n_filler: int,
    n_samples: int,
    depths: tuple[float, float] = (0.25, 0.75),
    n_distractors: int = 4,
    seed: int = 0,
) -> list[MultihopSample]:
    """A deterministic suite at one context length."""
    return [
        build_sample(
            n_filler=n_filler,
            depths=depths,
            n_distractors=n_distractors,
            seed=seed + i,
        )
        for i in range(n_samples)
    ]
