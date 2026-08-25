"""Needle in a haystack, and why it is an easy task wearing a hard task's clothes.

NIAH hides one sentence ("The magic number for Zurich is 7412.") inside a long
filler document and asks the model to repeat it back. It became the standard
long-context benchmark because it produces the pretty red-green grid of depth
against length that everyone recognises.

The structural problem is that NIAH is *lexical single-hop retrieval*. The
needle is the only span in the context that matches the question's surface
form: the question says "Zurich", exactly one line in 8000 tokens says
"Zurich", and attention only has to find it. That is close to the easiest
thing a transformer can do with a long context, because induction heads solve
it directly -- match the token, copy what follows. No composition, no holding
two facts at once, no reasoning over what was retrieved.

So NIAH measures whether the *positional encoding still functions* at that
length. It does not measure whether the model can *use* the context. Those
two come apart, and the gap between them is what ``multihop.py`` measures and
what this repo reports.

Filler generation is deterministic given a seed: the results file has to
regenerate byte-identically on a different machine, so nothing here may touch
the global RNG or the clock.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

CITIES = (
    "Zurich", "Lisbon", "Osaka", "Nairobi", "Bogota", "Helsinki",
    "Manila", "Toronto", "Dakar", "Perth", "Riga", "Quito",
)

# Bland, low-entropy filler. Deliberately not narrative: sentences that tell a
# story give the model semantic scaffolding to navigate by, which makes the
# haystack easier and flatters the score. Filler that is uniform means the
# needle is found by position and lexical match alone -- the thing under test.
FILLER = (
    "The quarterly maintenance window has been scheduled without incident.",
    "Routine inspection of the north corridor found no items requiring action.",
    "Attendance at the weekly coordination meeting remained within normal range.",
    "The supply inventory was reconciled against the ledger and matched.",
    "No changes to the standing operating procedure were proposed this period.",
    "Ambient conditions in the storage area stayed inside the specified band.",
    "The duty roster was circulated to all departments on the usual schedule.",
    "Equipment calibration records were reviewed and found to be current.",
    "The access log showed no entries outside of permitted hours.",
    "Consumable stock levels were reported as adequate for the coming month.",
)


@dataclass(frozen=True)
class NiahSample:
    """One NIAH item: the prompt, the string that must appear, and provenance."""

    prompt: str
    answer: str
    city: str
    depth: float
    n_filler: int

    @property
    def needle(self) -> str:
        return needle_sentence(self.city, self.answer)


def needle_sentence(city: str, value: str) -> str:
    return f"The magic number for {city} is {value}."


def build_sample(
    *,
    n_filler: int,
    depth: float,
    seed: int,
) -> NiahSample:
    """Build one NIAH item with the needle at fractional ``depth``.

    ``depth`` 0.0 puts the needle before all filler, 1.0 after all of it, 0.5
    in the middle. Sweeping depth is what exposes lost-in-the-middle: a model
    with a genuine positional deficit scores well at 0.0 and 1.0 and badly
    around 0.5, and reporting only the mean over depths hides exactly that.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {depth}")
    if n_filler < 0:
        raise ValueError(f"n_filler must be non-negative, got {n_filler}")

    rng = random.Random(seed)
    city = rng.choice(CITIES)
    value = str(rng.randrange(1000, 10000))

    lines = [rng.choice(FILLER) for _ in range(n_filler)]
    insert_at = round(depth * n_filler)
    lines.insert(insert_at, needle_sentence(city, value))

    body = "\n".join(lines)
    prompt = (
        "Read the document below and answer the question using only the "
        "document.\n\n"
        f"--- DOCUMENT ---\n{body}\n--- END DOCUMENT ---\n\n"
        f"Question: What is the magic number for {city}?\n"
        "Answer with the number only."
    )
    return NiahSample(
        prompt=prompt, answer=value, city=city, depth=depth, n_filler=n_filler
    )


def score(response: str, sample: NiahSample) -> float:
    """1.0 if the exact four-digit answer appears as a standalone number.

    Deliberately generous on formatting -- the model may say "7412" or "The
    magic number is 7412" and both are correct -- and deliberately strict on
    the value, matched on a word boundary so that 7412 does not count as a hit
    for 74123. Substring matching without the boundary is the single most
    common way a retrieval eval reports a score that is too high.
    """
    return 1.0 if re.search(rf"(?<!\d){re.escape(sample.answer)}(?!\d)", response) else 0.0


def build_suite(
    *,
    n_filler: int,
    depths: tuple[float, ...],
    n_per_depth: int,
    seed: int = 0,
) -> list[NiahSample]:
    """A full depth sweep at one context length. Deterministic given ``seed``.

    Seeds are derived by index rather than drawn from one shared generator so
    that changing ``n_per_depth`` does not renumber every other sample. A
    results file that shifts when an unrelated parameter changes cannot be
    diffed usefully.
    """
    out: list[NiahSample] = []
    for di, depth in enumerate(depths):
        for i in range(n_per_depth):
            out.append(
                build_sample(
                    n_filler=n_filler,
                    depth=depth,
                    seed=seed + 10_000 * di + i,
                )
            )
    return out
