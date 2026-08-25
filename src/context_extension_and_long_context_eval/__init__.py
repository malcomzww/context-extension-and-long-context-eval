"""RoPE scaling by hand - position interpolation, NTK-aware, YaRN - evaluated
with something better than needle-in-a-haystack.

Four modules, split by what they need to run:

    rope.py          the rotation formulation. NumPy only.
    scaling.py       PI, NTK-aware and YaRN as edits to one frequency vector.
    alternatives.py  learned/sinusoidal/ALiBi/NoPE, for contrast.
    niah.py          single-hop lexical retrieval.
    multihop.py      the same haystack, plus one hop.
    runner.py        model execution. The only module that imports torch, and
                     it does so lazily so everything above stays importable
                     with nothing but NumPy.

That last split is why CI can regenerate results/ without a model download.
"""

__version__ = "0.1.0"
