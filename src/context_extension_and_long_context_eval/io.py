"""File writing that does not depend on which OS ran the script.

One function, because it guards the CI gate and both scripts need it.
"""

from __future__ import annotations

from pathlib import Path


def write_lf(path: Path, text: str) -> None:
    """Write ``text`` with Unix line endings on every platform.

    ``Path.write_text`` translates "\\n" to the platform default, so the same
    generator emits CRLF on Windows and LF on Linux. Since the drift gate
    byte-compares ``results/``, that difference alone makes
    ``git diff --exit-code results/`` fail for a contributor on Windows even
    when every number in the file is identical.

    Found exactly that way: ``.gitattributes`` normalises to LF on commit, so
    the committed file was LF while local regeneration produced CRLF, and the
    gate reported drift on line 1 of a file whose content had not changed.
    Normalising on commit is not enough -- the *generator* has to be
    deterministic, because the gate compares the working tree.

    ``newline=""`` disables translation so the "\\n" characters in the string
    reach the file untouched.
    """
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
