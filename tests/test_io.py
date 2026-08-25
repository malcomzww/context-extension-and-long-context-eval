"""The line-ending guard on the drift gate.

Small, but it protects the one property the CI gate depends on and cannot
check for itself: that regenerating results/ produces the same bytes on
Windows and on Linux.
"""

from __future__ import annotations

from pathlib import Path

from context_extension_and_long_context_eval.io import write_lf


def test_write_lf_never_emits_crlf(tmp_path: Path):
    """The bug this function exists for.

    ``Path.write_text`` would translate these newlines to CRLF on Windows,
    making the drift gate report a diff on every line of an unchanged file.
    """
    p = tmp_path / "out.md"
    write_lf(p, "one\ntwo\nthree\n")
    raw = p.read_bytes()
    assert b"\r\n" not in raw
    assert raw == b"one\ntwo\nthree\n"


def test_write_lf_round_trips_exactly(tmp_path: Path):
    p = tmp_path / "out.md"
    text = "# Heading\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    write_lf(p, text)
    assert p.read_bytes() == text.encode("utf-8")


def test_write_lf_is_idempotent(tmp_path: Path):
    """Writing twice gives identical bytes -- what the gate actually compares."""
    p = tmp_path / "out.md"
    write_lf(p, "x\ny\n")
    first = p.read_bytes()
    write_lf(p, "x\ny\n")
    assert p.read_bytes() == first


def test_write_lf_handles_utf8(tmp_path: Path):
    p = tmp_path / "out.md"
    write_lf(p, "theta θ and 2×\n")
    assert p.read_text(encoding="utf-8") == "theta θ and 2×\n"
