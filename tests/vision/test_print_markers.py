from __future__ import annotations

import re

import numpy as np
import pytest

from racketguitar.config import MarkerConfig
from racketguitar.vision.markers import detect_markers, make_detector, marker_centre
from tools.print_markers import PAPER_MM, bilevel, main, render_page, write_pdf

DPI = 300


def media_box(pdf: bytes) -> tuple[float, float]:
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
    assert m, "no MediaBox"
    return float(m.group(1)), float(m.group(2))


@pytest.mark.parametrize("paper", sorted(PAPER_MM))
def test_writes_true_size_pdf(tmp_path, paper):
    out = write_pdf(tmp_path / "sub" / "markers.pdf", MarkerConfig(), paper, DPI)
    data = out.read_bytes()
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    w_pt, h_pt = media_box(data)
    w_mm, h_mm = PAPER_MM[paper]
    assert w_pt == pytest.approx(w_mm / 25.4 * 72, abs=0.5)
    assert h_pt == pytest.approx(h_mm / 25.4 * 72, abs=0.5)
    assert b"/CCITTFaxDecode" in data or b"/DCTDecode" in data


@pytest.mark.parametrize("paper", sorted(PAPER_MM))
@pytest.mark.parametrize("size_mm", [40.0, 60.0])
def test_rasterised_page_markers_detectable_at_true_size(paper, size_mm):
    cfg = MarkerConfig(size_mm=size_mm)
    page = render_page(cfg, paper, DPI)
    for img in (np.asarray(page), np.asarray(bilevel(page).convert("L"))):
        found = detect_markers(make_detector(cfg), img)
        assert set(found) == {cfg.tip_id, cfg.throat_id}
        for quad in found.values():
            sides = np.hypot(*(quad - np.roll(quad, -1, axis=0)).T)
            side_mm = sides * 25.4 / DPI
            assert np.all(np.abs(side_mm - size_mm) < 0.3), side_mm
        # Markers sit apart, with quiet zones that do not overlap.
        a, b = (marker_centre(found[i]) for i in (cfg.tip_id, cfg.throat_id))
        assert np.hypot(*(a - b)) * 25.4 / DPI > size_mm * 1.5


def test_page_has_a_100mm_ruler():
    page = np.asarray(render_page(MarkerConfig(), "a4", DPI))
    # The ruler is the lowest long horizontal black run on the page.
    rows = [(y, np.flatnonzero(page[y] < 128)) for y in range(page.shape[0])]
    runs = [(y, xs) for y, xs in rows if len(xs) and xs[-1] - xs[0] + 1 == len(xs)]
    y, xs = max((r for r in runs if len(r[1]) > 500), key=lambda r: r[0])
    assert (len(xs) - 1) * 25.4 / DPI == pytest.approx(100.0, abs=0.2)


def test_marker_too_big_for_page():
    with pytest.raises(ValueError, match="do not fit"):
        render_page(MarkerConfig(size_mm=200.0), "a4", DPI)


def test_bad_paper():
    with pytest.raises(ValueError, match="paper"):
        render_page(MarkerConfig(), "a3", DPI)


def test_cli(tmp_path, capsys):
    out = tmp_path / "m.pdf"
    assert main(["-o", str(out), "--paper", "letter", "--dpi", "150"]) == 0
    assert out.read_bytes().startswith(b"%PDF-")
    assert "letter" in capsys.readouterr().out
