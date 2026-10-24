"""Write a printable PDF with the tip and throat ArUco markers at true size.

    uv run python -m tools.print_markers                  # markers.pdf, A4
    uv run python -m tools.print_markers --paper letter -o out.pdf

Print at 100% ("Actual size", no "Fit to page") and check the 100 mm ruler line with a
real ruler before taping the markers on: the marker size feeds nothing in the app
(the pose uses marker centres), but a wrongly scaled print is usually also blurry or
clipped, and the ruler is the quickest way to spot it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from racketguitar.config import MarkerConfig, load_config

__all__ = ["PAPER_MM", "bilevel", "render_page", "write_pdf", "main"]

PAPER_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}
MARGIN_MM = 10.0
QUIET_ZONE_FRACTION = 0.25  # white border on each side, as a fraction of the marker side
GAP_MM = 8.0  # between the two marker cells
RULER_MM = 100.0
LABEL_PT = 14.0
SMALL_PT = 10.0


def _px(mm: float, dpi: int) -> int:
    return round(mm / 25.4 * dpi)


def _font(pt: float, dpi: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=max(8, round(pt / 72.0 * dpi)))


def _dashed_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], dash: int) -> None:
    x0, y0, x1, y1 = box
    for x in range(x0, x1, 2 * dash):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=0, width=1)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=0, width=1)
    for y in range(y0, y1, 2 * dash):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=0, width=1)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=0, width=1)


def render_page(cfg: MarkerConfig | None = None, paper: str = "a4", dpi: int = 300) -> Image.Image:
    """The page as a grayscale image whose pixel size matches ``paper`` at ``dpi``."""
    cfg = cfg or MarkerConfig()
    if paper not in PAPER_MM:
        raise ValueError(f"paper must be one of {sorted(PAPER_MM)}, not {paper!r}")
    page_w_mm, page_h_mm = PAPER_MM[paper]
    page = Image.new("L", (_px(page_w_mm, dpi), _px(page_h_mm, dpi)), 255)
    draw = ImageDraw.Draw(page)
    label, small = _font(LABEL_PT, dpi), _font(SMALL_PT, dpi)
    margin = _px(MARGIN_MM, dpi)
    line_h = _px(SMALL_PT * 0.5, dpi)  # about 1.4x the 10 pt text height

    y = margin
    draw.text((margin, y), "Racket Guitar markers", font=label, fill=0)
    y += round(line_h * 1.6)
    notes = (
        f"Print at 100% (Actual size). Each black square must measure {cfg.size_mm:g} mm.",
        "Cut along the dashed lines and keep the white border. Orientation does not matter.",
        "TIP goes at the top of the head frame, THROAT at the throat, both facing the camera.",
    )
    for note in notes:
        draw.text((margin, y), note, font=small, fill=0)
        y += line_h
    y += line_h

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
    side = _px(cfg.size_mm, dpi)
    quiet = _px(cfg.size_mm * QUIET_ZONE_FRACTION, dpi)
    cell = side + 2 * quiet
    gap = _px(GAP_MM, dpi)
    usable_w = page.width - 2 * margin
    row = 2 * cell + gap <= usable_w
    ruler_space = 4 * line_h
    needed_h = (cell + 2 * line_h) * (1 if row else 2) + (0 if row else gap) + ruler_space
    if cell > usable_w or y + needed_h > page.height - margin:
        raise ValueError(f"{cfg.size_mm:g} mm markers do not fit on {paper}")

    markers = ((cfg.tip_id, "TIP"), (cfg.throat_id, "THROAT"))
    for i, (marker_id, name) in enumerate(markers):
        if row:
            x0, y0 = margin + i * (cell + gap), y
        else:
            x0, y0 = margin, y + i * (cell + gap + 2 * line_h)
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, side, borderBits=1)
        page.paste(Image.fromarray(img), (x0 + quiet, y0 + quiet))
        _dashed_rect(draw, (x0, y0, x0 + cell, y0 + cell), dash=_px(2.0, dpi))
        text = f"{name}  id {marker_id}  ({cfg.dictionary}, {cfg.size_mm:g} mm)"
        draw.text((x0, y0 + cell + line_h // 3), text, font=small, fill=0)
    y += (cell + 2 * line_h) * (1 if row else 2) + (0 if row else gap) + line_h

    # Measured ruler: a 100 mm line with a tick every 10 mm.
    x0, x1 = margin, margin + _px(RULER_MM, dpi)
    tick = _px(3.0, dpi)
    width = max(1, _px(0.3, dpi))
    draw.line([(x0, y), (x1, y)], fill=0, width=width)
    for k in range(int(RULER_MM // 10) + 1):
        x = x0 + _px(10.0 * k, dpi)
        draw.line([(x, y - (2 * tick if k % 5 == 0 else tick)), (x, y)], fill=0, width=width)
    draw.text(
        (x0, y + line_h // 3),
        f"This line must measure exactly {RULER_MM:g} mm. If not, the printer scaled the page.",
        font=small,
        fill=0,
    )
    return page


def bilevel(page: Image.Image) -> Image.Image:
    """Threshold to 1-bit, which Pillow stores losslessly (fax encoding) instead of JPEG."""
    return page.point(lambda v: 255 if v >= 128 else 0).convert("1", dither=Image.Dither.NONE)


def write_pdf(
    path: str | Path, cfg: MarkerConfig | None = None, paper: str = "a4", dpi: int = 300
) -> Path:
    """Render the page and save it as a one-page PDF at true physical size."""
    path = Path(path)
    page = bilevel(render_page(cfg, paper, dpi))
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path, "PDF", resolution=float(dpi))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default="markers.pdf")
    ap.add_argument("--paper", choices=sorted(PAPER_MM), default="a4")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--config", help="settings TOML (default: ~/.racketguitar/config.toml)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config).markers
    out = write_pdf(args.out, cfg, args.paper, args.dpi)
    print(
        f"wrote {out} ({args.paper}, markers {cfg.size_mm:g} mm, ids {cfg.tip_id}/{cfg.throat_id})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
