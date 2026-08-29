"""Board figures showing what the model played against what the engine wants.

Rendering is `chessboard-image` (pure Pillow, no browser or system libraries); move
arrows are drawn on top. Colours follow the status palette used elsewhere in the
project: engine-preferred is the "good" hue, the model's own choice is drawn in the
"critical" hue only when it differs, so agreement and disagreement are distinguishable
without reading the caption.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import chess
from PIL import Image, ImageDraw, ImageFont

BOARD_PX = 480
COORD_PAD = 20            # chessboard-image adds a coordinate margin

C_BEST = (12, 163, 12)    # engine's preferred move
C_MODEL = (208, 59, 59)   # the model's move, when it differs
C_AGREE = (12, 163, 12)   # both agree -- one arrow
INK = (11, 11, 11)
INK_2 = (82, 81, 78)
SURFACE = (255, 255, 255)


def _font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _square_centre(sq: int, board_px: int, pad: int, flipped: bool) -> tuple[float, float]:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    if flipped:
        f, r = 7 - f, 7 - r
    cell = board_px / 8.0
    return pad + (f + 0.5) * cell, pad + (7 - r + 0.5) * cell


def _arrow(draw: ImageDraw.ImageDraw, p0, p1, colour, width=7, head=22, inset=16):
    """Arrow from square centre to square centre, pulled back so the piece stays visible."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    x0, y0 = x0 + ux * inset, y0 + uy * inset
    x1, y1 = x1 - ux * (head * 0.55), y1 - uy * (head * 0.55)
    draw.line([(x0, y0), (x1, y1)], fill=colour, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    for sign in (1, -1):
        a = ang + sign * math.radians(150)
        draw.line([(x1, y1), (x1 + head * math.cos(a), y1 + head * math.sin(a))],
                  fill=colour, width=width)
    draw.ellipse([x1 - width * 0.7, y1 - width * 0.7, x1 + width * 0.7, y1 + width * 0.7],
                 fill=colour)


@dataclass
class BoardPanel:
    fen: str
    best: str | None = None          # engine's preferred move, UCI
    played: str | None = None        # the model's move, UCI
    title: str = ""
    caption: str = ""


def render_panel(p: BoardPanel, size: int = BOARD_PX, theme: str = "wikipedia") -> Image.Image:
    import chessboard_image as cbi

    board = chess.Board(p.fen)
    flipped = not board.turn                    # orient the board for the side to move
    img = cbi.fen_to_pil(p.fen, size=size, theme_name=theme,
                         player_pov="white" if board.turn else "black",
                         show_coordinates=True).convert("RGB")
    pad = (img.size[0] - size) / 2
    draw = ImageDraw.Draw(img)

    agree = p.best and p.played and p.best == p.played
    if p.best:
        mv = chess.Move.from_uci(p.best)
        _arrow(draw, _square_centre(mv.from_square, size, pad, flipped),
               _square_centre(mv.to_square, size, pad, flipped),
               C_AGREE if agree else C_BEST)
    if p.played and not agree:
        try:
            mv = chess.Move.from_uci(p.played)
            _arrow(draw, _square_centre(mv.from_square, size, pad, flipped),
                   _square_centre(mv.to_square, size, pad, flipped), C_MODEL)
        except ValueError:
            pass
    return img


def compose(panels: list[BoardPanel], out_path: str, cols: int = 3,
            size: int = BOARD_PX, theme: str = "wikipedia") -> str:
    """Grid of boards with titles, captions and a legend."""
    f_title, f_cap, f_leg = _font(19), _font(15), _font(15)
    rendered = [render_panel(p, size, theme) for p in panels]
    bw, bh = rendered[0].size
    pad_x, pad_y, title_h, cap_h, legend_h = 22, 18, 30, 46, 40
    rows = (len(panels) + cols - 1) // cols
    W = cols * bw + (cols + 1) * pad_x
    H = legend_h + rows * (bh + title_h + cap_h + pad_y) + pad_y

    canvas = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(canvas)

    x = pad_x
    d.line([(x, 22), (x + 30, 22)], fill=C_BEST, width=6)
    d.text((x + 40, 13), "engine's move", font=f_leg, fill=INK)
    x += 190
    d.line([(x, 22), (x + 30, 22)], fill=C_MODEL, width=6)
    d.text((x + 40, 13), "model's move (when it differs)", font=f_leg, fill=INK)

    for i, (p, im) in enumerate(zip(panels, rendered)):
        r, c = divmod(i, cols)
        ox = pad_x + c * (bw + pad_x)
        oy = legend_h + r * (bh + title_h + cap_h + pad_y)
        d.text((ox, oy), p.title, font=f_title, fill=INK)
        canvas.paste(im, (ox, oy + title_h))
        ty = oy + title_h + bh + 6
        for line in p.caption.split("\n")[:2]:
            d.text((ox, ty), line, font=f_cap, fill=INK_2)
            ty += 19
    canvas.save(out_path)
    return out_path
