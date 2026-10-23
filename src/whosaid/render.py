"""Burn speaker-locked captions into the video."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .attribute import Chunk
from .faces import Track
from .layout import anchor_at, exit_edge, place_box, resolve_collisions, visible
from .media import ASD_FPS, VideoInfo, iter_frames

PALETTE = [(255, 214, 10), (64, 200, 224), (255, 105, 97), (120, 224, 143), (200, 150, 255), (255, 159, 67)]
UNKNOWN = (190, 190, 190)
FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


class _Layouts:
    """Per-chunk word wrapping, computed once."""

    def __init__(self, font, max_w: float, pad: int):
        self.font, self.max_w, self.pad, self.cache = font, max_w, pad, {}
        self.space = font.getlength(" ")
        self.line_h = int(font.size * 1.25)

    def get(self, chunk: Chunk):
        key = id(chunk)
        if key in self.cache:
            return self.cache[key]
        lines, cur, x = [], [], 0.0
        for i, w in enumerate(chunk.words):
            wl = self.font.getlength(w.text)
            if cur and x + wl > self.max_w:
                lines.append((cur, x - self.space))
                cur, x = [], 0.0
            cur.append((i, w.text, x))
            x += wl + self.space
        lines.append((cur, x - self.space))
        bw = max(l[1] for l in lines) + 2 * self.pad
        bh = len(lines) * self.line_h + 2 * self.pad
        self.cache[key] = (lines, bw, bh)
        return self.cache[key]


def _alpha(c: Chunk, t: float, fade_in: float = 0.12, fade_out: float = 0.25) -> float:
    """Soft appear/disappear so captions don't pop."""
    a_in = (t - (c.start - 0.05)) / fade_in
    a_out = (c.show_end - t) / fade_out
    return float(min(1.0, max(0.0, min(a_in, a_out))))


def render(video, info: VideoInfo, out_path, tracks: list[Track], chunks: list[Chunk], debug: bool = False, progress=None,
           style: str = "outline"):
    """style 'outline' (default): outlined text, no box, words appear as they are spoken, small type.
    style 'box': the original dark box with all words of the chunk visible."""
    W, H = info.width, info.height
    outline = style == "outline"
    font = load_font(max(15, int(min(H, W * 1.3) * (0.031 if outline else 0.034))))
    pad = max(2, font.size // 6) if outline else max(6, font.size // 3)
    stroke = max(2, round(font.size / 9)) if outline else 0
    margin = max(8, int(0.02 * H))
    max_w = ((0.30 if outline else 0.42) if W >= H else (0.62 if outline else 0.8)) * W
    layouts = _Layouts(font, max_w, pad)
    by_id = {t.id: t for t in tracks}
    edges = {t.id: exit_edge(t, W, H) for t in tracks}
    color_of = lambda spk: PALETTE[spk % len(PALETTE)] if spk is not None else UNKNOWN

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{info.fps}", "-i", "-",
         "-i", str(video), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out_path)],
        stdin=subprocess.PIPE,
    )
    n = int(info.duration * info.fps)
    try:
        for i, frame in enumerate(iter_frames(video, info, info.fps)):
            t = i / info.fps
            f25 = int(t * ASD_FPS)
            active = [c for c in chunks if visible(c.start, c.show_end, t, linger=0.0)]
            if active or debug:
                img = Image.fromarray(np.ascontiguousarray(frame[..., ::-1])).convert("RGBA")
                over = Image.new("RGBA", img.size, (0, 0, 0, 0))
                d = ImageDraw.Draw(over)
                if debug:
                    for tr in tracks:
                        p = tr.at(f25)
                        if p:
                            sc = float(tr.score[f25 - tr.start]) if tr.score is not None else 0.0
                            col = (0, 255, 0, 255) if sc > 0 else (255, 60, 60, 255)
                            d.rectangle([p[0] - p[2], p[1] - p[2], p[0] + p[2], p[1] + p[2]], outline=col, width=3)
                            d.text((p[0] - p[2], p[1] - p[2] - font.size), f"#{tr.id} {sc:+.1f}", font=font, fill=col)
                items = []
                for c in active:
                    lines, bw, bh = layouts.get(c)
                    if c.speaker is None:
                        a_x, a_y, leaving = W / 2, H - margin - bh, None
                        from .layout import Anchor
                        anchor = Anchor(a_x, a_y)
                    else:
                        anchor = anchor_at(by_id[c.speaker], f25, W, H, edges[c.speaker])
                    x, y = place_box(anchor, bw, bh, W, H, margin)
                    items.append((c, lines, bw, bh, x, y, anchor))
                ys = resolve_collisions([(it[4], it[5], it[2], it[3]) for it in items], H, margin)
                for (c, lines, bw, bh, x, _, anchor), y in zip(items, ys):
                    col = color_of(c.speaker)
                    k = _alpha(c, t)
                    a = int(255 * k)
                    if not outline:
                        d.rounded_rectangle([x, y, x + bw, y + bh], radius=pad, fill=(0, 0, 0, int(170 * k)))
                        d.rounded_rectangle([x, y, x + bw, y + 4], radius=2, fill=col + (a,))
                    if anchor.leaving:
                        _arrow(d, anchor.leaving, x, y, bw, bh, col + (a,), pad)
                    for li, (cells, lw) in enumerate(lines):
                        lx = x + pad + (bw - 2 * pad - lw) / 2
                        for wi, text, wx in cells:
                            w = c.words[wi]
                            if outline and w.start > t:
                                continue          # not spoken yet: don't cover the video with it
                            if w.start <= t <= w.end:
                                fill = col + (a,)
                            elif w.end < t:
                                fill = (255, 255, 255, a)
                            else:
                                fill = (165, 165, 165, a)
                            d.text((lx + wx, y + pad + li * layouts.line_h), text, font=font, fill=fill,
                                   stroke_width=stroke, stroke_fill=(0, 0, 0, int(a * 0.9)))
                frame = np.asarray(Image.alpha_composite(img, over).convert("RGB"))[..., ::-1]
            enc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if progress and i % 25 == 0:
                progress(i, n)
    finally:
        enc.stdin.close()
        enc.wait()


def _arrow(d, edge, x, y, bw, bh, col, pad):
    cy, cx, s = y + bh / 2, x + bw / 2, pad * 0.9
    pts = {
        "left": [(x - 2, cy), (x - 2 - s, cy - s * 0.8), (x - 2 - s, cy + s * 0.8)],
        "right": [(x + bw + 2, cy), (x + bw + 2 + s, cy - s * 0.8), (x + bw + 2 + s, cy + s * 0.8)],
        "top": [(cx, y - 2), (cx - s * 0.8, y - 2 - s), (cx + s * 0.8, y - 2 - s)],
        "bottom": [(cx, y + bh + 2), (cx - s * 0.8, y + bh + 2 + s), (cx + s * 0.8, y + bh + 2 + s)],
    }[edge]
    d.polygon(pts, fill=col if len(col) == 4 else col + (255,))
