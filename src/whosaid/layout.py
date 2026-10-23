"""Caption placement: lock to a face, slide to the exit edge when the face leaves, avoid overlaps."""
from __future__ import annotations

from dataclasses import dataclass

from .faces import Track
from .media import ASD_FPS

EDGE_FRAC = 0.22      # a face last seen within this fraction of a frame edge is treated as exiting through it
SLIDE_FRAMES = 8      # frames to glide from last face position to the edge
FACE_GAP = 1.15       # caption top = face centre + FACE_GAP * half-size (just below the chin)


@dataclass
class Anchor:
    x: float          # horizontal centre of the caption box
    y: float          # top of the caption box
    leaving: str | None = None  # edge the speaker left through, once sliding/settled there


def exit_edge(track: Track, W: int, H: int) -> str | None:
    x, y = float(track.cx[-1]), float(track.cy[-1])
    dist = {"left": x / W, "right": (W - x) / W, "top": y / H, "bottom": (H - y) / H}
    edge, d = min(dist.items(), key=lambda kv: kv[1])
    return edge if d <= EDGE_FRAC else None


def _ease(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def face_anchor(track: Track, f: int) -> tuple[float, float]:
    i = min(max(f - track.start, 0), len(track.frames) - 1)
    return float(track.cx[i]), float(track.cy[i] + FACE_GAP * track.s[i])


def anchor_at(track: Track, f: int, W: int, H: int, edge: str | None) -> Anchor:
    if f <= track.end:
        x, y = face_anchor(track, f)
        return Anchor(x, y)
    x, y = face_anchor(track, track.end)
    if edge is None:
        return Anchor(x, y)  # vanished mid-frame (turned away, occluded): hold position
    p = _ease((f - track.end) / SLIDE_FRAMES)
    tx, ty = {"left": (0.0, y), "right": (float(W), y), "top": (x, 0.0), "bottom": (x, float(H))}[edge]
    return Anchor(x + (tx - x) * p, y + (ty - y) * p, leaving=edge if p > 0 else None)


def place_box(a: Anchor, bw: float, bh: float, W: int, H: int, margin: float) -> list[float]:
    x = min(max(a.x - bw / 2, margin), W - bw - margin)
    y = min(max(a.y, margin), H - bh - margin)
    return [x, y]


def resolve_collisions(boxes: list[tuple[float, float, float, float]], H: int, margin: float, gap: float = 6.0):
    """boxes: (x, y, w, h) in priority order. Returns adjusted y per box; earlier boxes keep their place."""
    ys = [b[1] for b in boxes]
    for i in range(1, len(boxes)):
        for _ in range(4):
            moved = False
            for j in range(i):
                xi, wi, hi = boxes[i][0], boxes[i][2], boxes[i][3]
                xj, wj, hj = boxes[j][0], boxes[j][2], boxes[j][3]
                if xi < xj + wj and xj < xi + wi and ys[i] < ys[j] + hj + gap and ys[j] < ys[i] + hi + gap:
                    down = ys[j] + hj + gap
                    ys[i] = down if down + hi <= H - margin else max(margin, ys[j] - hi - gap)
                    moved = True
            if not moved:
                break
    return ys


def visible(chunk_start: float, chunk_end: float, t: float, lead: float = 0.05, linger: float = 0.35) -> bool:
    return chunk_start - lead <= t <= chunk_end + linger
