"""Static faces: photos, posters, wall portraits and frozen overlays that the face detector mistakes for people.

Seen on a real clip (a Turkish ranking game with actor photos pasted on the video): 54 face tracks, several of them
photos, one of which was credited with 70 words. A photo's face does not move and its mouth never changes, while a
person who ever speaks moves their mouth. So a track is "static" when its *mouth crop almost never changes* (90th
percentile of the frame-to-frame change, so a person who speaks only now and then is still safe) and the face itself
hardly moves.

Measured on the synthetic benchmark's ground truth (secondary faces = wall portraits / background photos): on dev the
rule catches 13/29 of them and wrongly flags 0/71 speaking people; on held-out test 4/5 and 0/23. It is deliberately
conservative: loosening the lip threshold to 2.0 starts excluding real speakers.
"""
from __future__ import annotations

import numpy as np

LIP_P90_MAX = 1.5    # 90th percentile of the (5-frame smoothed) mean abs frame-to-frame change of the 88x88 mouth crop
MOVE_MAX = 0.10      # face-centre std / face size
MIN_FRAMES = 50      # 2 s: shorter tracks are not judged


def static_stats(track) -> tuple[float, float] | None:
    """(lip_p90, move) for a track, or None if it has no mouth crops / is too short to judge."""
    if track.lips is None or len(track.frames) < MIN_FRAMES:
        return None
    d = np.abs(np.diff(track.lips.astype(np.float32), axis=0)).mean(axis=(1, 2))
    d = np.convolve(d, np.ones(5) / 5, mode="same")
    move = float(np.sqrt(np.var(track.cx) + np.var(track.cy)) / max(float(np.median(track.s)), 1e-6))
    return float(np.percentile(d, 90)), move


def is_static(track) -> bool:
    st = static_stats(track)
    return st is not None and st[0] < LIP_P90_MAX and st[1] < MOVE_MAX


def mark_static(tracks) -> list[int]:
    """In place: static faces get `static=True` and a constant 'not speaking' score. Returns their ids."""
    ids = []
    for t in tracks:
        if is_static(t):
            t.static = True
            if t.score is not None:
                if getattr(t, "score_raw", None) is None:
                    t.score_raw = t.score
                t.score = np.full(len(t.frames), -5.0)
            ids.append(t.id)
    return ids
