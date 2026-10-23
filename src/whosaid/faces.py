"""Face detection (OpenCV YuNet, handles profile/turned faces) and a small IoU tracker.

Tracks are sampled at ASD_FPS (25 fps). Each track holds per-frame face geometry,
with short detection gaps linearly interpolated.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .media import ASD_FPS, VideoInfo, iter_frames
from .paths import FACE_MODEL

MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DETECT_WIDTH = 960  # detection runs on a downscaled frame


@dataclass
class Track:
    id: int
    frames: np.ndarray            # (n,) int frame indices at ASD_FPS, contiguous
    cx: np.ndarray                # face-box centre x, original-frame pixels
    cy: np.ndarray                # face-box centre y
    s: np.ndarray                 # half-size of the face box (cheek-to-cheek / 2)
    observed: np.ndarray          # (n,) bool, False where interpolated
    score: np.ndarray | None = None  # filled by ASD: (n,) speaking logit (calibrated in place when asd_calib is on)
    score_raw: np.ndarray | None = None  # the detector's untouched output, kept when calibrating
    score_asd: np.ndarray | None = None  # the detector's (calibrated) score before voice-energy fusion
    static: bool = False             # a photo / poster / frozen overlay mistaken for a face (see photos.py)
    lm: np.ndarray | None = None     # (n, 5, 2) YuNet landmarks: eyes, nose tip, mouth corners (image coords)
    lips: np.ndarray | None = None   # (n, 88, 88) uint8 aligned mouth crops for Dolphin, filled with the ASD crops

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:  # inclusive
        return int(self.frames[-1])

    def at(self, frame: int):
        i = frame - self.start
        if 0 <= i < len(self.frames):
            return self.cx[i], self.cy[i], self.s[i]
        return None


@dataclass
class _Det:
    cx: float
    cy: float
    s: float
    lm: np.ndarray | None = None

    def box(self):
        return (self.cx - self.s, self.cy - self.s, self.cx + self.s, self.cy + self.s)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class _Live:
    id: int
    last: _Det
    last_frame: int
    pts: list = field(default_factory=list)  # (frame, cx, cy, s)


def _ensure_model(model_path: Path) -> Path:
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSL", "-o", str(model_path), MODEL_URL], check=True)
    return model_path


def _to_det(row, scale: float) -> _Det:
    x, y, w, h = (float(v) / scale for v in row[:4])
    return _Det(x + w / 2, y + h / 2, max(w, h) / 2, np.asarray(row[4:14], float).reshape(5, 2) / scale)


def track_faces(
    video: str | Path,
    info: VideoInfo,
    model_path: str | Path = FACE_MODEL,
    min_conf: float = 0.6,
    max_gap: int = 12,
    min_len: int = 12,
    iou_thresh: float = 0.15,
    progress=None,
) -> list[Track]:
    import cv2

    model_path = _ensure_model(Path(model_path))
    scale = min(1.0, DETECT_WIDTH / info.width)
    detector = None
    live: list[_Live] = []
    done: list[_Live] = []
    next_id = 0
    n_frames = int(info.duration * ASD_FPS)

    for f, small in enumerate(iter_frames(video, info, ASD_FPS, out_width=DETECT_WIDTH)):
        if detector is None:
            detector = cv2.FaceDetectorYN.create(
                str(model_path), "", (small.shape[1], small.shape[0]), min_conf, 0.3, 5000
            )
        _, rows = detector.detect(small)
        dets = [_to_det(r, scale) for r in (rows if rows is not None else [])]

        # greedy IoU association, best pairs first
        pairs = sorted(
            ((_iou(t.last.box(), d.box()), ti, di) for ti, t in enumerate(live) for di, d in enumerate(dets)),
            reverse=True,
        )
        used_t, used_d = set(), set()
        for iou, ti, di in pairs:
            if iou < iou_thresh:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            d = dets[di]
            live[ti].pts.append((f, d.cx, d.cy, d.s, *d.lm.ravel()))
            live[ti].last, live[ti].last_frame = d, f
        for di, d in enumerate(dets):
            if di not in used_d:
                live.append(_Live(next_id, d, f, [(f, d.cx, d.cy, d.s, *d.lm.ravel())]))
                next_id += 1
        still = []
        for t in live:
            (still if f - t.last_frame <= max_gap else done).append(t)
        live = still
        if progress and f % 25 == 0:
            progress(f, n_frames)
    done.extend(live)
    return _finalize(done, min_len)


def _finalize(raw: list[_Live], min_len: int) -> list[Track]:
    tracks = []
    for t in raw:
        if len(t.pts) < min_len:
            continue
        pts = np.array(t.pts, dtype=float)
        frames = np.arange(int(pts[0, 0]), int(pts[-1, 0]) + 1)
        cx, cy, s = (np.interp(frames, pts[:, 0], pts[:, k]) for k in (1, 2, 3))
        lm = np.stack([_smooth(np.interp(frames, pts[:, 0], pts[:, k]), 2.0) for k in range(4, 14)], 1).reshape(-1, 5, 2)
        observed = np.isin(frames, pts[:, 0].astype(int))
        cx, cy, s = (_smooth(a, 2.0) for a in (cx, cy, s))
        tracks.append(Track(len(tracks), frames, cx, cy, s, observed, lm=lm))
    return tracks


def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(a, sigma, mode="nearest") if len(a) > 1 else a
