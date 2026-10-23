"""Active speaker detection: score every face track with LR-ASD."""
from __future__ import annotations

import numpy as np

from .faces import Track
from .media import ASD_FPS, VideoInfo, iter_frames, read_wav
from .paths import ASD_MODEL

CROP = 112
DURATIONS = (1, 2, 3, 4, 6)  # seconds; scores are averaged across chunk sizes like upstream eval


def crop_mouth_region(gray: np.ndarray, cx: float, cy: float, s: float) -> np.ndarray:
    """Crop the 112x112 lower-face window LR-ASD was trained on (matches upstream's crop-then-centre-crop)."""
    import cv2

    x0, x1 = cx - 0.7 * s, cx + 0.7 * s
    y0, y1 = cy - 0.3 * s, cy + 1.1 * s
    sx, sy = CROP / (x1 - x0), CROP / (y1 - y0)
    m = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]], dtype=np.float32)
    return cv2.warpAffine(gray, m, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=110)


def collect_crops(video, info: VideoInfo, tracks: list[Track], lips: bool = True) -> dict[int, np.ndarray]:
    import cv2

    from .lips import LIP, lip_crop

    crops = {t.id: np.zeros((len(t.frames), CROP, CROP), np.uint8) for t in tracks}
    if lips:
        for t in tracks:
            t.lips = np.zeros((len(t.frames), LIP, LIP), np.uint8)
    for f, frame in enumerate(iter_frames(video, info, ASD_FPS)):
        active = [t for t in tracks if t.start <= f <= t.end]
        if not active:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for t in active:
            i = f - t.start
            crops[t.id][i] = crop_mouth_region(gray, t.cx[i], t.cy[i], t.s[i])
            if lips:
                t.lips[i] = lip_crop(gray, t.lm[i])
    return crops


def mfcc_features(wav_path) -> np.ndarray:
    import python_speech_features

    audio = read_wav(wav_path)
    return python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)


def score_track(model, mfcc: np.ndarray, crops: np.ndarray, start_frame: int) -> np.ndarray:
    import torch

    a = mfcc[start_frame * 4 : (start_frame + len(crops)) * 4]
    n = min(len(a) // 4, len(crops))
    if n < 4:
        return np.full(len(crops), -5.0)
    a, v = a[: n * 4], crops[:n]
    runs = []
    for dur in DURATIONS:
        out = []
        for i in range(int(np.ceil(n / (dur * ASD_FPS)))):
            lo, hi = i * dur * ASD_FPS, min((i + 1) * dur * ASD_FPS, n)
            if hi - lo < 1:
                continue
            ia = torch.from_numpy(a[lo * 4 : hi * 4]).float().unsqueeze(0)
            iv = torch.from_numpy(v[lo:hi]).float().unsqueeze(0)
            out.append(model.score(ia, iv).numpy())
        runs.append(np.concatenate(out))
    scores = np.mean(runs, axis=0)
    if len(scores) < len(crops):
        scores = np.pad(scores, (0, len(crops) - len(scores)), mode="edge")
    return scores


def score_tracks(video, info: VideoInfo, tracks: list[Track], wav_path, model_path=ASD_MODEL, progress=None, lips: bool = True) -> None:
    from .lrasd.model import load_model

    if not tracks:
        return
    model = load_model(model_path)
    crops = collect_crops(video, info, tracks, lips=lips)
    mfcc = mfcc_features(wav_path)
    for k, t in enumerate(tracks):
        t.score = score_track(model, mfcc, crops[t.id], t.start)
        if progress:
            progress(k + 1, len(tracks))
