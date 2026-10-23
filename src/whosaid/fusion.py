"""Fuse the speaking detector's score with how much of a face's own voice the extractor finds in the mix.

The detector (LR-ASD) mostly checks "lips are moving and speech is audible". A listener whose lips move while somebody
else talks fools it. Dolphin extracts, from the mixture, the voice that matches *one face's lips*, so the energy share of
that extracted voice is a lip-to-voice consistency check: near zero for a mouthing listener, large for the real speaker.
The two signals fail on different frames, so a small logistic model on both beats either alone.

Features per tracked face and 25-fps frame (only for faces that were separated):
  asd        the detector's score (canonical scale, see calibrate.py)
  lr         log of the extracted-voice RMS share of the mix around the frame (+-0.2 s)
  asd_x_lr   interaction
  share      this face's ratio / sum of all separated faces' ratios (who is the loudest source)
  lr_other   log of the strongest *other* separated face's ratio (someone else clearly talking)
The fused logit is mapped to the same canonical scale as the detector score, so every downstream threshold is unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
FPS = 25
CTX = 0.2
ALL_FEATURES = ["asd", "lr", "asd_x_lr", "share", "lr_other"]
EPS = 1e-3


def frame_ratio(track, stem: np.ndarray, mix: np.ndarray, ctx: float = CTX) -> np.ndarray:
    """RMS(stem) / RMS(mix) in a window of one frame +-ctx seconds around each frame of the track."""
    cs = np.concatenate([[0.0], np.cumsum(stem.astype(np.float64) ** 2)])
    cm = np.concatenate([[0.0], np.cumsum(mix.astype(np.float64) ** 2)])
    hop = SR // FPS
    c = (track.start + np.arange(len(track.frames))) * hop + hop // 2
    a = np.clip(c - int((ctx + 0.02) * SR), 0, len(mix))
    b = np.clip(c + int((ctx + 0.02) * SR), 0, len(mix))
    num, den = cs[np.minimum(b, len(cs) - 1)] - cs[np.minimum(a, len(cs) - 1)], cm[b] - cm[a]
    return np.sqrt(np.maximum(num, 0.0) / (den + 1e-9))


def scene_features(tracks, stems: dict, asd: dict[int, np.ndarray], mix: np.ndarray) -> dict[int, np.ndarray]:
    """Feature matrix (n_frames, len(ALL_FEATURES)) for every track that has a stem."""
    ids = [t.id for t in tracks if t.id in stems]
    by_id = {t.id: t for t in tracks}
    ratio = {i: frame_ratio(by_id[i], stems[i], mix) for i in ids}
    n_abs = max((by_id[i].end + 1 for i in ids), default=0)
    absr = {i: np.zeros(n_abs) for i in ids}
    for i in ids:
        absr[i][by_id[i].start: by_id[i].end + 1] = ratio[i]
    total = sum(absr.values()) if ids else np.zeros(0)
    out = {}
    for i in ids:
        t = by_id[i]
        f = np.arange(t.start, t.end + 1)
        others = [absr[j][f] for j in ids if j != i]
        other_max = np.max(others, axis=0) if others else np.zeros(len(f))
        lr = np.log(ratio[i] + EPS)
        a = np.asarray(asd[i], float)[: len(f)]
        n = min(len(a), len(f))
        out[i] = np.stack([a[:n], lr[:n], a[:n] * lr[:n], (ratio[i] / (total[f] + 1e-6))[:n], np.log(other_max + EPS)[:n]], 1)
    return out


@dataclass
class Fuser:
    features: list[str]
    coef: list[float]
    intercept: float
    meta: dict            # t0 / gap of the fused logit on its training frames, for the canonical mapping

    def columns(self) -> list[int]:
        return [ALL_FEATURES.index(f) for f in self.features]

    def logit(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.columns()] @ np.asarray(self.coef) + self.intercept

    def score(self, X: np.ndarray) -> np.ndarray:
        from .calibrate import to_canonical

        return to_canonical(self.logit(X), self.meta)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(dict(features=self.features, coef=self.coef, intercept=self.intercept, meta=self.meta), indent=1))

    @classmethod
    def load(cls, path) -> "Fuser":
        d = json.loads(Path(path).read_text())
        return cls(d["features"], d["coef"], d["intercept"], d["meta"])


def fit(X: np.ndarray, y: np.ndarray, features: list[str], C: float = 1.0) -> Fuser:
    """Logistic regression on standardised features; coefficients are folded back to raw feature units."""
    from sklearn.linear_model import LogisticRegression

    from .calibrate import youden_threshold

    cols = [ALL_FEATURES.index(f) for f in features]
    Xs = X[:, cols]
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    m = LogisticRegression(C=C, max_iter=1000).fit((Xs - mu) / sd, y)
    coef = (m.coef_[0] / sd)
    intercept = float(m.intercept_[0] - (coef * mu).sum())
    z = Xs @ coef + intercept
    meta = dict(t0=youden_threshold(z[y == 1], z[y == 0]), gap=float(z[y == 1].mean() - z[y == 0].mean()))
    return Fuser(features, coef.tolist(), intercept, meta)


def apply_fusion(tracks, stems: dict, mix: np.ndarray, fuser: Fuser) -> int:
    """In place: faces with a stem get the fused score (the detector's own stays in Track.score_asd). Returns how many."""
    asd = {t.id: np.asarray(t.score, float) for t in tracks if t.score is not None and t.id in stems}
    feats = scene_features(tracks, stems, asd, mix)
    n = 0
    for t in tracks:
        if t.id in feats:
            if getattr(t, "score_asd", None) is None:
                t.score_asd = t.score
            fused = fuser.score(feats[t.id])
            t.score = np.concatenate([fused, np.asarray(t.score_asd, float)[len(fused):]])   # rare length mismatch: keep the rest
            n += 1
    return n
