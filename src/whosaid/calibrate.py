"""Speaking-score calibration, so the pipeline's thresholds mean the same thing for any detector and any face.

The pipeline's constants (speaking if raw > 1.0, "clearly not speaking" below -0.5, margins in logits ...) were tuned
for the pretrained LR-ASD score scale. A different detector (or the same detector on a person whose mouth/voice it
scores systematically low) breaks them. Two corrections, applied in memory to `Track.score`:

 1. model level   s -> 1 + (s - t0) * G / g
    t0 is the detector's best speaking/non-speaking threshold and g its mean score gap, both measured once on labelled
    validation frames (`compute_meta`); G = 1.8 is the pretrained detector's gap, the scale the pipeline constants were
    tuned on. After this, "1.0" is that detector's own decision threshold and a speaking-to-other swing is ~1.8 units
    for every detector.
 2. face level    s -> s - shift_f
    Per face: if its own scores form two clear groups (silent / speaking) the threshold moves to the valley between
    them, clipped to +-SHIFT_MAX so a noisy face cannot drag the threshold anywhere. A face that is not clearly
    bimodal (always speaking, never speaking, too little data) is left alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

G_CAN = 1.8        # canonical speaking-vs-other gap = the pretrained detector's measured gap on validation frames (1.81),
                   # i.e. the scale the pipeline constants were tuned on, so the pretrained model maps almost to itself
T_CAN = 1.0        # canonical decision threshold
ETA_MIN = 0.45     # a 2-way split must explain this share of the face's score variance to count as bimodal
MASS_MIN = 0.08    # ...and each group must hold at least this share of the frames
GAP_MIN = 1.0      # ...and the groups' means must be this far apart (canonical units)
SHIFT_MAX = 1.0    # per-face threshold may move at most this far (canonical units)
MIN_FRAMES = 100   # 4 s of face


def meta_path(model_path) -> Path:
    return Path(model_path).with_suffix(".meta.json")


def default_meta() -> dict:
    return dict(t0=T_CAN, gap=G_CAN, source="default (identity)")


def load_meta(model_path) -> dict:
    p = meta_path(model_path)
    return json.loads(p.read_text()) if p.exists() else default_meta()


def youden_threshold(pos: np.ndarray, neg: np.ndarray) -> float:
    """Threshold maximising TPR - FPR."""
    cand = np.quantile(np.r_[pos, neg], np.linspace(0.02, 0.98, 97))
    j = [(pos > c).mean() - (neg > c).mean() for c in cand]
    return float(cand[int(np.argmax(j))])


def compute_meta(model_path, val_dir, log=print) -> dict:
    """Measure t0 and gap on labelled validation frames, scoring exactly as the pipeline does, and store them."""
    from .asd import score_track
    from .asd_train import load_scenes
    from .lrasd.model import load_model

    model = load_model(model_path).cpu().eval()
    pos, neg = [], []
    for s in load_scenes(Path(val_dir), "val_"):
        for tid, tr in s.tracks.items():
            sc = score_track(model, s.mfcc, tr["crops"], tr["start"])
            y = tr["y"][: len(sc)].astype(bool)
            real = tr["cat"][: len(sc)] != 4                    # ignore the false-face frames
            pos.append(sc[y & real])
            neg.append(sc[~y & real])
    pos, neg = np.concatenate(pos), np.concatenate(neg)
    qs = np.linspace(0, 100, 101)
    meta = dict(t0=youden_threshold(pos, neg), gap=float(pos.mean() - neg.mean()), n_pos=int(len(pos)), n_neg=int(len(neg)),
                source=f"validation frames of {Path(val_dir)}",
                pos_q=np.percentile(pos, qs).round(3).tolist(), neg_q=np.percentile(neg, qs).round(3).tolist())
    meta["recall_at_t0"] = float((pos > meta["t0"]).mean())
    meta_path(model_path).write_text(json.dumps(meta, indent=1))
    log(f"{Path(model_path).name}: t0 {meta['t0']:+.2f}, gap {meta['gap']:.2f} ({len(pos)} speaking / {len(neg)} other frames)")
    return meta


def with_recall(meta: dict, recall: float) -> dict:
    """Same detector, but the threshold t0 is the score above which `recall` of validation speaking frames lie.
    Word attribution cares more about not missing a speaker (words only occur while someone talks) than about firing
    during pauses, so a recall target can be a better operating point than Youden's balanced one."""
    q = np.asarray(meta["pos_q"])
    t0 = float(np.interp(100 * (1 - recall), np.arange(101), q))
    return dict(meta, t0=t0, recall_target=recall, source=meta.get("source", "") + f"; t0 at recall {recall:.2f}")


def load_meta_for(model_path, recall: float | None = None, match_pretrained: bool = False) -> dict:
    """Meta for a detector. recall=R sets its threshold at speaking-recall R; match_pretrained=True uses the recall the
    pretrained detector has at its own threshold (so every detector is run at the pretrained operating point)."""
    from .paths import ASD_MODEL

    m = load_meta(model_path)
    if match_pretrained:
        recall = load_meta(ASD_MODEL).get("recall_at_t0")
    return with_recall(m, recall) if recall and "pos_q" in m else m


def to_canonical(scores: np.ndarray, meta: dict) -> np.ndarray:
    return T_CAN + (scores - meta["t0"]) * (G_CAN / max(meta["gap"], 1e-3))


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def face_shift(scores: np.ndarray) -> tuple[float, dict]:
    """How far to move this face's threshold (canonical units). 0 when the face is not clearly two-humped."""
    x = _smooth(np.asarray(scores, float))
    info = dict(bimodal=False)
    if len(x) < MIN_FRAMES:
        return 0.0, info
    c0, c1 = np.percentile(x, 15), np.percentile(x, 85)
    if c1 - c0 < 1e-6:
        return 0.0, info
    for _ in range(25):                                         # 1-D 2-means
        hi = np.abs(x - c1) < np.abs(x - c0)
        if hi.all() or (~hi).all():
            return 0.0, info
        c0, c1 = x[~hi].mean(), x[hi].mean()
    hi = np.abs(x - c1) < np.abs(x - c0)
    p1 = float(hi.mean())
    between = p1 * (1 - p1) * (c1 - c0) ** 2
    eta = between / (x.var() + 1e-9)
    info = dict(bimodal=False, c0=float(c0), c1=float(c1), p_speaking=p1, eta=float(eta))
    if eta < ETA_MIN or min(p1, 1 - p1) < MASS_MIN or (c1 - c0) < GAP_MIN:
        return 0.0, info
    valley = (c0 + c1) / 2
    shift = float(np.clip(valley - T_CAN, -SHIFT_MAX, SHIFT_MAX))
    info.update(bimodal=True, valley=float(valley), shift=shift)
    return shift, info


def calibrate_tracks(tracks, meta: dict, mode: str = "face") -> dict:
    """In place. mode: 'none' (leave raw), 'model' (layer 1 only), 'face' (layers 1 and 2). Returns per-track info."""
    infos = {}
    if mode == "none":
        return infos
    for t in tracks:
        if t.score is None:
            continue
        if getattr(t, "score_raw", None) is None:
            t.score_raw = t.score
        s = to_canonical(np.asarray(t.score_raw, float), meta)
        if mode == "face":
            shift, info = face_shift(s)
            s = s - shift
            infos[t.id] = dict(info, applied_shift=shift)
        t.score = s
    return infos
