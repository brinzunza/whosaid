"""Voice identity: cluster the speech by speaker embedding, then tie each voice to a face.

Per-frame active speaker scores are noisy under crosstalk (a reacting face can out-score the real
speaker). Pooling evidence over *all* speech of one voice is much sturdier: a face belongs to the
voice it lights up for more than it does for the other voices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .asr import Word
from .faces import Track
from .media import ASD_FPS, SAMPLE_RATE, read_wav
from .paths import MODELS_DIR

WIN = 2.0            # s, embedding window (1.2s was too noisy on real, compressed audio)
HOP = 0.4            # s
MIN_SPEECH = 0.6     # fraction of a window that must be speech
MAX_VOICES = 5
MIN_SILHOUETTE = 0.15  # below this the speech is treated as a single voice
MIN_CLUSTER_WINDOWS = 4
MIN_CONTRAST = 0.5   # logits a track must light up more for a voice than for the others
SPEAK_THRESH = 0.0
MIN_WORDS = 2        # words needed to judge a (voice, track) pair


@dataclass
class VoiceMap:
    tracks_of: dict[int, list[int]]  # voice -> track ids
    contrast: dict[tuple[int, int], float]


def _speech_mask(words: list[Word], n: int, step: float, bridge: float = 0.3) -> np.ndarray:
    m = np.zeros(n, bool)
    prev_end = None
    for w in words:
        lo, hi = int(w.start / step), min(n, int(np.ceil(w.end / step)))
        m[lo:hi] = True
        if prev_end is not None and w.start - prev_end <= bridge:
            m[int(prev_end / step) : lo] = True
        prev_end = w.end
    return m


def embed_windows(wav_path, words: list[Word]):
    """Return (centres[s], embeddings[n, d]) for windows that are mostly speech."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    audio = read_wav(wav_path).astype(np.float32) / 32768.0
    dur = len(audio) / SAMPLE_RATE
    step = 0.01
    mask = _speech_mask(words, int(dur / step) + 2, step)
    starts = np.arange(0, max(dur - WIN, 0) + 1e-9, HOP)
    keep = [s for s in starts if mask[int(s / step) : int((s + WIN) / step)].mean() >= MIN_SPEECH]
    if not keep:
        return np.zeros(0), np.zeros((0, 192))
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(MODELS_DIR / "ecapa"), run_opts={"device": "cpu"}
    )
    n = int(WIN * SAMPLE_RATE)
    batch = torch.from_numpy(np.stack([audio[int(s * SAMPLE_RATE) : int(s * SAMPLE_RATE) + n] for s in keep]))
    embs = []
    with torch.no_grad():
        for i in range(0, len(batch), 64):
            embs.append(model.encode_batch(batch[i : i + 64]).squeeze(1).numpy())
    e = np.concatenate(embs)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
    return np.array(keep) + WIN / 2, e


def _silhouette(emb: np.ndarray, labels: np.ndarray) -> float:
    D = 1 - emb @ emb.T
    scores = []
    for i in range(len(emb)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == k].mean() for k in set(labels) if k != labels[i])
        scores.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(scores)) if scores else -1.0


def cluster_windows(emb: np.ndarray, max_k: int = MAX_VOICES, min_sil: float = MIN_SILHOUETTE,
                    min_windows: int = MIN_CLUSTER_WINDOWS) -> np.ndarray:
    """k-means on unit embeddings; k chosen by silhouette. Returns all zeros if no clear voice structure."""
    from scipy.cluster.vq import kmeans2

    n = len(emb)
    if n < 2 * min_windows:
        return np.zeros(n, int)
    best_sil, best = min_sil, np.zeros(n, int)
    for k in range(2, min(max_k, n // min_windows) + 1):
        for seed in range(5):
            np.random.seed(seed)  # kmeans2 draws from the global RNG; keep runs reproducible
            _, lab = kmeans2(emb, k, minit="++", seed=seed)
            if len(set(lab)) < k or np.bincount(lab).min() < min_windows:
                continue
            sil = _silhouette(emb, lab)
            if sil > best_sil:
                best_sil, best = sil, lab.copy()
    return best


def label_words(words: list[Word], centres: np.ndarray, labels: np.ndarray) -> None:
    """Each word takes the voice of the windows covering its midpoint (weighted by closeness)."""
    if len(centres) == 0:
        return
    for w in words:
        mid = (w.start + w.end) / 2
        d = np.abs(centres - mid)
        near = np.where(d <= WIN / 2)[0]
        if len(near) == 0:
            w.voice = int(labels[np.argmin(d)])
            continue
        weight = {}
        for j in near:
            weight[labels[j]] = weight.get(labels[j], 0.0) + (WIN / 2 - d[j] + 1e-3)
        w.voice = int(max(weight, key=weight.get))


def _word_scores(words: list[Word], tracks: list[Track], smooth) -> dict[tuple[int, int], float]:
    out = {}
    for i, w in enumerate(words):
        f0, f1 = int(w.start * ASD_FPS), max(int(w.end * ASD_FPS), int(w.start * ASD_FPS) + 1)
        for t in tracks:
            lo, hi = max(f0, t.start), min(f1, t.end + 1)
            if hi - lo >= 0.5 * (f1 - f0):
                out[(i, t.id)] = float(smooth[t.id][lo - t.start : hi - t.start].mean())
    return out


def match_voices_to_tracks(words: list[Word], tracks: list[Track], smooth) -> tuple[VoiceMap, dict]:
    voices = sorted({w.voice for w in words if w.voice is not None})
    ws = _word_scores(words, tracks, smooth)
    # E[v, t]: mean ASD logit of track t over the words of voice v
    E, N = {}, {}
    for v in voices:
        for t in tracks:
            vals = [ws[(i, t.id)] for i, w in enumerate(words) if w.voice == v and (i, t.id) in ws]
            if len(vals) >= MIN_WORDS:
                E[(v, t.id)], N[(v, t.id)] = float(np.mean(vals)), len(vals)
    contrast = {}
    for (v, t), e in E.items():
        others = [(E[(u, t)], N[(u, t)]) for u in voices if u != v and (u, t) in E]
        if others:
            ref = sum(x * n for x, n in others) / sum(n for _, n in others)
            contrast[(v, t)] = e - ref
    tracks_of: dict[int, list[int]] = {v: [] for v in voices}
    if contrast:
        from scipy.optimize import linear_sum_assignment

        tids = [t.id for t in tracks]
        M = np.array([[contrast.get((v, t), -1e3) for t in tids] for v in voices])
        for r, c in zip(*linear_sum_assignment(-M)):
            if M[r, c] > MIN_CONTRAST:
                tracks_of[voices[r]].append(tids[c])
        # faces that light up for every voice give no contrast. Once the discriminating faces are matched,
        # leftover voices take leftover faces by absolute evidence (speaking logit above zero).
        left_v = [v for v in voices if not tracks_of[v]]
        left_t = [t for t in tids if all(t not in ts for ts in tracks_of.values())]
        if left_v and left_t:
            A = np.array([[E.get((v, t), -1e3) for t in left_t] for v in left_v])
            for r, c in zip(*linear_sum_assignment(-A)):
                if A[r, c] > SPEAK_THRESH:
                    tracks_of[left_v[r]].append(left_t[c])
        # extra tracks (same person re-entering) may join a voice if they never overlap its existing faces
        by_id = {t.id: t for t in tracks}
        taken = {t for ts in tracks_of.values() for t in ts}
        for t in tracks:
            if t.id in taken:
                continue
            best = max(voices, key=lambda v: contrast.get((v, t.id), -1e3))
            if contrast.get((best, t.id), -1e3) > MIN_CONTRAST and all(
                t.end < by_id[o].start or by_id[o].end < t.start for o in tracks_of[best]
            ):
                tracks_of[best].append(t.id)
    return VoiceMap(tracks_of, contrast), ws
