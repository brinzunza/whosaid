"""Lip-to-voice sync scorer: a small audio-visual network trained from scratch to tell whether a mouth and a sound belong together.

Why a separate model: the pretrained detector (LR-ASD) mostly checks "lips move and speech is audible", so a listener who
mouths along while somebody else talks fools it. This network sees the same inputs (the 112x112 lower-face crop and the
scene's MFCCs) but is trained with the confusions as explicit negatives: audio shifted by a fraction of a second, audio
from elsewhere in the scene, and listeners whose lips move against another person's voice. It is tiny (~0.3 M weights),
so a few thousand labelled frames from a handful of people are enough to train it on a CPU in minutes.

Architecture (fully convolutional in time, so any track length works):
  video  (B,1,T,56,56) -> 3 x [Conv3d, BN, ReLU] with spatial pooling      -> (B, D, T)
  audio  (B,13,4T)     -> Conv1d stack, last layer stride 4 (100 Hz -> 25 Hz) -> (B, D, T)
  joint  [v*a, |v-a|, v, a] -> Conv1d (k=5) -> Conv1d (k=1)                -> one logit per video frame
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .asd_train import CAT, Scene, _auc, load_scenes
from .disk import ensure_free

D = 64
SIDE = 56
T_WIN = 50
CTX_PAD = 25      # frames of extra context scored on either side of a chunk at inference


LAGS = 4          # lag head: similarity of video frame t with audio frames t-LAGS .. t+LAGS


def _make_net(arch: str = "lag"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            def c3(i, o, k, s=(1, 1, 1)):
                return nn.Sequential(nn.Conv3d(i, o, k, s, padding=tuple(x // 2 for x in k), bias=False), nn.BatchNorm3d(o), nn.ReLU(True))

            self.v = nn.Sequential(c3(1, 16, (5, 5, 5), (1, 2, 2)), nn.MaxPool3d((1, 2, 2)),
                                   c3(16, 32, (3, 3, 3)), nn.MaxPool3d((1, 2, 2)),
                                   c3(32, D, (3, 3, 3)), nn.AdaptiveAvgPool3d((None, 1, 1)))
            self.v1 = nn.Conv1d(D, D, 3, padding=1)
            self.a_in = nn.BatchNorm1d(13, affine=False)
            self.a = nn.Sequential(nn.Conv1d(13, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(True),
                                   nn.Conv1d(32, D, 5, padding=2), nn.BatchNorm1d(D), nn.ReLU(True),
                                   nn.Conv1d(D, D, 4, stride=4))
            self.a1 = nn.Conv1d(D, D, 3, padding=1)
            self.arch = arch
            if arch == "lag":      # sees only the similarity profile across lags: sync evidence, not 'speech is present'
                self.scale = nn.Parameter(torch.tensor(5.0))
                self.joint = nn.Sequential(nn.Conv1d(2 * LAGS + 1, 32, 5, padding=2), nn.ReLU(True), nn.Dropout(0.2), nn.Conv1d(32, 1, 1))
            else:                  # "concat": the first version, sees the embeddings themselves
                self.joint = nn.Sequential(nn.Conv1d(4 * D, 64, 5, padding=2), nn.ReLU(True), nn.Dropout(0.2), nn.Conv1d(64, 1, 1))

        def embed_v(self, x):                                   # x (B,T,56,56) float 0-255
            x = ((x / 255 - 0.42) / 0.17).unsqueeze(1)
            return self.v1(self.v(x)[:, :, :, 0, 0])

        def embed_a(self, m):                                   # m (B,4T,13)
            m = m.transpose(1, 2)
            m = m - m.mean(2, keepdim=True)
            return self.a1(self.a(self.a_in(m)))

        def embeddings(self, x, m):
            v, a = self.embed_v(x), self.embed_a(m)
            n = min(v.shape[2], a.shape[2])
            return F.normalize(v[:, :, :n], dim=1), F.normalize(a[:, :, :n], dim=1)

        def forward(self, x, m):
            v, a = self.embeddings(x, m)
            if self.arch == "lag":
                ap = F.pad(a, (LAGS, LAGS))
                n = v.shape[2]
                sims = torch.stack([(v * ap[:, :, k: k + n]).sum(1) for k in range(2 * LAGS + 1)], 1)
                return self.joint(self.scale * sims).squeeze(1)
            return self.joint(torch.cat([v * a, (v - a).abs(), v, a], 1)).squeeze(1)

    return Net()


def new_model(arch: str = "lag"):
    return _make_net(arch)


def load_sync(path):
    import torch

    ck = torch.load(path, map_location="cpu")
    m = new_model(ck["arch"])
    m.load_state_dict(ck["state"])
    return m.eval()


def info_nce(v, a, y, tau: float = 0.1, guard: int = 2):
    """Dense contrastive loss. For every frame of an aligned (y=1) window, the audio frame at the same time must be more
    similar to the video frame than any other audio frame in the batch except its own neighbours (+-guard frames)."""
    import torch
    import torch.nn.functional as F

    B, _, T = v.shape
    n = min(T, y.shape[1])
    V = v[:, :, :n].permute(0, 2, 1).reshape(B * n, -1)
    A = a[:, :, :n].permute(0, 2, 1).reshape(B * n, -1)
    logits = V @ A.T / tau
    idx = torch.arange(B * n)
    win, t = idx // n, idx % n
    near = (win[:, None] == win[None, :]) & ((t[:, None] - t[None, :]).abs() <= guard)
    near &= ~torch.eye(B * n, dtype=torch.bool)
    logits = logits.masked_fill(near, -1e4)
    keep = y[:, :n].reshape(-1) > 0.5
    if not keep.any():
        return logits.sum() * 0
    return F.cross_entropy(logits[keep], idx[keep])


def downscale(crops: np.ndarray) -> np.ndarray:
    """(n,112,112) uint8 -> (n,56,56) uint8 by 2x2 averaging."""
    n = len(crops)
    return crops.reshape(n, SIDE, 2, SIDE, 2).mean((2, 4)).astype(np.uint8)


def score_track(model, mfcc: np.ndarray, crops56: np.ndarray, start_frame: int, chunk: int = 150) -> np.ndarray:
    """Per-frame sync logit of one track (crops56 already downscaled). Chunks carry CTX_PAD frames of context each side."""
    import torch

    n = len(crops56)
    out = np.full(n, -5.0, np.float32)
    with torch.no_grad():
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            lo, hi = max(0, i - CTX_PAD), min(n, j + CTX_PAD)
            a = mfcc[(start_frame + lo) * 4: (start_frame + hi) * 4]
            k = min(len(a) // 4, hi - lo)
            if k < 8:
                continue
            x = torch.from_numpy(crops56[lo:lo + k].astype(np.float32))[None]
            m = torch.from_numpy(a[: k * 4].astype(np.float32))[None]
            s = model(x, m)[0].numpy()
            seg = s[i - lo: i - lo + (j - i)]
            out[i: i + len(seg)] = seg
    return out


# ---------------------------------------------------------------------------------- data

class SyncScene:
    def __init__(self, s: Scene):
        self.mfcc = s.mfcc
        self.tracks = {tid: dict(crops=downscale(tr["crops"]), start=tr["start"], y=tr["y"], cat=tr["cat"]) for tid, tr in s.tracks.items()}
        self.energy = s.mfcc[:, 0]
        self.loud = float(np.percentile(self.energy, 40))


def sample(scenes: list[SyncScene], rng, p_shift: float = 0.35, p_other: float = 0.10, p_hard: float = 0.35):
    """One window: (mfcc [4T,13], crops [T,56,56], label [T], weight [T]).

    Negatives made on the fly: the audio shifted by 0.12-10 s inside the same scene (lips against the wrong sound), or
    audio taken from a different scene altogether. Windows with many 'mouthing' frames are oversampled."""
    index = [(s, tid) for s in scenes for tid in s.tracks]
    weights = np.array([len(s.tracks[tid]["y"]) for s, tid in index], float)
    for _ in range(20):
        s, tid = index[rng.choice(len(index), p=weights / weights.sum())]
        tr = s.tracks[tid]
        n = len(tr["y"])
        if n < T_WIN:
            continue
        if rng.random() < p_hard and (tr["cat"] == 1).any():
            c = np.where(tr["cat"] == 1)[0]
            i = int(np.clip(rng.choice(c) - rng.integers(0, T_WIN), 0, n - T_WIN))
        else:
            i = int(rng.integers(0, n - T_WIN + 1))
        y = tr["y"][i: i + T_WIN].astype(np.float32)
        cat = tr["cat"][i: i + T_WIN]
        w = np.where(cat == 1, 3.0, 1.0).astype(np.float32)
        a0, src = (tr["start"] + i) * 4, s
        r = rng.random()
        if r < p_shift:
            off = int(rng.choice([-1, 1]) * rng.uniform(0.12, 10.0) * 100)
            b0 = a0 + off
            if 0 <= b0 and b0 + T_WIN * 4 <= len(s.mfcc) and s.energy[b0: b0 + T_WIN * 4].mean() > s.loud:
                a0, y, w = b0, np.zeros(T_WIN, np.float32), np.where(tr["y"][i: i + T_WIN] == 1, 2.0, 1.0).astype(np.float32)
        elif r < p_shift + p_other:
            o = scenes[int(rng.integers(len(scenes)))]
            if o is not s:
                b0 = int(rng.integers(0, max(1, len(o.mfcc) - T_WIN * 4)))
                if o.energy[b0: b0 + T_WIN * 4].mean() > o.loud:
                    a0, src, y, w = b0, o, np.zeros(T_WIN, np.float32), np.where(tr["y"][i: i + T_WIN] == 1, 2.0, 1.0).astype(np.float32)
        aud = src.mfcc[a0: a0 + T_WIN * 4]
        if len(aud) < T_WIN * 4:
            continue
        vid = tr["crops"][i: i + T_WIN].astype(np.float32)
        if rng.random() < 0.5:
            vid = vid[:, :, ::-1]
        vid = np.clip(vid * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15), 0, 255)
        dy, dx = (int(rng.integers(-2, 3)) for _ in range(2))
        vid = np.roll(vid, (dy, dx), (1, 2))
        aud = aud + rng.normal(0, 0.3, aud.shape).astype(np.float32)
        return aud, vid.copy(), y, w
    return None


# ---------------------------------------------------------------------------------- evaluation

def scene_scores(model, scenes: list[SyncScene]) -> list[dict]:
    return [{tid: score_track(model, s.mfcc, tr["crops"], tr["start"]) for tid, tr in s.tracks.items()} for s in scenes]


def evaluate_scores(scenes, scores: list[dict]) -> dict:
    """Same breakdown as asd_train.evaluate: AUC overall and against each negative kind."""
    by_cat = {c: [] for c in CAT}
    for s, sc in zip(scenes, scores):
        for tid, tr in s.tracks.items():
            x = sc[tid]
            for c in CAT:
                by_cat[c].append(x[: len(tr["cat"])][tr["cat"][: len(x)] == c])
    cat = {c: np.concatenate(v) if v else np.array([]) for c, v in by_cat.items()}
    neg = np.concatenate([cat[1], cat[2], cat[3], cat[4]])
    return dict(auc=_auc(cat[0], neg), auc_mouthing=_auc(cat[0], cat[1]), auc_freeze=_auc(cat[0], cat[2]),
                auc_silent=_auc(cat[0], cat[3]), auc_false_face=_auc(cat[0], cat[4]),
                frames={CAT[c]: int(len(v)) for c, v in cat.items()})


def fmt(name: str, r: dict) -> str:
    return (f"{name:<14} AUC {r['auc']:.3f} | vs mouthing {r['auc_mouthing']:.3f}  freeze {r['auc_freeze']:.3f}  "
            f"silent {r['auc_silent']:.3f}  false-face {r['auc_false_face']:.3f}")


def asd_scores(scenes_raw: list[Scene], model_path=None) -> list[dict]:
    """The pipeline's own (multi-duration) LR-ASD scores for the same tracks, for comparison and fusion."""
    from .asd import score_track as asd_track
    from .lrasd.model import load_model
    from .paths import ASD_MODEL

    m = load_model(model_path or ASD_MODEL).cpu()
    return [{tid: asd_track(m, s.mfcc, tr["crops"], tr["start"]) for tid, tr in s.tracks.items()} for s in scenes_raw]


# ---------------------------------------------------------------------------------- training

def train(train_scenes, val_scenes, out_path: Path, epochs: int = 12, steps: int = 150, batch: int = 16, lr: float = 2e-3,
          seed: int = 0, log=print, arch: str = "lag", nce: float = 1.0, **sample_kw) -> dict:
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = new_model(arch)
    log(f"sync scorer: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M weights")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    best, hist = -1.0, []
    for ep in range(1, epochs + 1):
        ensure_free(f"sync epoch {ep}")
        model.train()
        tot, t0 = 0.0, time.time()
        for _ in range(steps):
            items = [x for x in (sample(train_scenes, rng, **sample_kw) for _ in range(batch * 2)) if x is not None][:batch]
            m = torch.from_numpy(np.stack([i[0] for i in items]))
            x = torch.from_numpy(np.stack([i[1] for i in items]))
            y = torch.from_numpy(np.stack([i[2] for i in items]))
            w = torch.from_numpy(np.stack([i[3] for i in items]))
            logit = model(x, m)
            n = logit.shape[1]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, y[:, :n], reduction="none") * w[:, :n]).sum() / w[:, :n].sum()
            if nce:
                ev, ea = model.embeddings(x, m)
                loss = loss + nce * info_nce(ev, ea, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        model.eval()
        r = evaluate_scores(val_scenes, scene_scores(model, val_scenes))
        hist.append(dict(epoch=ep, loss=tot / steps, **{k: r[k] for k in ("auc", "auc_mouthing", "auc_freeze", "auc_false_face")}))
        log(f"epoch {ep}: loss {tot / steps:.3f}  " + fmt("val", r) + f"  [{time.time() - t0:.0f}s]")
        score = r["auc"] + r["auc_mouthing"]
        if score > best:
            best = score
            torch.save(dict(arch=arch, state=model.state_dict()), out_path)
    return dict(best=best, history=hist)


def _load(d: Path, prefix: str) -> list[SyncScene]:
    return [SyncScene(s) for s in load_scenes(d, prefix)]


def main(argv):
    import argparse

    p = argparse.ArgumentParser(prog="whosaid sync")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/asd")
    t.add_argument("--out", default="models/sync.pt")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--steps", type=int, default=150)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--arch", default="lag", choices=["lag", "concat"])
    t.add_argument("--nce", type=float, default=1.0, help="weight of the dense contrastive loss (0 = off)")
    e = sub.add_parser("eval")
    e.add_argument("--data", default="data/asd")
    e.add_argument("--prefix", default="test_")
    e.add_argument("--model", default="models/sync.pt")
    a = p.parse_args(argv)
    d = Path(a.data)
    if a.cmd == "train":
        tr, va = _load(d, "train_"), _load(d, "val_")
        print(f"{len(tr)} train scenes, {len(va)} val scenes")
        res = train(tr, va, Path(a.out), a.epochs, a.steps, lr=a.lr, seed=a.seed, arch=a.arch, nce=a.nce)
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1))
    else:
        raw = load_scenes(d, a.prefix)
        sc = [SyncScene(s) for s in raw]
        print(fmt("sync", evaluate_scores(sc, scene_scores(load_sync(a.model), sc))))
        print(fmt("lr-asd", evaluate_scores(sc, asd_scores(raw))))
