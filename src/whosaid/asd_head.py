"""Head-only fine-tuning of LR-ASD: freeze both encoders, cache their embeddings, train fusion + GRU + classifier.

Why: full fine-tuning on a handful of identities memorised faces instead of learning lip/voice synchrony. Freezing the
encoders leaves ~0.1M trainable weights and lets us train in seconds, with synchrony negatives made at the embedding
level: audio embeddings from another moment of the same scene paired with this face's video embeddings.
Embeddings are tiny (128 floats per frame), so the cache is a few MB.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .asd_train import CAT, T_WIN, Scene, _auc, load_scenes
from .disk import ensure_free


def embed_scene(model, s: Scene, device: str):
    """Frozen-encoder embeddings: audio for the whole scene (one vector per video frame), video per track."""
    import torch

    def chunks(n):
        return [(i, min(i + T_WIN, n)) for i in range(0, n, T_WIN)]

    nframes = len(s.mfcc) // 4
    A = np.zeros((nframes, 128), np.float32)
    with torch.no_grad():
        for i, j in chunks(nframes):
            a = torch.from_numpy(s.mfcc[i * 4: j * 4])[None].to(device)
            if a.shape[1] < 16:
                continue
            e = model.model.forward_audio_frontend(a)[0].cpu().numpy()
            A[i: i + len(e)] = e
        V = {}
        for tid, tr in s.tracks.items():
            n = len(tr["y"])
            out = np.zeros((n, 128), np.float32)
            for i, j in chunks(n):
                if j - i < 4:
                    continue
                v = torch.from_numpy(tr["crops"][i:j].astype(np.float32))[None].to(device)
                out[i:j] = model.model.forward_visual_frontend(v)[0].cpu().numpy()
            V[tid] = out
    return A, V


def build_embeddings(data_dir: Path, out_dir: Path, device: str | None = None, log=print) -> None:
    import torch

    from .lrasd.model import load_model
    from .paths import ASD_MODEL

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(ASD_MODEL).to(device).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(data_dir).glob("*.npz")):
        out = out_dir / p.name
        if out.exists():
            continue
        ensure_free(f"embedding {p.stem}")
        t0 = time.time()
        s = Scene(p)
        A, V = embed_scene(model, s, device)
        arrays = {"A": A.astype(np.float16), "meta": np.array(json.dumps(s.meta))}
        for tid, tr in s.tracks.items():
            arrays[f"V_{tid}"] = V[tid].astype(np.float16)
            arrays[f"start_{tid}"] = np.array(tr["start"])
            arrays[f"y_{tid}"] = tr["y"]
            arrays[f"cat_{tid}"] = tr["cat"]
        np.savez_compressed(out, **arrays)
        log(f"{p.stem}: {time.time() - t0:.0f}s")


class HScene:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.A = z["A"].astype(np.float32)
        self.tracks = {tid: dict(V=z[f"V_{tid}"].astype(np.float32), start=int(z[f"start_{tid}"]), y=z[f"y_{tid}"], cat=z[f"cat_{tid}"])
                       for tid in self.meta["tracks"]}
        e = np.linalg.norm(self.A, axis=1)
        self.speech_like = e > np.percentile(e, 40)              # frames with audio activity


def load_hscenes(d: Path, prefix: str | None = None) -> list[HScene]:
    return [HScene(p) for p in sorted(Path(d).glob("*.npz")) if prefix is None or p.name.startswith(prefix)]


class Head:
    """Fusion + GRU detector + classifier, initialised from the pretrained model."""

    def __init__(self, base, device: str):
        import copy

        self.fusion = copy.deepcopy(base.model.fusion).to(device)
        self.detector = copy.deepcopy(base.model.detector).to(device)
        self.fc = copy.deepcopy(base.lossAV.FC).to(device)
        self.device = device

    def modules(self):
        return [self.fusion, self.detector, self.fc]

    def parameters(self):
        return [p for m in self.modules() for p in m.parameters()]

    def __call__(self, a, v):                       # a, v: [B, T, 128]
        import torch

        x = self.fusion(a, v)
        x = self.detector(x)
        return self.fc(torch.reshape(x, (-1, 128)))[:, 1]

    def train(self):
        [m.train() for m in self.modules()]

    def eval(self):
        [m.eval() for m in self.modules()]

    def state(self) -> dict:
        return {f"{n}.{k}": v.cpu() for n, m in (("fusion", self.fusion), ("detector", self.detector), ("fc", self.fc)) for k, v in m.state_dict().items()}


def sample(scenes: list[HScene], rng, hard_p: float, shift_p: float):
    weights = np.array([len(t["y"]) for s in scenes for t in s.tracks.values()], float)
    index = [(s, tid) for s in scenes for tid in s.tracks]
    s, tid = index[rng.choice(len(index), p=weights / weights.sum())]
    tr = s.tracks[tid]
    n = len(tr["y"])
    if n < T_WIN:
        return None
    hs = tr.get("hard")
    if hs is None:
        m = np.r_[0, np.cumsum((tr["cat"] == 1).astype(np.float32))]
        tr["hard"] = hs = np.where((m[T_WIN:] - m[:-T_WIN]) / T_WIN >= 0.3)[0]
    i = int(rng.choice(hs)) if len(hs) and rng.random() < hard_p else int(rng.integers(0, n - T_WIN + 1))
    cat = tr["cat"][i: i + T_WIN]
    y = tr["y"][i: i + T_WIN].astype(np.float32)
    w = np.where(cat == 1, 3.0, 1.0).astype(np.float32)
    a0 = tr["start"] + i
    if rng.random() < shift_p:
        for _ in range(10):
            b0 = a0 + int(rng.choice([-1, 1]) * rng.uniform(1.5, 10.0) * 25)
            if 0 <= b0 and b0 + T_WIN <= len(s.A) and s.speech_like[b0: b0 + T_WIN].mean() > 0.5:
                a0, y, w = b0, np.zeros(T_WIN, np.float32), np.where(tr["y"][i: i + T_WIN] == 1, 2.0, 1.0).astype(np.float32)
                break
    if a0 + T_WIN > len(s.A):
        return None
    return s.A[a0: a0 + T_WIN], tr["V"][i: i + T_WIN], y, w


def head_scores(head: Head, s: HScene, tid: int) -> np.ndarray:
    import torch

    tr = s.tracks[tid]
    n = len(tr["y"])
    out = []
    head.eval()
    with torch.no_grad():
        for i in range(0, n, T_WIN):
            j = min(i + T_WIN, n)
            a0 = tr["start"] + i
            if j - i < 4 or a0 + (j - i) > len(s.A):
                out.append(np.full(j - i, -5.0))
                continue
            a = torch.from_numpy(s.A[a0: a0 + j - i])[None].to(head.device)
            v = torch.from_numpy(tr["V"][i:j])[None].to(head.device)
            out.append(head(a, v).cpu().numpy())
    return np.concatenate(out)


def evaluate_head(head: Head, scenes: list[HScene], thresh: float = 1.0) -> dict:
    by = {c: [] for c in CAT}
    for s in scenes:
        for tid, tr in s.tracks.items():
            sc = head_scores(head, s, tid)
            for c in CAT:
                by[c].append(sc[tr["cat"][: len(sc)] == c])
    cat = {c: np.concatenate(v) if v else np.array([]) for c, v in by.items()}
    pos = cat[0]
    neg = np.concatenate([cat[1], cat[2], cat[3], cat[4]])
    r = dict(auc=_auc(pos, neg), auc_mouthing=_auc(pos, cat[1]), auc_freeze=_auc(pos, cat[2]), auc_silent=_auc(pos, cat[3]),
             auc_false_face=_auc(pos, cat[4]), tpr=float((pos > thresh).mean()))
    for c in (1, 2, 3, 4):
        r[f"fpr_{CAT[c]}"] = float((cat[c] > thresh).mean()) if len(cat[c]) else float("nan")
    return r


def train_head(emb_dir: Path, out_path: Path, epochs: int = 30, steps: int = 60, batch: int = 32, lr: float = 3e-4,
               hard_p: float = 0.5, shift_p: float = 0.4, seed: int = 0, log=print) -> dict:
    import torch

    from .lrasd.model import load_model
    from .paths import ASD_MODEL

    device = "cpu"          # tiny model, cached embeddings: CPU is plenty and avoids GPU contention
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tr_s, va_s = load_hscenes(emb_dir, "train_"), load_hscenes(emb_dir, "val_")
    base = load_model(ASD_MODEL)
    head = Head(base, device)
    n_par = sum(p.numel() for p in head.parameters())
    log(f"head-only: {n_par / 1e6:.2f}M trainable weights; train scenes {len(tr_s)}, val scenes {len(va_s)}")
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    ev0 = evaluate_head(head, va_s)
    log(f"epoch 0 (pretrained head): val AUC {ev0['auc']:.3f} (mouthing {ev0['auc_mouthing']:.3f}, freeze {ev0['auc_freeze']:.3f}, silent {ev0['auc_silent']:.3f})")
    best, hist = ev0["auc"], []
    torch.save(head.state(), out_path)
    for ep in range(1, epochs + 1):
        head.train()
        tot = 0.0
        for _ in range(steps):
            items = [x for x in (sample(tr_s, rng, hard_p, shift_p) for _ in range(batch * 2)) if x is not None][:batch]
            a = torch.from_numpy(np.stack([i[0] for i in items])); v = torch.from_numpy(np.stack([i[1] for i in items]))
            y = torch.from_numpy(np.stack([i[2] for i in items])).reshape(-1); w = torch.from_numpy(np.stack([i[3] for i in items])).reshape(-1)
            x1 = head(a, v)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(x1, y, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0); opt.step()
            tot += float(loss.detach())
        ev = evaluate_head(head, va_s)
        hist.append(dict(epoch=ep, loss=tot / steps, **{k: ev[k] for k in ("auc", "auc_mouthing", "auc_freeze", "auc_silent")}))
        if ep % 3 == 0 or ep == 1:
            log(f"epoch {ep}: loss {tot / steps:.3f}  val AUC {ev['auc']:.3f} (mouthing {ev['auc_mouthing']:.3f}, freeze {ev['auc_freeze']:.3f}, silent {ev['auc_silent']:.3f})")
        if ev["auc"] > best:
            best = ev["auc"]
            torch.save(head.state(), out_path)
    log(f"best val AUC {best:.3f} -> {out_path}")
    return dict(best_val_auc=best, history=hist)


def head_to_model_state(head_state: dict, base_path) -> dict:
    """Pretrained full model state with the head replaced -> loadable with lrasd.model.load_model."""
    import torch

    sd = torch.load(base_path, map_location="cpu")
    for k, v in head_state.items():
        n, rest = k.split(".", 1)
        sd[{"fusion": "model.fusion.", "detector": "model.detector.", "fc": "lossAV.FC."}[n] + rest] = v
    return sd


def main(argv):
    import argparse

    import torch

    p = argparse.ArgumentParser(prog="whosaid asd-head")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("embed"); b.add_argument("--data", default="data/asd"); b.add_argument("--out", default="data/asd_emb")
    t = sub.add_parser("train"); t.add_argument("--emb", default="data/asd_emb"); t.add_argument("--out", default="models/lrasd_head.pt")
    t.add_argument("--epochs", type=int, default=30); t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--shift-p", type=float, default=0.4); t.add_argument("--hard-p", type=float, default=0.5)
    t.add_argument("--export", default="models/lrasd_head.model", help="write a full model file usable as --asd-model")
    a = p.parse_args(argv)
    if a.cmd == "embed":
        build_embeddings(Path(a.data), Path(a.out))
    else:
        from .paths import ASD_MODEL

        res = train_head(Path(a.emb), Path(a.out), epochs=a.epochs, lr=a.lr, shift_p=a.shift_p, hard_p=a.hard_p)
        torch.save(head_to_model_state(torch.load(a.out), ASD_MODEL), a.export)
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1))


def calibrate_offset(head: Head, base_head: Head, scenes: list[HScene], thresh: float = 1.0) -> float:
    """Bias shift for the new head so that, on `scenes`, the share of speaking frames scoring above `thresh` matches the
    pretrained head's. The pipeline's thresholds (raw > 1 ...) were tuned for the pretrained score scale."""
    def pos_scores(h):
        out = []
        for s in scenes:
            for tid, tr in s.tracks.items():
                sc = head_scores(h, s, tid)
                out.append(sc[tr["cat"][: len(sc)] == 0])
        return np.concatenate(out)

    target = float((pos_scores(base_head) > thresh).mean())
    new = pos_scores(head)
    # the (1 - target) quantile of the new speaking scores must land on `thresh`
    return float(thresh - np.quantile(new, 1.0 - target))
