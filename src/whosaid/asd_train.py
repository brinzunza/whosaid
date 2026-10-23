"""Fine-tune the speaking detector (LR-ASD) on synthetic scenes with exact labels.

Input to the network is what it sees at inference: lower-face crops of one tracked face + the *mixed* scene audio.
The label per frame is whether that person is the one speaking. The scenes deliberately contain hard cases:
listeners whose lips keep moving while somebody else's voice plays (`mouthing`), static faces, a wall portrait that
is detected as a face, off-screen voices.

Frame categories (stored per track):
  0 speaking        1 mouthing   (listener's lips move, audio is someone else)
  2 freeze          (listener frozen, someone else speaks)      3 silent  (nobody speaks)
  4 false_face      (a small secondary face in the tile: never speaking)
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from .disk import ensure_free

CAT = {0: "speaking", 1: "mouthing", 2: "freeze", 3: "silent", 4: "false_face"}


# ---------------------------------------------------------------------------------- dataset building

def person_flags(labels: dict, tracks) -> dict[int, bool]:
    """Ground truth per track: is it the person of its tile (True) or a small secondary face such as a wall portrait or a
    background photo (False)? A secondary face is one under half the size of the biggest face in its tile."""
    from .bench import track_tile

    tiles = labels["tiles"]
    sizes = {t.id: float(np.median(t.s)) for t in tracks}
    tile_of = {t.id: track_tile(t, t.start + len(t.frames) // 2, tiles) for t in tracks}
    biggest: dict = {}
    for t in tracks:
        biggest[tile_of[t.id]] = max(biggest.get(tile_of[t.id], 0.0), sizes[t.id])
    return {t.id: sizes[t.id] >= 0.5 * biggest[tile_of[t.id]] for t in tracks}


def frame_labels(labels: dict, tracks) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per tracked face and 25-fps frame: (is this person speaking, category) from a scene's ground truth.
    Categories: see the module docstring. A small secondary face inside a tile is never speaking (false face)."""
    from .bench import track_tile

    n_frames = int(labels["duration"] * 25) + 2
    speak = {tid: np.zeros(n_frames, bool) for tid in labels["speech"]}
    for tid, spans in labels["speech"].items():
        for a, b in spans:
            speak[tid][int(a * 25): int(np.ceil(b * 25)) + 1] = True
    anyone = np.any(list(speak.values()), axis=0)
    tiles = labels["tiles"]
    sizes = {t.id: float(np.median(t.s)) for t in tracks}
    tile_of = {t.id: track_tile(t, t.start + len(t.frames) // 2, tiles) for t in tracks}
    biggest: dict = {}
    for t in tracks:
        biggest[tile_of[t.id]] = max(biggest.get(tile_of[t.id], 0.0), sizes[t.id])
    out = {}
    for t in tracks:
        tile = tile_of[t.id]
        person = sizes[t.id] >= 0.5 * biggest[tile]
        f = np.minimum(np.arange(t.start, t.end + 1), n_frames - 1)
        own = speak[tile][f] if tile in speak else np.zeros(len(f), bool)
        other = np.zeros(len(f), bool)
        for tid, sp in speak.items():
            if tid != tile:
                other |= sp[f]
        y = own & person
        cat = np.full(len(f), 3, np.uint8)                       # silent
        cat[~own & other & (labels["listener"] == "freeze")] = 2
        cat[~own & other & (labels["listener"] == "mouthing")] = 1
        cat[y] = 0
        cat[~person & anyone[f]] = 4
        out[t.id] = (y, cat)
    return out


def build_scene(video: Path, labels_path: Path, out_path: Path, log=print) -> bool:
    from .asd import collect_crops, mfcc_features
    from .faces import track_faces
    from .media import extract_audio, probe

    labels = json.loads(Path(labels_path).read_text())
    info = probe(video)
    tracks = track_faces(video, info)
    if not tracks:
        return False
    crops = collect_crops(video, info, tracks, lips=False)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        extract_audio(video, wav)
        mfcc = mfcc_features(wav).astype(np.float16)
    fl = frame_labels(labels, tracks)
    arrays = {"mfcc": mfcc, "meta": np.array(json.dumps(dict(
        scene=labels["scene"], kind=labels["kind"], listener=labels["listener"], split=labels["split"],
        tracks=[t.id for t in tracks])))}
    for t in tracks:
        y, cat = fl[t.id]
        arrays[f"crops_{t.id}"] = crops[t.id]
        arrays[f"start_{t.id}"] = np.array(t.start)
        arrays[f"y_{t.id}"] = y.astype(np.uint8)
        arrays[f"cat_{t.id}"] = cat
    np.savez_compressed(out_path, **arrays)
    return True


def build_all(scenes_dir: Path, out_dir: Path, splits: list[str], log=print) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in splits:
        for lp in sorted((scenes_dir / sp).glob("*.labels.json")):
            name = lp.name.replace(".labels.json", "")
            out = out_dir / f"{name}.npz"
            if out.exists():
                continue
            ensure_free(f"building {name}")
            t0 = time.time()
            ok = build_scene(lp.with_name(name + ".mp4"), lp, out)
            log(f"{name}: {'ok' if ok else 'no faces'} ({time.time() - t0:.0f}s)")


class Scene:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.mfcc = z["mfcc"].astype(np.float32)
        self.tracks = {}
        for tid in self.meta["tracks"]:
            self.tracks[tid] = dict(crops=z[f"crops_{tid}"], start=int(z[f"start_{tid}"]), y=z[f"y_{tid}"], cat=z[f"cat_{tid}"])


def load_scenes(d: Path, prefix: str | None = None) -> list[Scene]:
    return [Scene(p) for p in sorted(Path(d).glob("*.npz")) if prefix is None or p.name.startswith(prefix)]


# ---------------------------------------------------------------------------------- training

T_WIN = 75


def _prep_hard(scenes: list[Scene]) -> None:
    """For each track, the window starts whose frames are mostly 'mouthing' (listener lips moving, someone else talking)."""
    for s in scenes:
        for tr in s.tracks.values():
            m = (tr["cat"] == 1).astype(np.float32)
            c = np.r_[0, np.cumsum(m)]
            n = len(m)
            frac = (c[T_WIN:] - c[:-T_WIN]) / T_WIN if n >= T_WIN else np.array([])
            tr["hard_starts"] = np.where(frac >= 0.3)[0]


def _speech_energy(mfcc: np.ndarray) -> np.ndarray:
    return mfcc[:, 0]        # c0 ~ log energy


def _sample(scenes: list[Scene], rng, train: bool, hard_p: float = 0.0, shift_p: float = 0.0):
    """One training window: (audio mfcc [4T,13], face crops [T,112,112], label [T], loss weight [T]).

    hard_p: chance of choosing a window that is mostly 'mouthing' (when the track has any).
    shift_p: chance of replacing the audio by speech from another moment of the same scene; the lips then move against
    the wrong sound, so every frame is labelled 'not speaking' (and weighted up where the lips were really talking)."""
    weights = np.array([len(tr["y"]) for s in scenes for tr in s.tracks.values()], float)
    index = [(s, tid) for s in scenes for tid in s.tracks]
    s, tid = index[rng.choice(len(index), p=weights / weights.sum())]
    tr = s.tracks[tid]
    n = len(tr["y"])
    if n < T_WIN:
        return None
    hs = tr.get("hard_starts")
    if hs is not None and len(hs) and rng.random() < hard_p:
        i = int(rng.choice(hs))
    else:
        i = int(rng.integers(0, n - T_WIN + 1))
    y = tr["y"][i: i + T_WIN].astype(np.float32)
    cat = tr["cat"][i: i + T_WIN]
    w = np.where(cat == 1, 3.0, 1.0).astype(np.float32)
    a0 = (tr["start"] + i) * 4
    if shift_p and rng.random() < shift_p:
        for _ in range(10):
            off = int(rng.choice([-1, 1]) * rng.uniform(1.5, 10.0) * 100)
            b0 = a0 + off
            if 0 <= b0 and b0 + T_WIN * 4 <= len(s.mfcc) and _speech_energy(s.mfcc[b0: b0 + T_WIN * 4]).mean() > np.percentile(_speech_energy(s.mfcc), 40):
                a0, y, w = b0, np.zeros(T_WIN, np.float32), np.where(tr["y"][i: i + T_WIN] == 1, 2.0, 1.0).astype(np.float32)
                break
    aud = s.mfcc[a0: a0 + T_WIN * 4]
    if len(aud) < T_WIN * 4:
        return None
    vid = tr["crops"][i: i + T_WIN].astype(np.float32)
    if train:
        if rng.random() < 0.5:
            vid = vid[:, :, ::-1]
        vid = vid * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15)
        dy, dx = (int(rng.integers(-4, 5)) for _ in range(2))
        vid = np.roll(vid, (dy, dx), (1, 2))
        vid = np.clip(vid, 0, 255)
    return aud, vid.copy(), y, w


def train(train_dir: Path, val_dir: Path, out_path: Path, epochs: int = 10, batch: int = 8, lr: float = 2e-4,
          steps_per_epoch: int | None = None, device: str | None = None, seed: int = 0, log=print,
          freeze_audio: bool = False, hard_p: float = 0.5, shift_p: float = 0.35) -> dict:
    import torch

    from .lrasd.model import load_model
    from .paths import ASD_MODEL

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tr_scenes, va_scenes = load_scenes(train_dir), load_scenes(val_dir)
    _prep_hard(tr_scenes)
    frames = sum(len(t["y"]) for s in tr_scenes for t in s.tracks.values())
    steps = steps_per_epoch or max(frames // (T_WIN * batch), 20)
    log(f"train scenes {len(tr_scenes)} ({frames} face-frames), val scenes {len(va_scenes)}, {steps} steps/epoch on {device}")
    model = load_model(ASD_MODEL).to(device)
    if freeze_audio:
        for p in model.model.audioEncoder.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    base = evaluate(model, va_scenes, device, faithful=False)
    log("epoch 0 (pretrained): val AUC %.3f (vs mouthing %.3f, vs false face %.3f)" % (base["auc"], base["auc_mouthing"], base["auc_false_face"]))
    best, history = base["auc_mouthing"], []
    torch.save(model.state_dict(), out_path)
    for ep in range(1, epochs + 1):
        ensure_free(f"epoch {ep}")
        model.train()
        if freeze_audio:
            model.model.audioEncoder.eval()
        tot, t0 = 0.0, time.time()
        for st in range(steps):
            items = [x for x in (_sample(tr_scenes, rng, True, hard_p, shift_p) for _ in range(batch * 2)) if x is not None][:batch]
            a = torch.from_numpy(np.stack([i[0] for i in items])).to(device)
            v = torch.from_numpy(np.stack([i[1] for i in items])).to(device)
            y = torch.from_numpy(np.stack([i[2] for i in items])).to(device).reshape(-1)
            wt = torch.from_numpy(np.stack([i[3] for i in items])).to(device).reshape(-1)
            ae = model.model.forward_audio_frontend(a)
            ve = model.model.forward_visual_frontend(v)
            x1 = model.lossAV.FC(model.model.forward_audio_visual_backend(ae, ve))[:, 1]   # same score scale as the original
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(x1, y, reduction="none") * wt).sum() / wt.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        va = evaluate(model, va_scenes, device, faithful=False)
        history.append(dict(epoch=ep, train_loss=tot / steps, **{k: va[k] for k in ("auc", "auc_mouthing", "auc_false_face")}))
        log(f"epoch {ep}: loss {tot / steps:.3f}  val AUC {va['auc']:.3f} (vs mouthing {va['auc_mouthing']:.3f}, "
            f"vs false face {va['auc_false_face']:.3f})  [{time.time() - t0:.0f}s]")
        if va["auc_mouthing"] > best:
            best = va["auc_mouthing"]
            torch.save(model.state_dict(), out_path)
    log(f"best val AUC vs mouthing {best:.3f} -> {out_path}")
    return dict(best_val_auc=best, history=history)


# ---------------------------------------------------------------------------------- evaluation

def _fast_scores(model, s: Scene, tid: int, device: str) -> np.ndarray:
    """Single-duration (3 s chunks) scoring, quick enough to use every epoch."""
    import torch

    tr = s.tracks[tid]
    n = len(tr["y"])
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, n, T_WIN):
            j = min(i + T_WIN, n)
            a0 = (tr["start"] + i) * 4
            aud = s.mfcc[a0: a0 + (j - i) * 4]
            if len(aud) < (j - i) * 4 or j - i < 4:
                out.append(np.full(j - i, -5.0))
                continue
            a = torch.from_numpy(aud)[None].to(device)
            v = torch.from_numpy(tr["crops"][i:j].astype(np.float32))[None].to(device)
            out.append(model.score(a, v).cpu().numpy())
    return np.concatenate(out)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def evaluate(model, scenes: list[Scene], device: str = "cpu", faithful: bool = True, thresh: float = 1.0) -> dict:
    """Frame-level AUC overall and against each kind of negative, plus the pipeline's operating point (score > thresh)."""
    from .asd import score_track

    model.eval()
    by_cat = {c: [] for c in CAT}
    for s in scenes:
        for tid, tr in s.tracks.items():
            if faithful:                                   # exactly what the pipeline computes (multi-duration, CPU)
                sc = score_track(model, s.mfcc, tr["crops"], tr["start"])
            else:
                sc = _fast_scores(model, s, tid, device)
            for c in CAT:
                by_cat[c].append(sc[: len(tr["cat"])][tr["cat"][: len(sc)] == c])
    cat = {c: np.concatenate(v) if v else np.array([]) for c, v in by_cat.items()}
    pos = cat[0]
    neg = np.concatenate([cat[1], cat[2], cat[3], cat[4]])
    out = dict(auc=_auc(pos, neg), auc_mouthing=_auc(pos, cat[1]), auc_freeze=_auc(pos, cat[2]), auc_silent=_auc(pos, cat[3]),
               auc_false_face=_auc(pos, cat[4]), tpr=float((pos > thresh).mean()) if len(pos) else float("nan"),
               frames={CAT[c]: int(len(v)) for c, v in cat.items()})
    for c in (1, 2, 3, 4):
        out[f"fpr_{CAT[c]}"] = float((cat[c] > thresh).mean()) if len(cat[c]) else float("nan")
    return out


def format_eval(name: str, r: dict) -> str:
    return (f"{name:<12} AUC {r['auc']:.3f} | vs mouthing {r['auc_mouthing']:.3f}  freeze {r['auc_freeze']:.3f}  silent {r['auc_silent']:.3f}  "
            f"false-face {r['auc_false_face']:.3f} | at score>1: speaking recalled {100 * r['tpr']:.0f}%, wrongly fires on "
            f"mouthing {100 * r['fpr_mouthing']:.0f}%  freeze {100 * r['fpr_freeze']:.0f}%  silent {100 * r['fpr_silent']:.0f}%  "
            f"false-face {100 * r['fpr_false_face']:.0f}%")


def main(argv):
    import argparse

    p = argparse.ArgumentParser(prog="whosaid asd")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="turn synthetic scenes into training arrays")
    b.add_argument("--scenes", default="data/synth")
    b.add_argument("--out", default="data/asd")
    b.add_argument("--splits", default="train,val,test")
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/asd")
    t.add_argument("--out", default="models/lrasd_ft.model")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--freeze-audio", action="store_true")
    t.add_argument("--hard-p", type=float, default=0.5)
    t.add_argument("--shift-p", type=float, default=0.35)
    t.add_argument("--steps", type=int)
    m = sub.add_parser("meta", help="measure a detector's threshold and score gap on validation frames (needed once per detector)")
    m.add_argument("--model", required=True)
    m.add_argument("--data", default="data/asd")
    e = sub.add_parser("eval")
    e.add_argument("--data", default="data/asd")
    e.add_argument("--prefix", default="test_")
    e.add_argument("--models", nargs="+", required=True, help="model files; 'pretrained' = the shipped LR-ASD weights")
    a = p.parse_args(argv)
    if a.cmd == "build":
        build_all(Path(a.scenes), Path(a.out), a.splits.split(","))
    elif a.cmd == "meta":
        from .calibrate import compute_meta

        compute_meta(a.model, a.data)
    elif a.cmd == "train":
        d = Path(a.data)
        # train/val npz share a directory, split by filename prefix
        tmp_tr, tmp_va = d / "_train", d / "_val"
        for tmp, pre in ((tmp_tr, "train_"), (tmp_va, "val_")):
            tmp.mkdir(exist_ok=True)
            for f in d.glob(f"{pre}*.npz"):
                link = tmp / f.name
                if not link.exists():
                    link.symlink_to(f.resolve())
        res = train(tmp_tr, tmp_va, Path(a.out), epochs=a.epochs, lr=a.lr, steps_per_epoch=a.steps, freeze_audio=a.freeze_audio,
                    hard_p=a.hard_p, shift_p=a.shift_p)
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1))
    else:
        from .lrasd.model import load_model
        from .paths import ASD_MODEL

        scenes = load_scenes(Path(a.data), a.prefix)
        print(f"{len(scenes)} scenes with prefix '{a.prefix}'")
        for m in a.models:
            model = load_model(ASD_MODEL if m == "pretrained" else m).cpu()
            print(format_eval("pretrained" if m == "pretrained" else Path(m).stem, evaluate(model, scenes)))
