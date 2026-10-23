"""Collect labelled frames from cached scenes, compare fusion feature sets by scene-grouped CV, fit and evaluate."""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np

from .asd_train import CAT, _auc, frame_labels
from .calibrate import calibrate_tracks, load_meta
from .fusion import ALL_FEATURES, Fuser, fit, scene_features
from .media import extract_audio, read_wav
from .paths import ASD_MODEL

SETS = {
    "asd + voice": ["asd", "lr", "asd_x_lr"],
    "asd + voice + share": ["asd", "lr", "asd_x_lr", "share"],
    "asd + voice + share + others": ["asd", "lr", "asd_x_lr", "share", "lr_other"],
    "voice only": ["lr", "share", "lr_other"],
}


def collect(workdirs: Path, scenes_dir: Path, pattern: str, log=print) -> dict[str, dict]:
    """scene -> dict(X, y, cat, tid arrays) for every ASD-active face that has a cached stem."""
    from .pipeline import active_track_ids

    out = {}
    for wd in sorted(workdirs.glob(f"{pattern}.work")):
        name = wd.name.replace(".work", "")
        lab = next(iter(scenes_dir.rglob(f"{name}.labels.json")), None)
        if lab is None or not (wd / "stems.pkl").exists():
            continue
        tracks = pickle.loads((wd / "tracks.pkl").read_bytes())
        stems = pickle.loads((wd / "stems.pkl").read_bytes())
        calibrate_tracks(tracks, load_meta(ASD_MODEL), "model")
        active = active_track_ids(tracks)
        stems = {i: s for i, s in stems.items() if i in active}
        if not stems:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            extract_audio(lab.with_name(name + ".mp4"), Path(tmp) / "a.wav")
            mix = read_wav(Path(tmp) / "a.wav").astype(np.float32) / 32768.0
        feats = scene_features(tracks, stems, {t.id: np.asarray(t.score, float) for t in tracks}, mix)
        fl = frame_labels(json.loads(lab.read_text()), tracks)
        X, y, cat = [], [], []
        for tid, F in feats.items():
            yy, cc = fl[tid]
            n = min(len(F), len(yy))
            X.append(F[:n]); y.append(yy[:n].astype(int)); cat.append(cc[:n])
        out[name] = dict(X=np.concatenate(X), y=np.concatenate(y), cat=np.concatenate(cat))
    log(f"{len(out)} scenes, {sum(len(d['y']) for d in out.values())} face-frames")
    return out


def aucs(score: np.ndarray, cat: np.ndarray) -> dict:
    pos = score[cat == 0]
    r = {CAT[c]: _auc(pos, score[cat == c]) for c in (1, 2, 3, 4)}
    r["all"] = _auc(pos, score[cat != 0])
    return r


def fmt(name: str, r: dict) -> str:
    return f"{name:<32} all {r['all']:.3f} | vs mouthing {r['mouthing']:.3f}  freeze {r['freeze']:.3f}  silent {r['silent']:.3f}"


def cross_validate(data: dict[str, dict], k: int = 6, seed: int = 0) -> dict[str, np.ndarray]:
    names = sorted(data)
    folds = [names[i::k] for i in range(k)]
    oof = {s: np.zeros(sum(len(d["y"]) for d in data.values())) for s in SETS}
    offs = {}
    o = 0
    for n in names:
        offs[n] = (o, o + len(data[n]["y"])); o += len(data[n]["y"])
    for held in folds:
        tr = [n for n in names if n not in held]
        X = np.concatenate([data[n]["X"] for n in tr]); y = np.concatenate([data[n]["y"] for n in tr])
        for sname, feats in SETS.items():
            f = fit(X, y, feats)
            for n in held:
                a, b = offs[n]
                oof[sname][a:b] = f.logit(data[n]["X"])
    return oof


def main(argv):
    import argparse

    p = argparse.ArgumentParser(prog="whosaid fuse")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("cv", help="compare feature sets on the dev scenes by scene-grouped cross-validation")
    e.add_argument("--workdirs", default="data/bench/dev_v2")
    e.add_argument("--scenes", default="data/synth")
    e.add_argument("--pattern", default="*")
    t = sub.add_parser("train", help="fit on all dev scenes and save")
    t.add_argument("--workdirs", default="data/bench/dev_v2")
    t.add_argument("--scenes", default="data/synth")
    t.add_argument("--features", default="asd + voice + share + others", choices=list(SETS))
    t.add_argument("--out", default="models/fusion.json")
    v = sub.add_parser("eval", help="frame-level evaluation of a saved fuser on other scenes (e.g. test)")
    v.add_argument("--workdirs", default="data/bench/baseline_v1")
    v.add_argument("--scenes", default="data/synth")
    v.add_argument("--pattern", default="test_*_00_*")
    v.add_argument("--fuser", default="models/fusion.json")
    a = p.parse_args(argv)
    if a.cmd == "cv":
        data = collect(Path(a.workdirs), Path(a.scenes), a.pattern)
        names = sorted(data)
        cat = np.concatenate([data[n]["cat"] for n in names]); y = np.concatenate([data[n]["y"] for n in names])
        asd = np.concatenate([data[n]["X"][:, 0] for n in names])
        print(f"speaking {int((cat == 0).sum())}, mouthing {int((cat == 1).sum())}, freeze {int((cat == 2).sum())}, silent {int((cat == 3).sum())}, false-face {int((cat == 4).sum())}\n")
        print(fmt("detector alone", aucs(asd, cat)))
        for s, z in cross_validate(data).items():
            print(fmt(s, aucs(z, cat)))
    elif a.cmd == "train":
        data = collect(Path(a.workdirs), Path(a.scenes), "*")
        X = np.concatenate([data[n]["X"] for n in sorted(data)]); y = np.concatenate([data[n]["y"] for n in sorted(data)])
        f = fit(X, y, SETS[a.features])
        f.save(a.out)
        print(f"saved {a.out}: features {f.features}, coef {np.round(f.coef, 3).tolist()}, threshold t0 {f.meta['t0']:+.2f}, gap {f.meta['gap']:.2f}")
    else:
        data = collect(Path(a.workdirs), Path(a.scenes), a.pattern)
        f = Fuser.load(a.fuser)
        names = sorted(data)
        X = np.concatenate([data[n]["X"] for n in names]); cat = np.concatenate([data[n]["cat"] for n in names])
        print(fmt("detector alone", aucs(X[:, 0], cat)))
        print(fmt("fused", aucs(f.score(X), cat)))
        pos = cat == 0
        for name, sc in (("detector alone", X[:, 0]), ("fused", f.score(X))):
            print(f"{name:<16} at score > 1.0: speaking recalled {100 * (sc[pos] > 1).mean():.0f}%, wrongly fires on " +
                  ", ".join(f"{CAT[c]} {100 * (sc[cat == c] > 1).mean():.0f}%" for c in (1, 2, 3)))
