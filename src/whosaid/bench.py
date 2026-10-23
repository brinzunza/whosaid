"""Benchmark the pipeline on synthetic scenes (see synth/compose.py) and report speaker-attributed word accuracy.

Speaker identity is judged by *where* a predicted word's face is: the scene layout says which tile (person) each
canvas pixel belongs to, so scoring does not depend on the face detector being right about anything but position.
Words with no face are "off-screen" and are compared with the audio-only voice (tile X) if the scene has one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .disk import ensure_free


def norm(t: str) -> str:
    return "".join(c for c in t.lower() if c.isalnum())


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def track_tile(track, f: int, tiles: list[dict]) -> str:
    i = min(max(f - track.start, 0), len(track.frames) - 1)
    x, y = float(track.cx[i]), float(track.cy[i])
    best, bd = None, 1e18
    for t in tiles:
        if not t["rect"]:
            continue
        rx, ry, rw, rh = t["rect"]
        dx, dy = max(rx - x, 0, x - (rx + rw)), max(ry - y, 0, y - (ry + rh))
        d = dx * dx + dy * dy
        if d < bd:
            best, bd = t["id"], d
    return best


def predict_tiles(a, labels) -> list[dict]:
    """Map each predicted word to a tile id ('X' = off-screen bucket)."""
    by_id = {t.id: t for t in a.tracks}
    out = []
    for w in a.words:
        if w.speaker is None:
            tile = "X"
        else:
            tile = track_tile(by_id[w.speaker], int((w.start + w.end) / 2 * 25), labels["tiles"])
        out.append(dict(start=w.start, end=w.end, text=w.text, tile=tile, source=w.source))
    return out


def score(pred: list[dict], labels: dict) -> dict:
    ref = labels["words"]
    ids = sorted({t["id"] for t in labels["tiles"]} | {"X"})
    errs = tot = 0
    per = {}
    for tid in ids:
        r = [norm(w["text"]) for w in ref if w["tile"] == tid]
        h = [norm(w["text"]) for w in sorted(pred, key=lambda w: w["start"]) if w["tile"] == tid]
        r, h = [x for x in r if x], [x for x in h if x]
        e = edit_distance(r, h)
        errs += e; tot += len(r)
        per[tid] = dict(ref=len(r), hyp=len(h), errors=e)
    overlap = labels["overlap"]
    inov = lambda w: any(a <= (w["start"] + w["end"]) / 2 <= b for a, b in overlap)

    # temporal+text matching of hypothesis words to reference words (any tile): attribution accuracy
    matched = {"ov": [0, 0], "no": [0, 0]}      # [correct, matched]
    ref_hit = {"ov": [0, 0], "no": [0, 0]}      # [recalled, total ref]
    used = set()
    for w in pred:
        best = None
        for j, r in enumerate(ref):
            if j in used or norm(r["text"]) != norm(w["text"]):
                continue
            ov = min(w["end"], r["end"]) - max(w["start"], r["start"])
            if ov > 0.5 * min(w["end"] - w["start"], r["end"] - r["start"]):
                best = j; break
        if best is not None:
            used.add(best)
            k = "ov" if inov(ref[best]) else "no"
            matched[k][1] += 1
            matched[k][0] += int(ref[best]["tile"] == w["tile"])
    for j, r in enumerate(ref):
        k = "ov" if inov(r) else "no"
        ref_hit[k][1] += 1
        ref_hit[k][0] += int(j in used)
    return dict(cpwer=errs / max(tot, 1), errors=errs, ref_words=tot, per_tile=per,
                attr_acc_overlap=matched["ov"], attr_acc_clear=matched["no"],
                recall_overlap=ref_hit["ov"], recall_clear=ref_hit["no"])


def diagnose(a, labels) -> dict:
    """Classify every error. Reference words: ok / misattributed (by where the hypothesis came from) / filtered out by
    our sanity checks / never transcribed. Hypothesis words not matching any reference: leaked copies vs garbage."""
    pred = predict_tiles(a, labels)
    ref = labels["words"]
    ov = labels["overlap"]
    inov = lambda w: any(x <= (w["start"] + w["end"]) / 2 <= y for x, y in ov)

    def hit(w, r):
        o = min(w["end"], r["end"]) - max(w["start"], r["start"])
        return norm(w["text"]) == norm(r["text"]) and o > 0.5 * min(w["end"] - w["start"], r["end"] - r["start"])

    out = {}
    def bump(region, key):
        out.setdefault(region, {}).setdefault(key, 0)
        out[region][key] += 1

    used_h = set()
    dropped = [(w, tid, why) for w, tid, why in (a.dropped or [])]
    for r in ref:
        reg = "overlap" if inov(r) else "clear"
        j = next((k for k, w in enumerate(pred) if k not in used_h and hit(w, r)), None)
        if j is not None:
            used_h.add(j)
            src = pred[j]["source"]
            bump(reg, "ok" if pred[j]["tile"] == r["tile"] else f"wrong_tile({src},{'off' if pred[j]['tile']=='X' else 'other'})")
        else:
            d = next((why for w, tid, why in dropped if norm(w.text) == norm(r["text"]) and
                      min(w.end, r["end"]) - max(w.start, r["start"]) > 0.5 * (r["end"] - r["start"])), None)
            bump(reg, f"missed_filtered({d})" if d else "missed_never_heard")
    for k, w in enumerate(pred):
        if k in used_h:
            continue
        reg = "overlap" if inov(dict(start=w["start"], end=w["end"])) else "clear"
        near = any(norm(r["text"]) == norm(w["text"]) and abs(r["start"] - w["start"]) < 1.5 for r in ref)
        bump(reg, f"extra_{w['source']}_{'timing_or_dup' if near else 'garbage'}")
    return out


def si_snr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est, ref = est - est.mean(), ref - ref.mean()
    t = (est @ ref) / (ref @ ref + eps) * ref
    return float(10 * np.log10((t @ t + eps) / ((est - t) @ (est - t) + eps)))


def stem_quality(a, labels, stems_npz) -> dict:
    """SI-SNR (dB) of each extracted voice against the true dry stem of the tile it was assigned to."""
    if a.stems is None:
        return {}
    truth = np.load(stems_npz)
    by_id = {t.id: t for t in a.tracks}
    out = {}
    for tid, st in a.stems.items():
        tile = track_tile(by_id[tid], by_id[tid].start + len(by_id[tid].frames) // 2, labels["tiles"])
        if tile not in truth:
            continue
        ref = truth[tile].astype(np.float32)
        n = min(len(ref), len(st))
        active = np.abs(ref[:n]) > 0.01
        if active.sum() < 8000:
            continue
        key = f"{tile}"
        v = si_snr(st[:n][active], ref[:n][active])
        out[key] = max(v, out.get(key, -99.0))
    return out


def run_scene(video: Path, labels_path: Path, workdir: Path, whisper_model="small", separate="auto", log=lambda m: None,
              fallback: str = "asd", asd_model: str | None = None, asd_calib: str = "none", fuse: str | None = None,
              static_filter: bool = False) -> dict:
    from .pipeline import analyze, attribute

    labels = json.loads(labels_path.read_text())
    timings: dict = {}
    t0 = time.time()
    a = analyze(video, workdir, whisper_model, language="en", use_voices=True, log=log, separate=separate, timings=timings, asd_model=asd_model, asd_calib=asd_calib, fuse=fuse, static_filter=static_filter)
    attribute(a, log, fallback)
    timings["total_cold"] = time.time() - t0
    pred = predict_tiles(a, labels)
    res = score(pred, labels)
    res["diagnosis"] = diagnose(a, labels)
    res["si_snr_db"] = stem_quality(a, labels, str(labels_path).replace(".labels.json", ".stems.npz"))
    res["timings"] = {k: round(v, 1) for k, v in timings.items()}
    res["n_tracks"] = len(a.tracks)
    res["n_tiles_on_screen"] = sum(t["on_screen"] for t in labels["tiles"])
    res["kind"], res["split"], res["listener"] = labels["kind"], labels["split"], labels["listener"]
    res["duration"] = labels["duration"]
    res["separated"] = a.stems is not None
    return res


def summarize(results: dict[str, dict]) -> str:
    def agg(rs):
        e = sum(r["errors"] for r in rs); n = sum(r["ref_words"] for r in rs)
        ao = [sum(r["attr_acc_overlap"][0] for r in rs), sum(r["attr_acc_overlap"][1] for r in rs)]
        ac = [sum(r["attr_acc_clear"][0] for r in rs), sum(r["attr_acc_clear"][1] for r in rs)]
        ro = [sum(r["recall_overlap"][0] for r in rs), sum(r["recall_overlap"][1] for r in rs)]
        rc = [sum(r["recall_clear"][0] for r in rs), sum(r["recall_clear"][1] for r in rs)]
        pct = lambda p: f"{100 * p[0] / p[1]:5.1f}%" if p[1] else "   - "
        return f"{len(rs):>3} scenes {n:>5} ref words | cpWER {100 * e / max(n, 1):5.1f}% | attribution ok: clear {pct(ac)} overlap {pct(ao)} | recall: clear {pct(rc)} overlap {pct(ro)}"
    lines = ["ALL        " + agg(list(results.values()))]
    for k in sorted({r["kind"] for r in results.values()}):
        lines.append(f"{k:<10} " + agg([r for r in results.values() if r["kind"] == k]))
    for k in sorted({r["listener"] for r in results.values()}):
        lines.append(f"listen={k:<5}" + agg([r for r in results.values() if r["listener"] == k]))
    snr = [v for r in results.values() for v in r["si_snr_db"].values()]
    if snr:
        lines.append(f"extracted-voice SI-SNR: mean {np.mean(snr):.1f} dB, median {np.median(snr):.1f} dB over {len(snr)} voices")
    stages = {}
    for r in results.values():
        for k, v in r["timings"].items():
            stages.setdefault(k, []).append(v / max(r["duration"], 1))
    lines.append("seconds per second of video: " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in stages.items()))
    return "\n".join(lines)


def main(argv):
    import argparse

    p = argparse.ArgumentParser(prog="whosaid bench")
    p.add_argument("scenes", help="dir with *.mp4 + *.labels.json (searched recursively)")
    p.add_argument("--out", required=True)
    p.add_argument("--split", help="only scenes of these splits, comma separated (e.g. val,train)")
    p.add_argument("--glob", help="only scenes whose name matches this glob, e.g. 'test_*_00_*'")
    p.add_argument("--limit", type=int)
    p.add_argument("--separate", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--fallback", choices=["off", "asd", "asd_if_alone", "hybrid"], default="asd",
                   help="what to do with words only the full mix produced")
    p.add_argument("--asd-model", help="speaking-detector weights to re-score the cached faces with (e.g. models/lrasd_ft.model)")
    p.add_argument("--asd-calib", choices=["none", "model", "match", "face"], default="none")
    p.add_argument("--fuse", help="fusion model json (voice energy + detector score), or 'on' for models/fusion.json")
    p.add_argument("--stem-min", type=float, help="minimum (fused) speaking score for a face to own a stem word (default -0.5)")
    p.add_argument("--static-filter", choices=["on", "off"], default="off")
    p.add_argument("--workdirs", help="reuse cached stages from this dir (default: --out); lets you re-score new logic in seconds")
    a = p.parse_args(argv)
    if a.stem_min is not None:
        from . import pipeline as _pl

        _pl.STEM_MIN[0] = a.stem_min
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    labs = sorted(Path(a.scenes).rglob("*.labels.json"))
    if a.split:
        labs = [l for l in labs if l.name.startswith(tuple(x + "_" for x in a.split.split(",")))]
    if a.glob:
        import fnmatch

        labs = [l for l in labs if fnmatch.fnmatch(l.name, a.glob + ".labels.json")]
    if a.limit:
        labs = labs[: a.limit]
    for lp in labs:
        name = lp.name.replace(".labels.json", "")
        if name in results:
            continue
        ensure_free(f"scoring {name}")
        t0 = time.time()
        results[name] = run_scene(lp.with_name(name + ".mp4"), lp, Path(a.workdirs or out) / f"{name}.work", a.whisper_model,
                                  a.separate, fallback=a.fallback, asd_model=a.asd_model, asd_calib=a.asd_calib, fuse=a.fuse, static_filter=a.static_filter == "on")
        res_path.write_text(json.dumps(results, indent=1))
        r = results[name]
        print(f"{name}: cpWER {100 * r['cpwer']:.1f}% tracks={r['n_tracks']}/{r['n_tiles_on_screen']} ({time.time() - t0:.0f}s)", flush=True)
    print(summarize(results))
