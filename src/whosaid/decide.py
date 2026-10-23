"""Learned word-level decision model.

The hand-written rules produce a word list with a speaker per word. This model looks at the same words and, for each
one, chooses among: each speaking face, "off-screen", or "drop" (garbage / duplicate). It is trained on synthetic
scenes where the truth is known, and compared with the rules on the very same words, so any gain is the decision
logic alone (it cannot recover words nobody transcribed).

Formulation: one row per (word, candidate). Label 1 if that candidate is correct for the word. A gradient-boosted tree
scores every row; per word the highest-scoring candidate wins. Candidate-type flags let one model handle faces,
off-screen and drop.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .asr import Word
from .attribute import smoothed_scores
from .bench import norm, track_tile
from .disk import ensure_free
from .media import ASD_FPS

R_ACT = 0.25

FEATURES = [
    "is_track", "is_off", "is_drop", "is_rule_pick",
    "src_stem", "src_mix_clear", "src_mix", "dur", "nchar", "word_margin",
    "n_cands", "n_active_ratio", "overlap_load",
    "cov", "raw", "cal", "cal_rank", "cal_gap", "ratio", "share", "from_this_stem", "tr_len_s", "tr_size", "tr_active",
    "max_cal", "max_raw", "max_ratio",
    "in_mix", "n_streams", "in_stem_this", "other_stems_with", "prev_same", "next_same", "gap_prev", "gap_next",
    "vsim", "vsim_gap", "max_vsim", "enroll_n", "vsim_rank",
]


@dataclass
class Row:
    word: int            # index into the word list
    cand: tuple          # ("track", id) | ("off",) | ("drop",)
    x: list[float]
    y: int = -1          # 1 correct, 0 wrong, -1 unlabeled


def candidate_tracks(a) -> list:
    from .pipeline import active_track_ids

    act = active_track_ids(a.tracks)
    return [t for t in a.tracks if t.id in act] or list(a.tracks)


_ECAPA = None


def _ecapa():
    global _ECAPA
    if _ECAPA is None:
        from speechbrain.inference.speaker import EncoderClassifier

        from .paths import MODELS_DIR

        _ECAPA = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(MODELS_DIR / "ecapa"),
                                                run_opts={"device": "cpu"})
    return _ECAPA


def voice_sims(words: list[Word], mix: np.ndarray, track_ids: list[int], win: float = 2.0, max_enroll: int = 40):
    """Self-enrolled voiceprints. A face's voiceprint is the mean ECAPA embedding of 2 s windows of the *mix* centred on
    its own confident, non-overlapping words (stem/clear-mix words, or a large calibrated lead). Each word is then compared
    with every voiceprint (leaving out enrollment windows within 1.5 s of the word, so a word doesn't vote for itself).
    Returns (sims[i][tid] or missing, enrollment size per track)."""
    import torch

    sr = 16000
    n = int(win * sr)
    mids = np.array([(w.start + w.end) / 2 for w in words])
    if len(words) == 0 or mix is None:
        return [{} for _ in words], {t: 0 for t in track_ids}
    segs = []
    for m in mids:
        a0 = int(max(0, min(m * sr - n / 2, len(mix) - n)))
        segs.append(mix[a0:a0 + n])
    model = _ecapa()
    embs = []
    with torch.no_grad():
        for i in range(0, len(segs), 64):
            b = torch.from_numpy(np.stack([np.pad(x, (0, n - len(x))) for x in segs[i:i + 64]]))
            embs.append(model.encode_batch(b).squeeze(1).numpy())
    E = np.concatenate(embs)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    spans = [(w.start, w.end, w.speaker) for w in words]
    enroll = {}
    for tid in track_ids:
        ids = []
        for i, w in enumerate(words):
            if w.speaker != tid:
                continue
            alone = not any(j != i and spk != tid and min(e, w.end + 0.3) - max(s_, w.start - 0.3) > 0 for j, (s_, e, spk) in enumerate(spans))
            confident = w.source in ("stem", "mix_clear") or w.margin >= 1.5
            if alone and confident:
                ids.append(i)
        if len(ids) > max_enroll:
            ids = [ids[k] for k in np.linspace(0, len(ids) - 1, max_enroll).astype(int)]
        enroll[tid] = ids
    sims: list[dict] = []
    for i in range(len(words)):
        d = {}
        for tid, ids in enroll.items():
            use = [j for j in ids if abs(mids[j] - mids[i]) > 1.5]
            if len(use) >= 3:
                c = E[use].mean(0)
                d[tid] = float(E[i] @ c / (np.linalg.norm(c) + 1e-9))
        sims.append(d)
    return sims, {t: len(v) for t, v in enroll.items()}


def _contains(stream: list[tuple], w: Word) -> bool:
    n = norm(w.text)
    return any(t == n and min(e, w.end) - max(b, w.start) > 0.3 * min(e - b, w.end - w.start) for b, e, t in stream)


def featurize(a, words: list[Word], mix_words: list[tuple] | None = None, mix: np.ndarray | None = None) -> list[Row]:
    """Rows for every (word, candidate). `words` are the rule pipeline's output (a.words after attribute())."""
    from .separate import energy_ratio

    scores = smoothed_scores(a.tracks)
    cands = candidate_tracks(a)
    H = a.info.height
    # per-track baseline = median raw ASD over these words (same calibration idea as the rules)
    raw = {}   # (word idx, track id) -> (raw, coverage)
    for i, w in enumerate(words):
        f0, f1 = int(w.start * ASD_FPS), max(int(w.end * ASD_FPS), int(w.start * ASD_FPS) + 1)
        for t in cands:
            if t.id not in scores:
                continue
            lo, hi = max(f0, t.start), min(f1, t.end + 1)
            cov = max(hi - lo, 0) / max(f1 - f0, 1)
            raw[(i, t.id)] = (float(scores[t.id][lo - t.start: hi - t.start].mean()) if hi > lo else -9.0, cov)
    base = {t.id: float(np.median([raw[(i, t.id)][0] for i in range(len(words)) if (i, t.id) in raw and raw[(i, t.id)][1] > 0.5] or [0.0]))
            for t in cands}
    spans = [(w.start, w.end, w.speaker) for w in words]
    mix_stream = mix_words or []
    stem_streams = {tid: [(w.start, w.end, norm(w.text)) for w in ws] for tid, ws in (a.stem_words or {}).items()}

    vs, enroll_n = voice_sims(words, mix, [t.id for t in cands])
    rows: list[Row] = []
    for i, w in enumerate(words):
        vsim = vs[i]
        vorder = sorted(vsim, key=vsim.get, reverse=True)
        ratio = {}
        for t in cands:
            ratio[t.id] = energy_ratio(a.stems[t.id], a.mix, w.start, w.end) if a.stems and t.id in a.stems else 0.0
        tot = sum(ratio.values()) + 1e-6
        cal = {t.id: raw[(i, t.id)][0] - base[t.id] for t in cands if (i, t.id) in raw}
        order = sorted(cal, key=cal.get, reverse=True)
        n_act = sum(r >= R_ACT for r in ratio.values())
        in_mix = float(_contains(mix_stream, w))
        in_stem = {tid: float(_contains(st, w)) for tid, st in stem_streams.items()}
        prev_w, next_w = (words[i - 1] if i > 0 else None), (words[i + 1] if i + 1 < len(words) else None)
        gap_prev = (w.start - prev_w.end) if prev_w else 9.0
        gap_next = (next_w.start - w.end) if next_w else 9.0
        load = sum(1 for s, e, spk in spans if spk != w.speaker and min(e, w.end) - max(s, w.start) > 0.5 * (w.end - w.start))
        common = dict(
            src_stem=float(w.source == "stem"), src_mix_clear=float(w.source == "mix_clear"), src_mix=float(w.source == "mix"),
            dur=w.end - w.start, nchar=float(len(w.text)), word_margin=float(w.margin), n_cands=float(len(cands)),
            n_active_ratio=float(n_act), overlap_load=float(load),
            max_cal=max(cal.values(), default=-9.0), max_raw=max((raw[(i, t.id)][0] for t in cands if (i, t.id) in raw), default=-9.0),
            max_ratio=max(ratio.values(), default=0.0),
            in_mix=in_mix, n_streams=in_mix + sum(in_stem.values()), gap_prev=min(gap_prev, 9.0), gap_next=min(gap_next, 9.0),
            max_vsim=max(vsim.values(), default=0.0),
        )

        def make(kind, tid=None):
            d = dict(common)
            d.update(is_track=float(kind == "track"), is_off=float(kind == "off"), is_drop=float(kind == "drop"),
                     is_rule_pick=float((kind == "track" and w.speaker == tid) or (kind == "off" and w.speaker is None)))
            if kind == "track":
                t = next(t for t in cands if t.id == tid)
                r, cov = raw.get((i, tid), (-9.0, 0.0))
                others = [cal[o] for o in cal if o != tid]
                d.update(cov=cov, raw=r, cal=cal.get(tid, -9.0), cal_rank=float(order.index(tid)) if tid in order else 9.0,
                         cal_gap=cal.get(tid, -9.0) - (max(others) if others else 0.0), ratio=ratio[tid], share=ratio[tid] / tot,
                         from_this_stem=float(w.source == "stem" and w.speaker == tid), tr_len_s=len(t.frames) / ASD_FPS,
                         tr_size=float(np.median(t.s)) / H, tr_active=1.0,
                         in_stem_this=in_stem.get(tid, 0.0), other_stems_with=sum(v for k, v in in_stem.items() if k != tid),
                         prev_same=float(prev_w is not None and prev_w.speaker == tid and gap_prev < 0.8),
                         next_same=float(next_w is not None and next_w.speaker == tid and gap_next < 0.8),
                         vsim=vsim.get(tid, 0.0), vsim_gap=vsim.get(tid, 0.0) - max([v for k, v in vsim.items() if k != tid], default=0.0),
                         enroll_n=float(enroll_n.get(tid, 0)), vsim_rank=float(vorder.index(tid)) if tid in vorder else 9.0)
            else:
                d.update(cov=0.0, raw=-9.0, cal=-9.0, cal_rank=9.0, cal_gap=0.0, ratio=0.0, share=0.0, from_this_stem=0.0,
                         tr_len_s=0.0, tr_size=0.0, tr_active=0.0, in_stem_this=0.0, other_stems_with=sum(in_stem.values()),
                         prev_same=float(kind == "off" and prev_w is not None and prev_w.speaker is None and gap_prev < 0.8),
                         next_same=float(kind == "off" and next_w is not None and next_w.speaker is None and gap_next < 0.8),
                         vsim=0.0, vsim_gap=0.0, enroll_n=0.0, vsim_rank=9.0)
            return d

        for t in cands:
            rows.append(Row(i, ("track", t.id), [make("track", t.id)[f] for f in FEATURES]))
        rows.append(Row(i, ("off",), [make("off")[f] for f in FEATURES]))
        rows.append(Row(i, ("drop",), [make("drop")[f] for f in FEATURES]))
    return rows


def label_rows(a, words: list[Word], rows: list[Row], labels: dict) -> None:
    """Fill Row.y from the scene's ground truth. Words with no correct candidate available stay unlabeled (-1)."""
    ref, tiles = labels["words"], labels["tiles"]
    tile_of = {t.id: track_tile(t, t.start + len(t.frames) // 2, tiles) for t in a.tracks}
    used = set()
    truth: dict[int, str | None] = {}
    for i, w in enumerate(words):
        best = None
        for j, r in enumerate(ref):
            if j in used or norm(r["text"]) != norm(w.text):
                continue
            if min(w.end, r["end"]) - max(w.start, r["start"]) > 0.5 * min(w.end - w.start, r["end"] - r["start"]):
                best = j
                break
        if best is not None:
            used.add(best)
        truth[i] = ref[best]["tile"] if best is not None else None
    by_word: dict[int, list[Row]] = {}
    for r in rows:
        by_word.setdefault(r.word, []).append(r)
    for i, rs in by_word.items():
        tr = truth[i]
        for r in rs:
            if r.cand[0] == "drop":
                r.y = int(tr is None)
            elif tr is None:
                r.y = 0
            elif r.cand[0] == "off":
                r.y = int(tr == "X")
            else:
                r.y = int(tile_of[r.cand[1]] == tr)
        if not any(r.y == 1 for r in rs):
            for r in rs:
                r.y = -1          # the right answer is not among the candidates; cannot learn from this word


def choose(rows: list[Row], proba: np.ndarray, delta: float = 0.0, drop_p: float = 0.0) -> dict[int, tuple]:
    """Highest-probability candidate per word.

    delta: only overrule the rule pipeline's pick if the model's best beats it by this much probability.
    drop_p: only drop a word if the model is at least this sure it is garbage (else take its best non-drop candidate)."""
    by_word: dict[int, list[tuple[float, Row]]] = {}
    for r, p in zip(rows, proba):
        by_word.setdefault(r.word, []).append((float(p), r))
    out = {}
    for i, rs in by_word.items():
        ranked = sorted(rs, key=lambda t: -t[0])
        p0, best = ranked[0]
        if best.cand[0] == "drop" and p0 < drop_p:
            p0, best = next((p, r) for p, r in ranked if r.cand[0] != "drop")
        rule = next(((p, r) for p, r in rs if r.x[FEATURES.index("is_rule_pick")] == 1.0), None)
        if rule is not None and best is not rule[1] and p0 - rule[0] < delta and not (best.cand[0] == "drop" and p0 >= drop_p > 0):
            best = rule[1]
        out[i] = best.cand
    return out


def apply_choices(words: list[Word], choice: dict[int, tuple]) -> list[Word]:
    out = []
    for i, w in enumerate(words):
        c = choice[i]
        if c[0] == "drop":
            continue
        w.speaker = c[1] if c[0] == "track" else None
        out.append(w)
    return out


def make_model(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=250, l2_regularization=1.0,
                                          min_samples_leaf=20, random_state=seed)


def _cols(features):
    return [FEATURES.index(f) for f in (features or FEATURES)]


def fit(rows_by_scene: dict[str, list[Row]], scenes: list[str], seed: int = 0, features: list[str] | None = None):
    cols = _cols(features)
    X = np.array([[r.x[c] for c in cols] for s in scenes for r in rows_by_scene[s] if r.y >= 0], np.float32)
    y = np.array([r.y for s in scenes for r in rows_by_scene[s] if r.y >= 0])
    m = make_model(seed)
    m.fit(X, y)
    return m


def predict_choices(model, rows: list[Row], features: list[str] | None = None, delta: float = 0.0, drop_p: float = 0.0):
    cols = _cols(features)
    X = np.array([[r.x[c] for c in cols] for r in rows], np.float32)
    return choose(rows, model.predict_proba(X)[:, 1], delta, drop_p)


def save(model, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(model=model, features=FEATURES), open(path, "wb"))


def load(path):
    d = pickle.load(open(path, "rb"))
    assert d["features"] == FEATURES, "decider was trained with a different feature set"
    return d["model"]


# ----------------------------------------------------------------------------- experiment runner

def build_scene(video: Path, labels_path: Path, workdir: Path, whisper_model: str = "small") -> dict:
    """Run (cached) analysis + the rule attribution, then featurize and label. Everything needed to train/score later."""
    from .bench import predict_tiles
    from .pipeline import analyze, attribute

    labels = json.loads(Path(labels_path).read_text())
    a = analyze(video, workdir, whisper_model, language="en", separate="auto")
    mix_words = [(w.start, w.end, norm(w.text)) for w in a.words]     # the plain-mix transcript, before the rules merge things
    attribute(a)                                       # rules (default fallback)
    words = list(a.words)
    mix = a.mix
    if mix is None:                      # scenes without separation do not keep the audio; read it for the voiceprints
        import tempfile

        from .media import extract_audio, read_wav

        with tempfile.TemporaryDirectory() as tmp:
            extract_audio(video, Path(tmp) / "a.wav")
            mix = read_wav(Path(tmp) / "a.wav").astype(np.float32) / 32768.0
    rows = featurize(a, words, mix_words, mix)
    label_rows(a, words, rows, labels)
    tile_of = {t.id: track_tile(t, t.start + len(t.frames) // 2, labels["tiles"]) for t in a.tracks}
    rule_pred = predict_tiles(a, labels)
    return dict(scene=Path(labels_path).name.replace(".labels.json", ""), labels=labels, rows=rows, tile_of=tile_of,
                words=[dict(start=w.start, end=w.end, text=w.text, source=w.source, speaker=w.speaker) for w in words],
                rule_pred=rule_pred)


def _pred_from_choice(sd: dict, choice: dict[int, tuple]) -> list[dict]:
    out = []
    for i, w in enumerate(sd["words"]):
        c = choice[i]
        if c[0] == "drop":
            continue
        out.append(dict(start=w["start"], end=w["end"], text=w["text"], source=w["source"],
                        tile=sd["tile_of"][c[1]] if c[0] == "track" else "X"))
    return out


def _agg(results: list[dict]) -> dict:
    e = sum(r["errors"] for r in results); n = sum(r["ref_words"] for r in results)
    s = lambda k: [sum(r[k][0] for r in results), sum(r[k][1] for r in results)]
    pct = lambda p: 100 * p[0] / p[1] if p[1] else float("nan")
    return dict(scenes=len(results), cpwer=100 * e / max(n, 1), attr_clear=pct(s("attr_acc_clear")), attr_overlap=pct(s("attr_acc_overlap")),
                recall_clear=pct(s("recall_clear")), recall_overlap=pct(s("recall_overlap")))


def compare(data: dict[str, dict], preds_model: dict[str, list[dict]]) -> str:
    from .bench import score

    kinds = sorted({d["labels"]["kind"] for d in data.values()})
    lines = [f"{'':<12}{'':<8}{'scenes':>6}{'cpWER':>8}{'attr clear':>12}{'attr overlap':>14}{'recall clear':>14}{'recall overlap':>16}"]
    for group in ["ALL"] + kinds + ["freeze", "mouthing"]:
        names = [n for n, d in data.items() if group == "ALL" or d["labels"]["kind"] == group or d["labels"]["listener"] == group]
        if not names:
            continue
        for tag, get in (("rules", lambda n: data[n]["rule_pred"]), ("model", lambda n: preds_model[n])):
            g = _agg([score(get(n), data[n]["labels"]) for n in names])
            lines.append(f"{group:<12}{tag:<8}{g['scenes']:>6}{g['cpwer']:>7.1f}%{g['attr_clear']:>11.1f}%{g['attr_overlap']:>13.1f}%"
                         f"{g['recall_clear']:>13.1f}%{g['recall_overlap']:>15.1f}%")
    return "\n".join(lines)


def cross_validate(data: dict[str, dict], k: int = 6, seed: int = 0, features: list[str] | None = None,
                   delta: float = 0.0, drop_p: float = 0.0) -> dict[str, list[dict]]:
    """Group k-fold by scene: every scene is predicted by a model that never saw it."""
    rows_by = {n: d["rows"] for n, d in data.items()}
    names = sorted(data, key=lambda n: (data[n]["labels"]["kind"], n))
    folds = [names[i::k] for i in range(k)]
    out = {}
    for held in folds:
        train = [n for n in names if n not in held]
        m = fit(rows_by, train, seed, features)
        for n in held:
            out[n] = _pred_from_choice(data[n], predict_choices(m, rows_by[n], features, delta, drop_p))
    return out


def main(argv):
    import argparse

    p = argparse.ArgumentParser(prog="whosaid decide")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="featurize+label scenes into data/decide/<scene>.pkl")
    b.add_argument("--scenes", default="data/synth")
    b.add_argument("--glob", default="*")
    b.add_argument("--workdirs", required=True)
    b.add_argument("--out", default="data/decide")
    e = sub.add_parser("eval", help="cross-validate on dev scenes, then (optionally) score once on test")
    e.add_argument("--dev", default="data/decide/dev", help="dir of built dev pkls")
    e.add_argument("--test", help="dir of built test pkls (scored once)")
    e.add_argument("--folds", type=int, default=6)
    e.add_argument("--save", help="write the model trained on all dev scenes here")
    a = p.parse_args(argv)
    if a.cmd == "build":
        import fnmatch

        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        for lp in sorted(Path(a.scenes).rglob("*.labels.json")):
            name = lp.name.replace(".labels.json", "")
            if not fnmatch.fnmatch(name, a.glob) or (out / f"{name}.pkl").exists():
                continue
            wd = Path(a.workdirs) / f"{name}.work"
            if not wd.exists():
                continue
            ensure_free(f"featurizing {name}")
            sd = build_scene(lp.with_name(name + ".mp4"), lp, wd)
            pickle.dump(sd, open(out / f"{name}.pkl", "wb"))
            n = sum(r.y >= 0 for r in sd["rows"]) // 1
            print(f"{name}: {len(sd['words'])} words, {n} labeled rows", flush=True)
        return
    load_dir = lambda d: {p.stem: pickle.load(open(p, "rb")) for p in sorted(Path(d).glob("*.pkl"))}
    dev = load_dir(a.dev)
    print(f"dev scenes: {len(dev)}, rows: {sum(len(d['rows']) for d in dev.values())}\n")
    print("== grouped cross-validation on dev ==")
    print(compare(dev, cross_validate(dev, a.folds)))
    rows_by = {n: d["rows"] for n, d in dev.items()}
    model = fit(rows_by, list(dev))
    if a.save:
        save(model, a.save)
    if a.test:
        test = load_dir(a.test)
        print(f"\n== held-out test ({len(test)} scenes), model trained on all dev ==")
        print(compare(test, {n: _pred_from_choice(d, predict_choices(model, d["rows"])) for n, d in test.items()}))
