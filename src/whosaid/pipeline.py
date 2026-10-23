"""Orchestration. Heavy stages (faces, ASD scores, transcript, voice embeddings) are cached in a work dir so
attribution / layout / render changes can be re-run in seconds. Delete a file in the work dir to redo that stage."""
from __future__ import annotations

import json
import pickle
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .asd import score_tracks
from .asr import Word, transcribe
from .attribute import assign_hold, attribute_words, make_chunks, retime_even, smoothed_scores
from .faces import Track, track_faces
from .media import VideoInfo, extract_audio, probe
from .render import render


@dataclass
class Analysis:
    info: VideoInfo
    tracks: list[Track]
    words: list[Word]
    centres: np.ndarray   # voice-window centres (s)
    emb: np.ndarray       # voice-window embeddings
    mix: np.ndarray | None = None                    # 16 kHz mono mixture
    stems: dict | None = None                        # track id -> extracted voice (same length as mix)
    stem_words: dict | None = None                   # track id -> raw ASR words of the gated stem
    dropped: list | None = None                      # (word, reason) stem words rejected by the sanity checks
    langs: dict | None = None                        # 'mix' / track id -> (language, probability) when --translate-to is on


def _cached(path: Path | None, fresh: bool, fn):
    if path is not None and path.exists() and not fresh:
        return pickle.loads(path.read_bytes())
    out = fn()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(out))
    return out


def active_track_ids(tracks, thresh: float = 1.0, min_frames: int = 25) -> set[int]:
    """Faces that ever show real speaking evidence. Static false faces (a portrait on a wall) and people who never
    speak are not worth a separation pass, and extracting from them yields garbage words."""
    from .attribute import smoothed_scores

    sc = smoothed_scores(tracks)
    return {t.id for t in tracks if t.id in sc and int((sc[t.id] > thresh).sum()) >= min_frames}


def _overlapping(tracks) -> bool:
    tracks = [t for t in tracks if not getattr(t, "static", False)]       # a photo next to a person is not a second speaker
    return any(a.start <= b.end and b.start <= a.end for i, a in enumerate(tracks) for b in tracks[i + 1:])


OVERLAP_MIN_S = 3.0      # 'auto' separates voices only if at least this many seconds have two faces clearly speaking at once


def overlap_seconds(tracks, thresh: float = 1.0) -> float:
    """Seconds during which two or more on-screen (non-static) faces are both scored as speaking: the only time the
    mixture can hold two voices that a single transcript would garble. Separation costs two thirds of the run time and
    can hurt a clip of clean turn-taking (extracted voices leak and drop words), so 'auto' skips it without overlap."""
    from .attribute import smoothed_scores

    tracks = [t for t in tracks if not getattr(t, "static", False)]
    if len(tracks) < 2:
        return 0.0
    sc = smoothed_scores(tracks)
    n = max(t.end + 1 for t in tracks)
    speaking = np.zeros(n)
    for t in tracks:
        if t.id in sc:
            on = np.zeros(n, bool)
            on[t.start: t.end + 1] = sc[t.id][: len(t.frames)] > thresh
            speaking += on
    return float((speaking >= 2).sum()) / 25.0


def analyze(src, workdir=None, whisper_model="small", language=None, use_voices=True, fresh=False, log=lambda m: None,
            separate="auto", timings: dict | None = None, asd_model: str | None = None, asd_calib: str = "none", fuse: str | None = None,
            translate: str | None = None, static_filter: bool = False) -> Analysis:
    """Stages 1-3: face tracks, ASD scores, transcript, voice-window embeddings (all cached in workdir)."""
    from .translate import check_target, transcribe_for_captions

    check_target(translate)
    tr_tag = f".tr-{translate}" if translate else ""
    src = Path(src)
    info = probe(src)
    if not info.has_audio:
        raise SystemExit("input has no audio track")
    wd = Path(workdir) if workdir else None
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_audio(src, wav)

        tm = timings if timings is not None else {}

        def faces_and_scores():
            t0 = time.time()
            log("1/5 tracking faces")
            tracks = track_faces(src, info)
            tm["faces"] = time.time() - t0
            log(f"    {len(tracks)} face track(s) ({time.time() - t0:.0f}s)")
            log("2/5 detecting who is speaking (LR-ASD)")
            t0 = time.time()
            score_tracks(src, info, tracks, wav)
            tm["asd"] = time.time() - t0
            return tracks

        tracks = _cached(wd / "tracks.pkl" if wd else None, fresh, faces_and_scores)
        if asd_model and asd_model != "pretrained":       # re-score the cached faces with another speaking detector
            tag = Path(asd_model).stem

            def rescore():
                log(f"2/5 re-scoring faces with {tag}")
                t0 = time.time()
                score_tracks(src, info, tracks, wav, asd_model, lips=False)
                tm["asd_rescore"] = time.time() - t0
                return {t.id: t.score for t in tracks}

            sc = _cached(wd / f"asd.{tag}.pkl" if wd else None, fresh, rescore)
            for t in tracks:
                t.score = sc[t.id]

        def asr():
            log(f"3/5 transcribing (faster-whisper {whisper_model})")
            t0 = time.time()
            if translate:                                    # language is detected from the speech, then translated if needed
                from .media import read_wav

                r = transcribe_for_captions(read_wav(wav).astype(np.float32) / 32768.0, whisper_model, forced=language, target=translate)
                out = (r.words, r.source_lang, r.language_prob)
            else:
                out = transcribe(wav, whisper_model, language)
            tm["whisper_mix"] = time.time() - t0
            return out

        mix_obj = _cached(wd / f"words.{whisper_model}{tr_tag}.pkl" if wd else None, fresh, asr)
        words, mix_lang, mix_prob = mix_obj if translate else (mix_obj, None, 0.0)
        langs = {"mix": (mix_lang, mix_prob)} if translate else None
        if translate:
            log(f"    speech language of the mix: {mix_lang} ({mix_prob:.2f})")
        for w in words:  # the cache holds raw ASR output; attribution fields are recomputed every run
            w.speaker, w.voice, w.margin = None, None, 0.0
        if fuse and asd_calib == "none":
            asd_calib = "model"                               # the fuser was trained on calibrated detector scores
        if asd_calib != "none":
            from .calibrate import calibrate_tracks
            from .paths import ASD_MODEL

            for t in tracks:                                  # cached tracks may carry scores calibrated by an earlier call
                if getattr(t, "score_raw", None) is not None:
                    t.score = t.score_raw
            from .calibrate import load_meta_for

            mp = asd_model if asd_model and asd_model != "pretrained" else ASD_MODEL
            meta = load_meta_for(mp, match_pretrained=(asd_calib == "match"))
            calibrate_tracks(tracks, meta, "model" if asd_calib == "match" else asd_calib)
        if static_filter:
            from .photos import mark_static

            flagged = mark_static(tracks)                      # cached tracks start unmarked; recomputed on every call
            if flagged:
                log(f"    ignoring {len(flagged)} static face(s) (photos / posters / frozen overlays): {flagged}")
        active = active_track_ids(tracks)
        ovl = overlap_seconds(tracks) if separate == "auto" and _overlapping(tracks) else 0.0
        do_sep = bool(tracks) and (separate == "on" or (separate == "auto" and ovl >= OVERLAP_MIN_S and len(active) >= 1))
        if separate == "auto" and _overlapping(tracks) and not do_sep:
            log(f"    faces share the screen but only {ovl:.1f}s of speech overlaps: transcribing the mix without separation")
        centres, emb = np.zeros(0), np.zeros((0, 192))
        if use_voices and tracks and not do_sep:      # voice clustering is only the fallback when we don't separate
            def voices():
                from .voices import embed_windows

                log("    embedding voices (ECAPA)")
                t0 = time.time()
                out = embed_windows(wav, words)
                tm["voices"] = time.time() - t0
                return out

            centres, emb = _cached(wd / f"voices.{whisper_model}.pkl" if wd else None, fresh, voices)

        mix = stems = stem_words = None
        if do_sep:
            from .media import read_wav
            from .separate import extract_track, gate_stem, load_model

            mix = read_wav(wav).astype(np.float32) / 32768.0

            def extract():
                log(f"3b/5 separating the voice of {len(active)} speaking face(s) (Dolphin)")
                t0 = time.time()
                model = load_model()
                out = {t.id: extract_track(model, mix, t) for t in tracks if t.id in active}
                tm["separation"] = time.time() - t0
                return out

            stems = _cached(wd / "stems.pkl" if wd else None, fresh, extract)
            missing = [t for t in tracks if t.id in active and t.id not in stems]
            if missing:                                    # a different speaking detector marked more faces as speaking
                log(f"3b/5 separating {len(missing)} more face(s)")
                model = load_model()
                stems = {**stems, **{t.id: extract_track(model, mix, t) for t in missing}}
                if wd:
                    (wd / "stems.pkl").write_bytes(pickle.dumps(stems))
            stems = {i: st for i, st in stems.items() if i in active}

            def stem_tx(st):
                g = gate_stem(st, mix)
                if translate:                                # each voice gets its own language decision
                    r = transcribe_for_captions(g, whisper_model, forced=language, target=translate, fallback=mix_lang)
                    return (r.words, r.source_lang, r.language_prob)
                return transcribe(g, whisper_model, language)

            def stem_asr():
                log("    transcribing each separated voice")
                t0 = time.time()
                out = {i: stem_tx(st) for i, st in stems.items()}
                tm["whisper_stems"] = time.time() - t0
                return out

            sw_path = wd / f"stemwords.{whisper_model}{tr_tag}.pkl" if wd else None
            stem_obj = _cached(sw_path, fresh, stem_asr)
            for i, st in stems.items():
                if i not in stem_obj:
                    stem_obj[i] = stem_tx(st)
                    if wd:
                        sw_path.write_bytes(pickle.dumps(stem_obj))
            stem_obj = {i: o for i, o in stem_obj.items() if i in active}
            if translate:
                stem_words = {i: o[0] for i, o in stem_obj.items()}
                langs.update({i: (o[1], o[2]) for i, o in stem_obj.items()})
                log("    voice languages: " + ", ".join(f"face {i}: {l} ({p:.2f})" for i, (l, p) in langs.items() if i != "mix"))
            else:
                stem_words = stem_obj
            if fuse:
                from .fusion import Fuser, apply_fusion
                from .paths import MODELS_DIR

                n = apply_fusion(tracks, stems, mix, Fuser.load(MODELS_DIR / "fusion.json" if fuse == "on" else fuse))
                log(f"    fused voice energy into the speaking score of {n} face(s)")
    return Analysis(info, tracks, words, centres, emb, mix, stems, stem_words, langs=langs)


def attribute(a: Analysis, log=lambda m: None, fallback: str = "asd"):
    """Stage 4: voices -> labels, then words -> faces. Mutates a.words. Returns (n_voices, voice_tracks)."""
    if a.stems is not None:
        return _attribute_separated(a, log, fallback)
    n_voices, voice_tracks = 0, None
    if len(a.emb) >= 2:
        from .voices import cluster_windows, label_words, match_voices_to_tracks

        labels = cluster_windows(a.emb)
        label_words(a.words, a.centres, labels)
        n_voices = len(set(labels))
        log(f"    {n_voices} distinct voice(s)")
        if n_voices >= 2:
            voice_tracks = match_voices_to_tracks(a.words, a.tracks, smoothed_scores(a.tracks))[0].tracks_of
            log(f"    voice -> face tracks (tie-breaker only): {voice_tracks}")
    attribute_words(a.words, a.tracks, voice_tracks=voice_tracks)
    return n_voices, voice_tracks


RATIO_MIN = 0.2   # a stem word needs at least this fraction of the mix's energy
ASD_MIN = -0.5    # ...and its face must not be clearly "not speaking" (mean smoothed ASD logit over the word)
STEM_MIN = [ASD_MIN]   # the threshold actually used for stem words; a list so experiments can set it (bench --stem-min)


R_ACT = 0.25   # a face's extracted voice must carry this share of the mix to count as 'active' on a word


def _take_clear_mix_words(a: Analysis, accepted: dict):
    """Where exactly one face's voice is active, the plain mix transcript is the better transcript (the extractor
    recovered only ~72% of words even in clear speech, the mix ~92%). Those mix words go to that face and the stem
    words they overlap are dropped. Returns (remaining mix words, count); the taken words are kept on a._clear."""
    from .attribute import smoothed_scores
    from .separate import energy_ratio

    scores = smoothed_scores(a.tracks, asd_only=True)
    by_id = {t.id: t for t in a.tracks}
    clear, rest = [], []
    for w in a.words:
        f0, f1 = int(w.start * 25), max(int(w.end * 25), int(w.start * 25) + 1)
        act = []
        for tid, st in a.stems.items():
            t = by_id[tid]
            lo, hi = max(f0, t.start), min(f1, t.end + 1)
            asd = float(scores[tid][lo - t.start : hi - t.start].mean()) if hi > lo else -9.0
            if energy_ratio(st, a.mix, w.start, w.end) >= R_ACT and asd > ASD_MIN:
                act.append(tid)
        if len(act) == 1:
            w.speaker, w.source = act[0], "mix_clear"
            clear.append(w)
        else:
            rest.append(w)
    for tid in accepted:
        accepted[tid] = [x for x in accepted[tid] if not any(
            min(x.end, c.end) - max(x.start, c.start) > 0.5 * (x.end - x.start) for c in clear)]
    a._clear = clear
    return rest, len(clear)


def _mix_fallback(a: Analysis, accepted: dict, mode: str) -> int:
    """Words only the full mix produced. Default 'off': they are off-screen. 'asd': a face whose speaking score
    stands out gets them. 'asd_if_alone': same, but only where no separated voice is speaking (a lone speaker whose
    extraction failed); inside overlaps they stay off-screen, which is more likely an unextractable voice."""
    if mode == "off":
        return 0
    mix_only = [w for w in a.words if w.source == "mix"]
    if mode == "asd_if_alone":
        stem_spans = sorted((w.start, w.end) for ws in accepted.values() for w in ws)
        alone = lambda w: not any(s < w.end + 0.3 and e > w.start - 0.3 for s, e in stem_spans)
        mix_only = [w for w in mix_only if alone(w)]
    if not mix_only:
        return 0
    attribute_words(mix_only, a.tracks, baseline_from=a.words)
    return sum(w.speaker is not None for w in mix_only)


def _norm(t: str) -> str:
    return "".join(c for c in t.lower() if c.isalnum())


def _dedupe_leakage(accepted: dict, a: Analysis) -> int:
    """The same word heard in two faces' stems at the same time is leakage: keep the copy from the stem that
    carries more of the mix energy there, drop the other."""
    from .separate import energy_ratio

    items = [(w, tid) for tid, ws in accepted.items() for w in ws]
    dead = set()
    for i, (w, tid) in enumerate(items):
        for j in range(i + 1, len(items)):
            v, uid = items[j]
            if uid == tid:
                continue
            if w.translated or v.translated:               # approximate word times: compare midpoints instead of overlap
                same_moment = abs((w.start + w.end) / 2 - (v.start + v.end) / 2) <= 0.6
            else:
                if v.start >= w.end:
                    continue
                ov = min(w.end, v.end) - max(w.start, v.start)
                same_moment = ov > 0.5 * min(w.end - w.start, v.end - v.start)
            if same_moment and _norm(w.text) == _norm(v.text):
                lo, hi = max(w.start, v.start), max(min(w.end, v.end), max(w.start, v.start) + 0.05)
                mine = energy_ratio(a.stems[tid], a.mix, lo, hi)
                theirs = energy_ratio(a.stems[uid], a.mix, lo, hi)
                dead.add(id(v) if mine >= theirs else id(w))
    for tid in accepted:
        accepted[tid] = [w for w in accepted[tid] if id(w) not in dead]
    return len(dead)


def _attribute_separated(a: Analysis, log=lambda m: None, fallback: str = "asd"):
    """Words transcribed from a face's isolated voice belong to that face. Words only the full mix produced are
    off-screen. Stem words that fail the sanity checks are dropped (Whisper hallucinates on residue)."""
    from .attribute import merge_stem_words
    from .separate import energy_ratio

    scores = smoothed_scores(a.tracks)   # fused with voice energy when fusion is on, else the detector's score
    by_id = {t.id: t for t in a.tracks}
    accepted, dropped = {}, 0
    a.dropped = []
    for tid, ws in a.stem_words.items():
        t, keep = by_id[tid], []
        for w in ws:
            f0, f1 = int(w.start * 25), max(int(w.end * 25), int(w.start * 25) + 1)
            lo, hi = max(f0, t.start), min(f1, t.end + 1)
            asd = float(scores[tid][lo - t.start : hi - t.start].mean()) if hi > lo else -9.0
            if energy_ratio(a.stems[tid], a.mix, w.start, w.end) >= RATIO_MIN and asd > STEM_MIN[0]:
                w.speaker, w.source, w.margin = tid, "stem", asd
                keep.append(w)
            else:
                dropped += 1
                a.dropped.append((w, tid, "energy" if energy_ratio(a.stems[tid], a.mix, w.start, w.end) < RATIO_MIN else "asd"))
        accepted[tid] = keep
    dups = _dedupe_leakage(accepted, a)
    n_clear = 0
    if fallback == "hybrid":
        a.words, n_clear = _take_clear_mix_words(a, accepted)
        fallback = "asd_if_alone"
    a.words = merge_stem_words(a.words, accepted)
    a.words = sorted(a.words + getattr(a, "_clear", []), key=lambda w: w.start)
    n_fb = _mix_fallback(a, accepted, fallback)
    log(f"    separated voices: {sum(len(v) for v in accepted.values())} words kept, {dropped} dropped, {dups} leaked duplicates removed; "
        f"{n_clear} clear-speech words taken from the mix, "
        f"{sum(w.source == 'mix' for w in a.words)} mix-only words ({n_fb} given to a face by the speaking scores)")
    return 0, None


def run(src, dst, whisper_model="small", language=None, debug=False, dump_json=True, quiet=False,
        use_voices=True, workdir=None, fresh=False, separate="auto", caption_style="outline", asd_model=None,
        asd_calib="none", fuse=None, translate=None, static_filter=False):
    def log(msg):
        if not quiet:
            print(msg, file=sys.stderr, flush=True)

    if translate and whisper_model in ("tiny", "base", "small"):
        log(f"note: Whisper '{whisper_model}' translates other languages weakly; --whisper-model medium is much better if you have the disk (~1.5 GB)")
    src, dst = Path(src), Path(dst)
    if workdir is None:
        workdir = dst.parent / f"{dst.stem}.work"
    a = analyze(src, workdir, whisper_model, language, use_voices, fresh, log, separate, asd_model=asd_model, asd_calib=asd_calib, fuse=fuse, translate=translate, static_filter=static_filter)
    log(f"{src.name}: {a.info.width}x{a.info.height} {a.info.fps:.2f}fps {a.info.duration:.1f}s, "
        f"{len(a.tracks)} face track(s), {len(a.words)} words")
    log("4/5 attributing words to faces")
    attribute(a, log)
    chunks = make_chunks(a.words)
    if translate:
        retime_even(chunks)                                  # translated word times are approximate: spread them evenly
        assign_hold(chunks)                                  # leave captions up long enough to read
    if dump_json:
        dst.with_suffix(".json").write_text(json.dumps({
            "video": {"width": a.info.width, "height": a.info.height, "fps": a.info.fps, "duration": a.info.duration},
            "tracks": [{"id": t.id, "start": t.start / 25, "end": (t.end + 1) / 25,
                        "mean_score": float(t.score.mean()) if t.score is not None else None} for t in a.tracks],
            "words": [{"start": w.start, "end": w.end, "text": w.text, "speaker": w.speaker,
                       "voice": w.voice, "margin": round(w.margin, 2), "lang": w.lang, "translated": w.translated}
                      for w in a.words],
            "languages": {str(k): {"language": v[0], "prob": round(v[1], 3)} for k, v in (a.langs or {}).items()},
        }, indent=1))
    log("5/5 rendering")
    render(src, a.info, dst, a.tracks, chunks, debug=debug, style=caption_style)
    log(f"done -> {dst}")
