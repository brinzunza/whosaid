"""Compose synthetic multi-person scenes from single-speaker clips, with exact ground truth.

Each person's tile always plays *their own* video with *their own* audio (so lips and voice stay in sync); only the
schedule changes: who speaks when, how much they overlap, who is shown, who is heard but not seen.

Scene types
  single     one visible person, no overlap (control)
  turns      people take turns, with some interruptions (negative gaps) -> modest overlap
  overlap    everyone talks at once for the whole scene (stress test)
  offscreen  visible people plus an audio-only voice from a clip that is not shown
Listeners (a tile whose person is not talking) either `freeze` on a frame, or `mouthing`: their own video keeps
playing with the audio muted, a hard negative for any lip-based detector.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..disk import ensure_free
from ..media import SAMPLE_RATE

FPS = 25
W, H = 1280, 720


@dataclass
class Clip:
    id: str
    person: str
    path: Path
    meta: dict
    audio: np.ndarray
    audio_only: bool = False   # never shown on screen (e.g. scenery while talking); usable as an off-screen voice

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    @property
    def words(self) -> list[dict]:
        return self.meta["words"]


@dataclass
class Tile:
    id: str
    clip: Clip
    rect: tuple[int, int, int, int] | None          # (x, y, w, h) on the canvas; None = off-screen (audio only)
    windows: list[tuple[float, float, float]] = field(default_factory=list)  # (scene_t0, scene_t1, src_t0) speaking
    gain_db: float = 0.0


def load_library(ref_dir, clip_dir, persons: dict[str, str], audio_only=()) -> dict[str, Clip]:
    from ..media import read_wav

    out = {}
    for p in sorted(Path(ref_dir).glob("c*.json")):
        meta = json.loads(p.read_text())
        cid = meta["clip"]
        audio = read_wav(Path(ref_dir) / f"{cid}.wav").astype(np.float32) / 32768.0
        out[cid] = Clip(cid, persons.get(cid, cid), Path(clip_dir) / f"{cid}.mp4", meta, audio, cid in set(audio_only))
    return out


def utterances(clip: Clip, gap: float = 0.45, min_dur: float = 1.4, max_dur: float = 9.0) -> list[tuple[float, float]]:
    """Split a clip's reference words into utterances (pause > gap, or sentence end after a few words)."""
    out, cur = [], []
    for w in clip.words:
        if cur and (w["start"] - cur[-1]["end"] > gap or w["end"] - cur[0]["start"] > max_dur
                    or (len(cur) >= 4 and cur[-1]["text"][-1] in ".?!")):
            out.append((cur[0]["start"], cur[-1]["end"]))
            cur = []
        cur.append(w)
    if cur:
        out.append((cur[0]["start"], cur[-1]["end"]))
    return [(a, b) for a, b in out if b - a >= min_dur]


# ------------------------------------------------------------------ planning

def _layout(n: int) -> list[tuple[int, int, int, int]]:
    if n == 1:
        return [(0, 0, W, H)]
    if n == 2:
        return [(0, 0, W // 2, H), (W // 2, 0, W // 2, H)]
    w3 = W // 3
    return [(0, 0, w3, H), (w3, 0, w3, H), (2 * w3, 0, W - 2 * w3, H)]


def plan_overlap(rng, tiles: list[Tile], D: float) -> None:
    for t in tiles:
        src = rng.uniform(0, max(t.clip.duration - D, 0.0))
        t.windows = [(0.0, D, float(src))]


def plan_turns(rng, tiles: list[Tile], D: float) -> None:
    used = {t.id: [] for t in tiles}
    utts = {t.id: utterances(t.clip) for t in tiles}
    cursor, last = 0.0, None
    order = [t for t in tiles]
    while cursor < D - 1.5:
        cands = [t for t in order if t.id != (last.id if last else None)] or order
        t = cands[rng.integers(len(cands))]
        pool = utts[t.id]
        if not pool:
            break
        for _ in range(20):                     # one to three consecutive utterances, source range not reused
            i = int(rng.integers(len(pool)))
            k = int(rng.integers(1, 4))
            a, b = pool[i][0], pool[min(i + k - 1, len(pool) - 1)][1]
            if b - a > 9.0:
                b = pool[i][1]
            if all(b + 0.3 < u0 or a - 0.3 > u1 for u0, u1 in used[t.id]):
                break
        else:
            cursor += 1.0
            continue
        used[t.id].append((a, b))
        s0, s1 = max(a - 0.12, 0.0), min(b + 0.18, t.clip.duration)
        t0 = cursor
        t1 = min(t0 + (s1 - s0), D)
        t.windows.append((t0, t1, s0))
        cursor = t1 + float(rng.uniform(-1.4, 0.9))   # negative = interruption (overlap)
        cursor = max(cursor, t0 + 1.0)
        last = t


def build_scene(rng, lib: dict[str, Clip], split_persons: list[str], kind: str, D: float, listener: str):
    by_person: dict[str, list[Clip]] = {}
    for c in lib.values():
        if c.person in split_persons:
            by_person.setdefault(c.person, []).append(c)
    people = list(by_person)
    n_vis = {"single": 1, "turns": int(rng.integers(2, 4)), "overlap": int(rng.integers(2, 4)),
             "offscreen": int(rng.integers(1, 3))}[kind]
    need = n_vis + (1 if kind == "offscreen" else 0)
    if len(people) < need:
        return None
    chosen = [people[i] for i in rng.permutation(len(people))[:need]]
    tiles: list[Tile] = []
    rects = _layout(n_vis)
    for k, p in enumerate(chosen[:n_vis]):
        cands = [c for c in by_person[p] if not c.audio_only and c.meta.get("main_face") and c.meta["main_face"]["end"] - c.meta["main_face"]["start"] > 15]
        if not cands:
            return None
        tiles.append(Tile("ABC"[k], cands[rng.integers(len(cands))], rects[k]))
    if kind == "offscreen":
        pool = by_person[chosen[-1]]
        tiles.append(Tile("X", pool[rng.integers(len(pool))], None))
    for t in tiles:
        t.gain_db = float(rng.uniform(-4, 4))
    if kind == "overlap":
        D = min(D, min(t.clip.duration for t in tiles) - 0.5)
        plan_overlap(rng, tiles, D)
    elif kind == "offscreen":
        plan_turns(rng, tiles, D) if rng.random() < 0.7 else plan_overlap(rng, tiles, min(D, min(t.clip.duration for t in tiles) - 0.5))
        D = max(max((w[1] for w in t.windows), default=0.0) for t in tiles)
    elif kind == "turns":
        plan_turns(rng, tiles, D)
    else:
        plan_turns(rng, tiles[:1], D)
    D = float(min(D, max(w[1] for t in tiles for w in t.windows)))
    return tiles, D


# ------------------------------------------------------------------ video

def _framing(rng, clip: Clip, tw: int, th: int):
    mf = clip.meta["main_face"]
    cw, ch = clip.meta["width"], clip.meta["height"]
    s_target = th * float(rng.uniform(0.075, 0.16)) * (min(tw / th, 1.0) ** 0.5)
    k = max(s_target / mf["s"], tw / cw, th / ch)
    sw, sh = int(np.ceil(cw * k / 2) * 2), int(np.ceil(ch * k / 2) * 2)
    x = int(np.clip(mf["cx"] * k - tw / 2 + rng.uniform(-0.05, 0.05) * tw, 0, sw - tw))
    y = int(np.clip(mf["cy"] * k - th * rng.uniform(0.35, 0.5), 0, sh - th))
    return f"scale={sw}:{sh},crop={tw}:{th}:{x}:{y}"


def _decode(path, src_t0: float, n: int, vf: str, tw: int, th: int):
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-ss", f"{src_t0:.3f}", "-i", str(path), "-an",
         "-vf", f"fps={FPS},{vf}", "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE)
    size = tw * th * 3
    last = None
    try:
        for _ in range(n):
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            last = np.frombuffer(buf, np.uint8).reshape(th, tw, 3)
            yield last
    finally:
        proc.stdout.close(); proc.kill(); proc.wait()
    while last is not None:   # source ended early: hold the last frame
        yield last


def _tile_timeline(t: Tile, D: float, listener: str):
    """(n_frames, src_t0 | None, playing) runs covering [0, D): speaking windows are in sync; gaps freeze or keep mouthing."""
    wins = sorted(t.windows)
    if not wins:   # a silent participant who never takes a turn: listens for the whole scene
        return [(D, 0.0, listener == "mouthing")]
    runs, cur = [], 0.0
    prev_src_end = wins[0][2]
    for (a, b, s) in wins:
        if a > cur + 1e-6:
            if listener == "mouthing":
                runs.append((a - cur, prev_src_end, True))     # source keeps playing, audio muted
            else:
                runs.append((a - cur, prev_src_end, False))
        runs.append((b - a, s, True))
        prev_src_end, cur = s + (b - a), b
    if cur < D - 1e-6:
        runs.append((D - cur, prev_src_end, listener == "mouthing"))
    return runs


def _tile_frames(t: Tile, D: float, listener: str, rng):
    tw, th = t.rect[2], t.rect[3]
    vf = _framing(rng, t.clip, tw, th)
    total = int(round(D * FPS))
    emitted = 0
    for dur, src, playing in _tile_timeline(t, D, listener):
        n = int(round(dur * FPS))
        if n <= 0:
            continue
        src = min(src, max(t.clip.duration - n / FPS - 0.05, 0.0))
        if playing:
            for fr in _decode(t.clip.path, src, n, vf, tw, th):
                if emitted >= total:
                    return
                yield fr; emitted += 1
                n -= 1
                if n == 0:
                    break
        else:
            fr = next(_decode(t.clip.path, src, 1, vf, tw, th))
            for _ in range(n):
                if emitted >= total:
                    return
                yield fr; emitted += 1
    last = np.zeros((th, tw, 3), np.uint8)
    while emitted < total:
        yield last; emitted += 1


# ------------------------------------------------------------------ audio

def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def _synthetic_rir(rng, rt60: float) -> np.ndarray:
    n = int(rt60 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 4.0 * np.abs(ir[1:50]).max()    # strong direct path
    return (ir / np.linalg.norm(ir)).astype(np.float32)


def build_audio(rng, tiles: list[Tile], D: float, reverb: float | None, snr_db: float | None):
    n = int(D * SAMPLE_RATE)
    stems = {}
    for t in tiles:
        s = np.zeros(n, np.float32)
        for (a, b, src) in t.windows:
            i0, ln = int(a * SAMPLE_RATE), int((b - a) * SAMPLE_RATE)
            seg = t.clip.audio[int(src * SAMPLE_RATE): int(src * SAMPLE_RATE) + ln].copy()
            if len(seg) == 0:
                continue
            f = min(len(seg), int(0.03 * SAMPLE_RATE))
            seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
            s[i0:i0 + len(seg)] += seg[: n - i0]
        stems[t.id] = s
    target = 10 ** (-24 / 20)
    for t in tiles:
        active = np.abs(stems[t.id]) > 0.01
        base = _rms(stems[t.id][active]) if active.sum() > 1600 else _rms(stems[t.id])
        stems[t.id] *= target / base * 10 ** (t.gain_db / 20)
    mix_in = {k: v.copy() for k, v in stems.items()}
    if reverb:
        from scipy.signal import fftconvolve

        rir = _synthetic_rir(rng, reverb)
        mix_in = {k: fftconvolve(v, rir)[:n].astype(np.float32) for k, v in mix_in.items()}
    mix = sum(mix_in.values())
    if snr_db is not None:
        noise = rng.standard_normal(n).astype(np.float32)
        noise = np.cumsum(noise); noise -= np.convolve(noise, np.ones(800) / 800, mode="same")  # brown-ish, DC removed
        noise *= _rms(mix) / _rms(noise) * 10 ** (-snr_db / 20)
        mix = mix + noise
    peak = np.abs(mix).max()
    if peak > 0.95:
        mix *= 0.95 / peak
        stems = {k: v * 0.95 / peak for k, v in stems.items()}
    return mix.astype(np.float32), stems


# ------------------------------------------------------------------ labels + writing

def _labels(tiles: list[Tile], D: float, meta: dict) -> dict:
    words, speech = [], {}
    for t in tiles:
        spans = []
        for (a, b, src) in t.windows:
            for w in t.clip.words:
                if w["start"] >= src - 0.02 and w["end"] <= src + (b - a) + 0.02:
                    words.append(dict(start=round(w["start"] - src + a, 3), end=round(w["end"] - src + a, 3),
                                      text=w["text"], tile=t.id, clip=t.clip.id, person=t.clip.person))
                    spans.append([round(w["start"] - src + a, 3), round(w["end"] - src + a, 3)])
        merged = []
        for s, e in sorted(spans):
            if merged and s - merged[-1][1] < 0.3:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        speech[t.id] = merged
    words.sort(key=lambda w: w["start"])
    overlap, ids = [], list(speech)
    for i, p in enumerate(ids):
        for q in ids[i + 1:]:
            for a, b in speech[p]:
                for c, d in speech[q]:
                    lo, hi = max(a, c), min(b, d)
                    if hi - lo > 0.2:
                        overlap.append([round(lo, 3), round(hi, 3)])
    overlap.sort()
    return dict(meta, duration=round(D, 3), fps=FPS, size=[W, H], overlap=overlap, speech=speech, words=words,
                tiles=[dict(id=t.id, clip=t.clip.id, person=t.clip.person, rect=list(t.rect) if t.rect else None,
                            on_screen=t.rect is not None, gain_db=round(t.gain_db, 2),
                            windows=[[round(a, 3), round(b, 3), round(s, 3)] for a, b, s in t.windows]) for t in tiles])


def write_scene(out_dir: Path, name: str, tiles: list[Tile], D: float, listener: str, rng, meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    reverb = float(rng.uniform(0.2, 0.7)) if rng.random() < 0.4 else None
    snr = float(rng.uniform(15, 35)) if rng.random() < 0.5 else None
    mix, stems = build_audio(rng, tiles, D, reverb, snr)
    with tempfile.TemporaryDirectory() as tmp:
        from scipy.io import wavfile

        wav = Path(tmp) / "mix.wav"
        wavfile.write(wav, SAMPLE_RATE, (mix * 32767).astype(np.int16))
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest", str(out_dir / f"{name}.mp4")],
            stdin=subprocess.PIPE)
        vis = [t for t in tiles if t.rect]
        gens = [_tile_frames(t, D, listener, rng) for t in vis]
        total = int(round(D * FPS))
        canvas = np.full((H, W, 3), 28, np.uint8)
        for _ in range(total):
            for t, g in zip(vis, gens):
                x, y, w, h = t.rect
                canvas[y:y + h, x:x + w] = next(g)
            enc.stdin.write(canvas.tobytes())
        enc.stdin.close(); enc.wait()
    np.savez_compressed(out_dir / f"{name}.stems.npz", **{k: v.astype(np.float16) for k, v in stems.items()})
    labels = _labels(tiles, D, dict(scene=name, listener=listener, reverb_rt60=reverb, noise_snr_db=snr, **meta))
    (out_dir / f"{name}.labels.json").write_text(json.dumps(labels, indent=1))
    return labels


def generate(out_dir, lib: dict[str, Clip], splits: dict[str, list[str]], n: int, seed: int = 0,
             mix: dict[str, float] | None = None, log=print, mouthing_p: float = 0.3) -> list[str]:
    mix = mix or {"single": 0.1, "turns": 0.4, "overlap": 0.2, "offscreen": 0.3}
    kinds, probs = list(mix), np.array(list(mix.values())) / sum(mix.values())
    names = []
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        kind = str(rng.choice(kinds, p=probs))
        split = list(splits)[i % len(splits)]
        listener = "mouthing" if rng.random() < mouthing_p else "freeze"
        ensure_free(f"scene {i}")
        D = float(rng.uniform(24, 40))
        built = build_scene(rng, lib, splits[split], kind, D, listener)
        if built is None:
            continue
        tiles, D = built
        name = f"{split}_{kind}_{seed:02d}_{i:03d}"
        write_scene(Path(out_dir) / split, name, tiles, D, listener, rng, dict(kind=kind, split=split, seed=seed))
        log(f"{name}: {D:.1f}s, tiles={[(t.id, t.clip.id) for t in tiles]}, listener={listener}")
        names.append(name)
    return names
