"""Prepare a folder of single-speaker clips for synthetic scene generation.

For every clip we cache: 16 kHz audio, clean-audio reference words (the transcript ground truth), the main face
(median centre / size of the longest face track, used to frame the clip inside a scene tile) and a voice embedding
(used to check which clips are the same person, so train/val/test can be split by person).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from ..asr import transcribe
from ..faces import track_faces
from ..media import extract_audio, probe, read_wav


def prepare(clip_dir: str | Path, ref_dir: str | Path, whisper_model: str = "small", language: str = "en", log=print):
    clip_dir, ref_dir = Path(clip_dir), Path(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(clip_dir.glob("c*.mp4")):
        cid = f.stem
        meta_path = ref_dir / f"{cid}.json"
        if meta_path.exists():
            continue
        info = probe(f)
        wav = ref_dir / f"{cid}.wav"
        extract_audio(f, wav)
        audio = read_wav(wav).astype(np.float32) / 32768.0
        words = transcribe(audio, whisper_model, language)
        tracks = track_faces(f, info)
        main = None
        if tracks:
            t = max(tracks, key=lambda t: len(t.frames) * float(np.median(t.s)))
            main = dict(track=t.id, start=t.start / 25, end=(t.end + 1) / 25,
                        cx=float(np.median(t.cx)), cy=float(np.median(t.cy)), s=float(np.median(t.s)))
        meta = dict(clip=cid, width=info.width, height=info.height, fps=info.fps, duration=info.duration,
                    words=[dict(start=w.start, end=w.end, text=w.text) for w in words], main_face=main)
        meta_path.write_text(json.dumps(meta, indent=1))
        log(f"{cid}: {len(words)} words, main face: {main and (round(main['cx']), round(main['cy']), round(main['s']))}")


def voice_similarity(ref_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """Mean ECAPA embedding of each clip's speech -> cosine similarity matrix (same person ~ high)."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    from ..paths import MODELS_DIR

    ref_dir = Path(ref_dir)
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(MODELS_DIR / "ecapa"),
                                           run_opts={"device": "cpu"})
    ids, embs = [], []
    for p in sorted(ref_dir.glob("c*.json")):
        meta = json.loads(p.read_text())
        audio = read_wav(ref_dir / f"{meta['clip']}.wav").astype(np.float32) / 32768.0
        segs = []
        for w in meta["words"]:
            pass
        # 3 s windows centred on speech, up to 25 per clip
        spans = [(w["start"], w["end"]) for w in meta["words"]]
        starts = np.arange(0, meta["duration"] - 3, 2.0)
        for s in starts:
            if sum(max(0, min(e, s + 3) - max(b, s)) for b, e in spans) > 1.8:
                segs.append(audio[int(s * 16000):int((s + 3) * 16000)])
        segs = segs[:25]
        with torch.no_grad():
            e = model.encode_batch(torch.from_numpy(np.stack(segs))).squeeze(1).numpy()
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        m = e.mean(0)
        ids.append(meta["clip"]); embs.append(m / np.linalg.norm(m))
    E = np.stack(embs)
    return ids, E @ E.T
