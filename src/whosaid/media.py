"""ffmpeg / OpenCV helpers."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASD_FPS = 25  # LR-ASD expects 25 fps video and 100 Hz MFCC frames
SAMPLE_RATE = 16000


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = v.get("avg_frame_rate", "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 25.0
    # honour rotation metadata so geometry matches what OpenCV/ffmpeg decode
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        rot = int(sd.get("rotation", rot))
    rot = int(v.get("tags", {}).get("rotate", rot))
    if abs(rot) % 180 == 90:
        w, h = h, w
    dur = float(info["format"].get("duration") or v.get("duration") or 0)
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    return VideoInfo(w, h, fps or 25.0, dur, has_audio)


def extract_audio(path: str | Path, wav_path: str | Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav_path)],
        check=True,
    )
    return Path(wav_path)


def read_wav(path: str | Path) -> np.ndarray:
    from scipy.io import wavfile

    _, audio = wavfile.read(str(path))
    return audio


def iter_frames(path: str | Path, info: VideoInfo, fps: float = ASD_FPS, out_width: int | None = None):
    """Yield BGR uint8 frames resampled to `fps` (optionally downscaled to `out_width`) via an ffmpeg pipe."""
    w, h = info.width, info.height
    vf = f"fps={fps}"
    if out_width and out_width < w:
        nw = out_width - out_width % 2
        nh = int(round(h * nw / w / 2)) * 2
        vf += f",scale={nw}:{nh}"
        w, h = nw, nh
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-an", "-vf", vf,
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
