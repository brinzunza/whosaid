"""Face-conditioned speech extraction with Dolphin: mixture + one face's lip crops -> that person's voice.

Because the network is told *whose* mouth to follow, there is no output-permutation problem (the usual trouble
with blind separators), and it can also be run for a face that is silent (the output should then be near-silent).
"""
from __future__ import annotations

import numpy as np

from .faces import Track
from .media import ASD_FPS, SAMPLE_RATE

WIN_S = 2.0   # CPU cost grows faster than linearly with window length (a 4 s window cost ~2.4x two 2 s windows)
HOP_S = 1.6   # 0.4 s overlap, cross-faded with a Hann window
LIP_MEAN, LIP_STD = 0.421, 0.165


def load_model(device: str = "cpu"):
    import torch

    from .dolphin.dolphin import Dolphin

    m = Dolphin.from_pretrained("JusperLee/Dolphin").eval().to(device)
    torch.set_grad_enabled(False)
    return m


def extract_track(model, mix: np.ndarray, track: Track, device: str = "cpu", progress=None) -> np.ndarray:
    """Return a mono float32 waveform (same length as mix) holding this face's voice; zeros outside its time span."""
    import torch

    out = np.zeros(len(mix), np.float32)
    weight = np.zeros(len(mix), np.float32)
    t0, t1 = track.start / ASD_FPS, (track.end + 1) / ASD_FPS
    win, hop = int(WIN_S * SAMPLE_RATE), int(HOP_S * SAMPLE_RATE)
    first = int(t0 * SAMPLE_RATE)
    last = min(len(mix), int(t1 * SAMPLE_RATE))
    starts = list(range(first, max(last - win, first) + 1, hop))
    if starts and starts[-1] + win < last:
        starts.append(last - win)
    for k, s in enumerate(starts):
        e = min(s + win, len(mix))
        seg = mix[s:e]
        f0 = int(round(s / SAMPLE_RATE * ASD_FPS)) - track.start
        n_fr = int(round((e - s) / SAMPLE_RATE * ASD_FPS))
        idx = np.clip(np.arange(f0, f0 + n_fr), 0, len(track.lips) - 1)
        lips = (track.lips[idx].astype(np.float32) / 255.0 - LIP_MEAN) / LIP_STD
        est = model(
            torch.from_numpy(seg)[None].to(device), torch.from_numpy(lips)[None, None].to(device)
        )[0, 0].cpu().numpy()
        w = np.hanning(len(seg) + 2)[1:-1].astype(np.float32)
        out[s:e] += est[: len(seg)] * w
        weight[s:e] += w
        if progress:
            progress(k + 1, len(starts))
    nz = weight > 1e-6
    out[nz] /= weight[nz]
    return out


def gate_stem(stem: np.ndarray, mix: np.ndarray, floor: float = 0.12, frame_s: float = 0.1) -> np.ndarray:
    """Silence stretches where the extracted voice is a tiny fraction of the mix (the face is not speaking).
    Whisper hallucinates words on near-silent residue, so remove it before transcription."""
    n = int(frame_s * SAMPLE_RATE)
    k = len(stem) // n
    r = np.sqrt((stem[: k * n].reshape(k, n) ** 2).mean(1)) / (np.sqrt((mix[: k * n].reshape(k, n) ** 2).mean(1)) + 1e-6)
    r = np.convolve(r, np.ones(3) / 3, mode="same")  # 0.3 s smoothing
    keep = np.repeat(r >= floor, n)
    out = stem.copy()
    out[: len(keep)] *= keep
    out[len(keep):] = 0
    return out


def energy_ratio(stem: np.ndarray, mix: np.ndarray, start: float, end: float) -> float:
    a, b = int(start * SAMPLE_RATE), max(int(end * SAMPLE_RATE), int(start * SAMPLE_RATE) + 160)
    return float(np.sqrt(np.mean(stem[a:b] ** 2)) / (np.sqrt(np.mean(mix[a:b] ** 2)) + 1e-6))
