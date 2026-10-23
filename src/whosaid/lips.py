"""Aligned mouth crops in the format Dolphin / LRS3-style lip models expect (88x88 grayscale at 25 fps).

The reference pipeline aligns 68-point landmarks to a mean face at 256x256 and crops 96x96 around the mouth.
We only have YuNet's 5 points (eyes, nose tip, mouth corners), so we fit a similarity transform from those to the
corresponding points of the same mean face and crop the same region.
"""
from __future__ import annotations

import numpy as np

from .paths import MODELS_DIR

STD = 256
LIP = 88
# mean-face targets: eye centres, nose tip (30), mouth corners (48, 54), and the mouth-region centre (48:68)
_MEAN = None


def _mean_face():
    global _MEAN
    if _MEAN is None:
        m = np.load(MODELS_DIR / "20words_mean_face.npy")
        pts = np.array([m[36:42].mean(0), m[42:48].mean(0), m[30], m[48], m[54]], np.float32)
        _MEAN = (pts, m[48:68].mean(0))
    return _MEAN


def lip_crop(gray: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """gray: full frame; lm: (5, 2) YuNet landmarks in frame pixels -> (88, 88) uint8 mouth crop."""
    import cv2

    pts, mouth = _mean_face()
    M, _ = cv2.estimateAffinePartial2D(lm.astype(np.float32), pts, method=cv2.LMEDS)
    if M is None:
        return np.zeros((LIP, LIP), np.uint8)
    M = M.copy()
    # fold the crop (centre on mouth, size LIP) into the warp so only 88x88 pixels are produced
    M[:, 2] -= mouth - LIP / 2
    return cv2.warpAffine(gray, M, (LIP, LIP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
