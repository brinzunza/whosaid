import numpy as np

from whosaid.calibrate import (G_CAN, SHIFT_MAX, T_CAN, calibrate_tracks, face_shift, to_canonical, youden_threshold)
from whosaid.faces import Track


def bimodal(rng, lo, hi, p_speak=0.5, n=800, sd=0.5):
    blocks, i = [], 0
    out = np.zeros(n)
    speaking = False
    while i < n:                                   # alternating runs, like real turns
        ln = int(rng.integers(40, 120))
        out[i:i + ln] = rng.normal(hi if speaking else lo, sd, min(ln, n - i))
        i += ln
        speaking = rng.random() < p_speak
    return out


def trk(score):
    n = len(score)
    return Track(0, np.arange(n), np.zeros(n), np.zeros(n), np.ones(n), np.ones(n, bool), score=np.asarray(score, float))


def test_model_level_mapping_puts_threshold_at_one_and_gap_at_three():
    meta = dict(t0=-0.4, gap=1.5)                   # a detector whose scores live on a smaller, shifted scale
    s = to_canonical(np.array([-0.4, 0.35]), meta)
    assert abs(s[0] - T_CAN) < 1e-9 and abs((s[1] - s[0]) - G_CAN / 2) < 1e-9


def test_youden_finds_the_valley():
    rng = np.random.default_rng(0)
    t = youden_threshold(rng.normal(3, 0.6, 2000), rng.normal(-1, 0.6, 2000))
    assert 0.6 < t < 1.4


def test_face_shift_moves_threshold_to_the_valley_within_limits():
    rng = np.random.default_rng(1)
    sh, info = face_shift(bimodal(rng, -1.0, 3.0))              # valley at 1.0 = already canonical
    assert info["bimodal"] and abs(sh) < 0.3
    sh, _ = face_shift(bimodal(rng, -2.0, 0.6))                 # valley at -0.7: shift by -1.7, clipped
    assert sh == -SHIFT_MAX
    sh, _ = face_shift(bimodal(rng, 0.0, 2.6))                  # valley 1.3 -> small positive shift
    assert 0.0 < sh < 0.7


def test_unimodal_and_short_faces_are_left_alone():
    rng = np.random.default_rng(2)
    assert face_shift(rng.normal(3.0, 0.4, 800))[0] == 0.0       # always speaking
    assert face_shift(rng.normal(-1.0, 0.4, 800))[0] == 0.0      # never speaking
    assert face_shift(bimodal(rng, -1, 3, n=60))[0] == 0.0       # too little data
    s = rng.normal(0, 0.5, 800); s[:30] += 3                      # a tiny speaking minority is not a mode
    assert face_shift(s)[0] == 0.0


def test_calibrate_tracks_modes_and_idempotence():
    rng = np.random.default_rng(3)
    raw = bimodal(rng, -2.0, 0.6)
    t = trk(raw)
    meta = dict(t0=1.0, gap=G_CAN)               # identity mapping
    calibrate_tracks([t], meta, "none")
    assert np.array_equal(t.score, raw) and t.score_raw is None
    calibrate_tracks([t], meta, "model")
    assert np.allclose(t.score, raw) and np.array_equal(t.score_raw, raw)
    calibrate_tracks([t], meta, "face")                          # recomputed from the raw scores, not stacked
    once = t.score.copy()
    calibrate_tracks([t], meta, "face")
    assert np.allclose(t.score, once) and (t.score > raw).all() is np.True_


def test_recall_target_places_threshold_at_the_right_quantile():
    from whosaid.calibrate import with_recall

    rng = np.random.default_rng(5)
    pos = rng.normal(2.0, 1.0, 5000)
    meta = dict(t0=0.5, gap=3.0, pos_q=np.percentile(pos, np.arange(101)).tolist())
    m90 = with_recall(meta, 0.90)
    assert abs((pos > m90["t0"]).mean() - 0.90) < 0.02
    assert with_recall(meta, 0.95)["t0"] < m90["t0"]          # asking for more recall lowers the threshold
