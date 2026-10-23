import numpy as np

from whosaid.faces import Track
from whosaid.fusion import ALL_FEATURES, Fuser, apply_fusion, fit, frame_ratio, scene_features

SR = 16000


def trk(i, start, n, score):
    return Track(i, np.arange(start, start + n), np.zeros(n), np.zeros(n), np.ones(n), np.ones(n, bool), score=np.asarray(score, float))


def test_frame_ratio_is_the_rms_share_of_the_mix():
    mix = np.ones(SR * 4, np.float32)
    stem = np.zeros_like(mix); stem[SR * 2:] = 0.5            # the face's voice is present only from t = 2 s
    r = frame_ratio(trk(0, 0, 100, np.zeros(100)), stem, mix)
    assert r[10] < 0.01 and abs(r[80] - 0.5) < 0.01          # frame 10 = 0.4 s, frame 80 = 3.2 s


def test_scene_features_share_and_other_voice():
    mix = np.ones(SR * 2, np.float32)
    s0 = np.full_like(mix, 0.6); s1 = np.full_like(mix, 0.2)
    t0, t1 = trk(0, 0, 50, np.full(50, 2.0)), trk(1, 0, 50, np.full(50, 2.0))
    f = scene_features([t0, t1], {0: s0, 1: s1}, {0: t0.score, 1: t1.score}, mix)
    assert f[0].shape == (50, len(ALL_FEATURES))
    share = ALL_FEATURES.index("share")
    assert abs(f[0][20, share] - 0.75) < 0.02 and abs(f[1][20, share] - 0.25) < 0.02
    lr_other = ALL_FEATURES.index("lr_other")
    assert f[1][20, lr_other] > f[0][20, lr_other]            # face 1's strongest competitor (face 0) is louder


def test_fit_learns_that_a_loud_own_voice_means_speaking_and_a_mouthing_face_does_not():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    asd = np.where(y == 1, rng.normal(2.5, 1.0, n), rng.normal(0.0, 1.0, n))
    mouthing_frames = (y == 0) & (rng.random(n) < 0.3)             # the detector fires on 30% of non-speaking frames
    asd[mouthing_frames] = rng.normal(2.6, 0.8, mouthing_frames.sum())
    lr = np.where(y == 1, rng.normal(-0.7, 0.4, n), rng.normal(-4.0, 0.8, n))
    X = np.stack([asd, lr, asd * lr, np.zeros(n), np.zeros(n)], 1)
    f = fit(X, y, ["asd", "lr", "asd_x_lr"])
    mouthing = np.array([[2.6, -4.0, 2.6 * -4.0, 0, 0]]); speaking = np.array([[2.6, -0.7, 2.6 * -0.7, 0, 0]])
    assert f.score(speaking)[0] > 1.0 > f.score(mouthing)[0]  # same detector score, opposite verdict


def test_apply_fusion_replaces_scores_only_for_separated_faces_and_roundtrips(tmp_path):
    mix = np.ones(SR * 2, np.float32)
    t0, t1 = trk(0, 0, 50, np.full(50, 2.0)), trk(1, 0, 50, np.full(50, 2.0))
    f = Fuser(["asd", "lr"], [0.5, 1.0], 0.0, dict(t0=0.0, gap=1.8))
    f.save(tmp_path / "f.json")
    g = Fuser.load(tmp_path / "f.json")
    assert apply_fusion([t0, t1], {0: np.full_like(mix, 0.5)}, mix, g) == 1
    assert np.allclose(t0.score_asd, 2.0) and not np.allclose(t0.score, 2.0) and np.allclose(t1.score, 2.0)
