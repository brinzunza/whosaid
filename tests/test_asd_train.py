import numpy as np

from whosaid.asd_train import T_WIN, _auc, _sample


class FakeScene:
    def __init__(self, n=300, start=10):
        self.mfcc = np.arange((start + n) * 4 * 13, dtype=np.float32).reshape(-1, 13)   # row r holds the value r*13..
        y = (np.arange(n) % 50 < 25).astype(np.uint8)
        crops = np.zeros((n, 112, 112), np.uint8)
        crops[:, 0, 0] = np.arange(n) % 251                      # frame index marker
        self.tracks = {0: dict(crops=crops, start=start, y=y, cat=np.zeros(n, np.uint8))}


def test_sample_aligns_audio_video_and_labels():
    s = FakeScene()
    rng = np.random.default_rng(0)
    for _ in range(20):
        aud, vid, y, w = _sample([s], rng, train=False)
        assert aud.shape == (T_WIN * 4, 13) and vid.shape == (T_WIN, 112, 112) and y.shape == (T_WIN,)
        i = int(vid[0, 0, 0])                                    # first video frame within the track
        assert aud[0, 0] == (10 + i) * 4 * 13                    # audio starts at the same instant (4 mfcc frames / video frame)
        assert np.array_equal(y, (np.arange(i, i + T_WIN) % 50 < 25).astype(np.float32))


def test_sample_skips_tracks_shorter_than_a_window():
    s = FakeScene(n=T_WIN - 1)
    assert _sample([s], np.random.default_rng(0), True) is None


def test_auc_basics():
    assert _auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert _auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert np.isnan(_auc(np.array([]), np.array([1.0])))


def test_shift_augmentation_labels_everything_negative_and_misaligns_audio():
    s = FakeScene(n=240)   # keep n < 251: the frame-index marker wraps at 251
    s.mfcc[:, 0] = np.abs(np.sin(np.arange(len(s.mfcc)) / 50)) + 1.0        # plenty of 'speech' energy everywhere
    rng = np.random.default_rng(1)
    seen = 0
    for _ in range(60):
        aud, vid, y, w = _sample([s], rng, train=False, shift_p=1.0)
        i = int(vid[0, 0, 0])
        if aud[0, 1] != (10 + i) * 4 * 13 + 1:                              # audio taken from elsewhere (column 1 is the position marker)
            seen += 1
            assert y.sum() == 0 and (w >= 1).all()
    assert seen > 40


def test_hard_sampling_prefers_mouthing_windows():
    from whosaid.asd_train import _prep_hard

    s = FakeScene(n=600)
    cat = np.full(600, 3, np.uint8)
    cat[300:450] = 1
    s.tracks[0]["cat"] = cat
    _prep_hard([s])
    rng = np.random.default_rng(0)
    fr = []
    for _ in range(40):
        _, vid, _, w = _sample([s], rng, train=False, hard_p=1.0)
        fr.append((w == 3.0).mean())
    assert np.mean(fr) >= 0.3                                               # weight 3 marks mouthing frames
