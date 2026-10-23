import numpy as np

from whosaid.faces import Track
from whosaid.photos import LIP_P90_MAX, MIN_FRAMES, is_static, mark_static


def track(lips, move=0.0, n=None, score=2.0, tid=0):
    n = n or len(lips)
    rng = np.random.default_rng(0)
    cx = 300 + move * 100 * rng.standard_normal(n)
    return Track(tid, np.arange(n), cx, np.full(n, 300.0), np.full(n, 100.0), np.ones(n, bool),
                 score=np.full(n, score), lips=lips)


def crops(n, change):
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (88, 88)).astype(np.float32)
    return np.stack([np.clip(base + change * rng.standard_normal((88, 88)), 0, 255) for _ in range(n)]).astype(np.uint8)


def test_photo_has_no_mouth_change_and_no_movement():
    assert is_static(track(crops(200, 0.0)))


def test_person_who_speaks_is_never_static():
    c = crops(200, 0.0)
    c[60:140] = crops(80, 12.0)                         # speaks for 3 of 8 seconds: p90 still sees it
    assert not is_static(track(c, move=0.05))


def test_person_who_moves_around_is_not_static_even_with_a_still_mouth():
    assert not is_static(track(crops(200, 0.0), move=0.4))


def test_short_tracks_and_tracks_without_crops_are_not_judged():
    assert not is_static(track(crops(MIN_FRAMES - 1, 0.0)))
    t = track(crops(100, 0.0)); t.lips = None
    assert not is_static(t)


def test_mark_static_silences_only_the_static_faces():
    ph, person = track(crops(200, 0.0), tid=0), track(crops(200, 15.0), move=0.2, tid=1)
    ids = mark_static([ph, person])
    assert ids == [0] and ph.static and (ph.score == -5.0).all() and np.allclose(ph.score_raw, 2.0)
    assert not person.static and np.allclose(person.score, 2.0)


def test_overlap_test_ignores_static_faces():
    from whosaid.pipeline import _overlapping

    ph, person = track(crops(200, 0.0), tid=0), track(crops(200, 15.0), move=0.2, tid=1)
    assert _overlapping([ph, person])                    # before marking: two faces at once
    mark_static([ph, person])
    assert not _overlapping([ph, person])                # after: a photo is not a second speaker
