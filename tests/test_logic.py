import numpy as np

from whosaid.asr import Word
from whosaid.attribute import attribute_words, make_chunks
from whosaid.faces import Track
from whosaid.layout import SLIDE_FRAMES, anchor_at, exit_edge, place_box, resolve_collisions


def mk_track(id, start, end, cx, cy=300.0, s=50.0, score=0.0):
    n = end - start + 1
    return Track(id, np.arange(start, end + 1), np.full(n, cx, float) if np.isscalar(cx) else np.asarray(cx, float),
                 np.full(n, cy), np.full(n, s), np.ones(n, bool), np.full(n, score, float))


def test_words_go_to_the_speaking_face():
    a, b = mk_track(0, 0, 250, 200, score=-3), mk_track(1, 0, 250, 800, score=-3)
    a.score[25:75] = 4    # A speaks 1s-3s
    b.score[100:150] = 4  # B speaks 4s-6s
    words = [Word(1.2, 1.6, "hello"), Word(4.2, 4.6, "there"), Word(8.0, 8.4, "nobody")]
    attribute_words(words, [a, b])
    assert [w.speaker for w in words] == [0, 1, None]


def test_speaker_who_left_frame_keeps_their_words():
    a = mk_track(0, 0, 50, 100, score=3)  # visible 0-2s, then gone
    words = [Word(1.0, 1.5, "bye"), Word(2.4, 2.8, "now")]
    attribute_words(words, [a])
    assert [w.speaker for w in words] == [0, 0]


def test_chunking_splits_on_speaker_and_pause():
    ws = [Word(0, .3, "a", 0), Word(.4, .6, "b", 0), Word(.7, .9, "c", 1), Word(3, 3.2, "d", 1)]
    assert [len(c.words) for c in make_chunks(ws)] == [2, 1, 1]


def test_caption_follows_face_then_slides_to_exit_edge():
    t = mk_track(0, 0, 100, np.linspace(500, 40, 101), cy=300, s=40)  # walks out the left side
    W, H = 1280, 720
    assert exit_edge(t, W, H) == "left"
    live = anchor_at(t, 50, W, H, "left")
    assert live.leaving is None and abs(live.x - t.cx[50]) < 1e-6
    gone = anchor_at(t, 100 + SLIDE_FRAMES + 5, W, H, "left")
    assert gone.leaving == "left" and gone.x == 0
    x, _ = place_box(gone, 200, 60, W, H, margin=10)
    assert x == 10  # flush with the left edge


def test_face_that_vanishes_midframe_holds_position():
    t = mk_track(0, 0, 50, 640, cy=360)
    assert exit_edge(t, 1280, 720) is None
    a = anchor_at(t, 200, 1280, 720, None)
    assert a.x == 640 and a.leaving is None


def test_overlapping_captions_are_pushed_apart():
    ys = resolve_collisions([(100, 400, 200, 60), (120, 410, 200, 60)], H=720, margin=10)
    assert ys[0] == 400 and ys[1] >= 400 + 60


def test_voice_clustering_finds_two_voices_and_one():
    from whosaid.voices import cluster_windows

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=192), rng.normal(size=192)
    mk = lambda c, n: (c + 0.6 * rng.normal(size=(n, 192))) / 1.0
    emb = np.vstack([mk(a, 30), mk(b, 20)])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    lab = cluster_windows(emb)
    assert len(set(lab)) == 2 and len(set(lab[:30])) == 1 and len(set(lab[30:])) == 1 and lab[0] != lab[-1]
    one = mk(a, 40)
    one /= np.linalg.norm(one, axis=1, keepdims=True)
    assert len(set(cluster_windows(one))) == 1


def test_always_lit_face_is_calibrated_away():
    # A (right) is lit all the time; B (left) is silent except 4-8s, where A dips a little and B is clearly higher
    a, b = mk_track(0, 0, 400, 800, score=3.0), mk_track(1, 0, 400, 200, score=-1.0)
    a.score[100:200] = 1.6
    b.score[100:200] = 3.0
    words = [Word(1.0, 1.4, "a"), Word(2.0, 2.4, "b"), Word(5.0, 5.4, "c"), Word(6.0, 6.4, "d"), Word(10.0, 10.4, "e")]
    attribute_words(words, [a, b])
    assert [w.speaker for w in words] == [0, 0, 1, 1, 0]


def test_words_nobody_lights_up_for_are_offscreen():
    a, b = mk_track(0, 0, 400, 800, score=3.0), mk_track(1, 0, 400, 200, score=-1.0)
    a.score[200:300] = 0.4  # 8-12s: nobody in frame is speaking
    words = [Word(1.0, 1.4, "x"), Word(9.0, 9.4, "y")]
    attribute_words(words, [a, b])
    assert [w.speaker for w in words] == [0, None]


def test_voice_breaks_near_ties():
    a, b = mk_track(0, 0, 400, 800, score=2.0), mk_track(1, 0, 400, 200, score=2.2)
    words = [Word(1.0, 1.4, "x"), Word(2.0, 2.4, "y")]
    words[0].voice = words[1].voice = 7
    attribute_words(words, [a, b], voice_tracks={7: [0]})
    assert [w.speaker for w in words] == [0, 0]


def test_short_flips_are_smoothed_but_real_turns_stay():
    from whosaid.attribute import smooth_flips

    ws = [Word(i, i + .5, "w", s) for i, s in enumerate([0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1])]
    for w in ws[3:5]:
        w.margin = 0.5   # weak: a flip
    for w in ws[7:]:
        w.margin = 4.0
    smooth_flips(ws)
    assert [w.speaker for w in ws] == [0] * 7 + [1] * 4
    for w in ws:
        w.margin = 4.0  # strong evidence: even a 2-word run is a real turn
    ws[3].speaker = ws[4].speaker = 1
    smooth_flips(ws)
    assert [w.speaker for w in ws[3:5]] == [1, 1]


def test_label_scoring():
    from types import SimpleNamespace

    from whosaid.inspect import score_labels

    ws = [Word(1, 1.4, "a", 0), Word(2, 2.4, "b", 0), Word(6, 6.4, "c", 1), Word(9, 9.4, "d", None)]
    out = score_labels(SimpleNamespace(words=ws), [
        {"start": 0, "end": 5, "speaker": 0}, {"start": 5, "end": 8, "speaker": 0}, {"start": 8, "end": 10, "speaker": None}])
    assert "75.0%" in out and "6.0s 'c'" in out


def test_simultaneous_speakers_get_separate_chunks():
    ws = [Word(0.0, 0.3, "a", 0), Word(0.1, 0.4, "x", 1), Word(0.35, 0.6, "b", 0), Word(0.45, 0.7, "y", 1)]
    chunks = make_chunks(ws)
    assert sorted((c.speaker, [w.text for w in c.words]) for c in chunks) == [(0, ["a", "b"]), (1, ["x", "y"])]


def test_stem_words_win_and_mix_only_words_are_offscreen():
    from whosaid.attribute import merge_stem_words

    mix = [Word(0, .4, "hello"), Word(1, 1.4, "hidden"), Word(5, 5.4, "other")]
    stem = {0: [Word(0.05, .45, "hello", 0)], 1: [Word(1.0, 1.4, "hidden", 1)]}
    out = merge_stem_words(mix, stem)
    assert [(w.text, w.speaker) for w in out] == [("hello", 0), ("hidden", 1), ("other", None)]


def test_shot_cut_detection_splits_on_cuts_not_motion():
    import numpy as np

    from whosaid.faces import _thumb, is_cut

    rng = np.random.default_rng(0)
    a = np.full((90, 160, 3), (200, 80, 40), np.uint8)
    b = np.full((90, 160, 3), (30, 160, 220), np.uint8)
    shifted = np.roll(a, 3, axis=1)
    noisy = np.clip(a.astype(int) + rng.integers(-6, 7, a.shape), 0, 255).astype(np.uint8)
    assert is_cut(_thumb(a), _thumb(b))
    assert not is_cut(_thumb(a), _thumb(shifted))
    assert not is_cut(_thumb(a), _thumb(noisy))


def test_speaker_whose_shot_ended_is_not_carried_onto_the_next_shot():
    a = mk_track(0, 0, 50, 100, score=3)
    a.cut_end = True                          # the camera cut away; the person did not walk out of frame
    words = [Word(1.0, 1.5, "wait"), Word(2.4, 2.8, "what")]
    attribute_words(words, [a])
    assert [w.speaker for w in words] == [0, None]


def test_overlap_seconds_counts_only_simultaneous_speech():
    from whosaid.pipeline import overlap_seconds

    a, b = mk_track(0, 0, 250, 200, score=-3), mk_track(1, 0, 250, 800, score=-3)
    a.score[0:100] = 4
    b.score[100:200] = 4                      # clean turn-taking: never both at once
    assert overlap_seconds([a, b]) == 0.0
    b.score[50:150] = 4                       # now both speak for 2 s (frames 50-99)
    assert 1.5 <= overlap_seconds([a, b]) <= 2.5
    assert overlap_seconds([a]) == 0.0
