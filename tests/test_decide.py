import numpy as np
from types import SimpleNamespace

from whosaid.asr import Word
from whosaid.decide import FEATURES, Row, apply_choices, choose, label_rows
from whosaid.faces import Track


def trk(i, cx):
    n = 100
    return Track(i, np.arange(n), np.full(n, float(cx)), np.full(n, 300.0), np.full(n, 50.0), np.ones(n, bool))


LABELS = dict(tiles=[dict(id="A", rect=[0, 0, 640, 720], on_screen=True), dict(id="B", rect=[640, 0, 640, 720], on_screen=True),
                     dict(id="X", rect=None, on_screen=False)],
              words=[dict(start=0.0, end=0.4, text="hello", tile="A"), dict(start=1.0, end=1.4, text="there", tile="B"),
                     dict(start=2.0, end=2.4, text="far", tile="X")])


def rows_for(i, tracks):
    xs = [0.0] * len(FEATURES)
    return [Row(i, ("track", t), xs) for t in tracks] + [Row(i, ("off",), xs), Row(i, ("drop",), xs)]


def test_labels_follow_ground_truth():
    a = SimpleNamespace(tracks=[trk(0, 200), trk(1, 900)])
    words = [Word(0.0, 0.4, "hello"), Word(1.0, 1.4, "there"), Word(2.0, 2.4, "far"), Word(3.0, 3.4, "ghost")]
    rows = sum((rows_for(i, [0, 1]) for i in range(4)), [])
    label_rows(a, words, rows, LABELS)
    ys = {(r.word, r.cand): r.y for r in rows}
    assert ys[(0, ("track", 0))] == 1 and ys[(0, ("track", 1))] == 0 and ys[(0, ("off",))] == 0 and ys[(0, ("drop",))] == 0
    assert ys[(1, ("track", 1))] == 1
    assert ys[(2, ("off",))] == 1                       # a word of the off-screen voice
    assert ys[(3, ("drop",))] == 1 and ys[(3, ("track", 0))] == 0   # hallucination: drop is right


def test_word_with_no_correct_candidate_is_unlabeled():
    a = SimpleNamespace(tracks=[trk(0, 200)])           # nobody sits in tile B
    words = [Word(1.0, 1.4, "there")]
    rows = rows_for(0, [0])
    label_rows(a, words, rows, LABELS)
    assert all(r.y == -1 for r in rows)


def test_choose_picks_best_candidate_and_drop_removes_word():
    rows = rows_for(0, [0, 1]) + rows_for(1, [0, 1])
    proba = np.array([0.1, 0.7, 0.2, 0.0,      0.1, 0.1, 0.2, 0.9])
    choice = choose(rows, proba)
    assert choice == {0: ("track", 1), 1: ("drop",)}
    words = [Word(0, 1, "a", speaker=None), Word(1, 2, "b", speaker=0)]
    out = apply_choices(words, choice)
    assert [(w.text, w.speaker) for w in out] == [("a", 1)]
