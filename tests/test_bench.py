import numpy as np

from whosaid.bench import edit_distance, norm, score, si_snr, track_tile
from whosaid.faces import Track
from whosaid.synth.compose import Clip, Tile, _labels, _layout, utterances


def W(start, end, text, tile):
    return dict(start=start, end=end, text=text, tile=tile)


LABELS = dict(
    tiles=[dict(id="A", rect=[0, 0, 640, 720], on_screen=True), dict(id="B", rect=[640, 0, 640, 720], on_screen=True)],
    overlap=[[2.0, 3.0]],
    words=[W(0, .4, "hello", "A"), W(.5, .9, "there", "A"), W(2.0, 2.4, "good", "A"), W(2.1, 2.5, "morning", "B"), W(5, 5.4, "bye", "B")],
)


def test_edit_distance_basics():
    assert edit_distance(list("abc"), list("abc")) == 0
    assert edit_distance(list("abc"), list("abd")) == 1
    assert edit_distance([], list("ab")) == 2
    assert norm("Hello,") == "hello"


def test_perfect_prediction_scores_zero_error():
    pred = [dict(w) for w in LABELS["words"]]
    r = score(pred, LABELS)
    assert r["cpwer"] == 0 and r["attr_acc_overlap"] == [2, 2] and r["attr_acc_clear"] == [3, 3]


def test_right_words_wrong_speaker_is_penalised_and_visible_in_attribution():
    pred = [dict(w) for w in LABELS["words"]]
    pred[2]["tile"] = "B"          # 'good' credited to the wrong person
    r = score(pred, LABELS)
    assert r["errors"] == 2        # one deletion for A, one insertion for B
    assert r["attr_acc_overlap"] == [1, 2] and r["attr_acc_clear"] == [3, 3]


def test_missing_and_hallucinated_words():
    pred = [dict(w) for w in LABELS["words"] if w["text"] != "bye"] + [W(7, 7.4, "thanks", "X")]
    r = score(pred, LABELS)
    assert r["errors"] == 2 and r["ref_words"] == 5 and abs(r["cpwer"] - 0.4) < 1e-9
    assert r["recall_clear"][0] == 2 and r["recall_clear"][1] == 3


def test_track_tile_uses_position_and_clamps_frame():
    t = Track(0, np.arange(10, 20), np.full(10, 900.0), np.full(10, 300.0), np.full(10, 50.0), np.ones(10, bool))
    assert track_tile(t, 15, LABELS["tiles"]) == "B"
    assert track_tile(t, 99, LABELS["tiles"]) == "B"   # after the track ended: last position


def test_si_snr_is_scale_invariant_and_orders_quality():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000)
    assert si_snr(3 * ref, ref) > 60
    assert si_snr(ref + 0.1 * rng.standard_normal(16000), ref) > si_snr(ref + 1.0 * rng.standard_normal(16000), ref)


def _clip(words):
    return Clip("c99", "P", "x.mp4", {"words": words, "width": 100, "height": 100}, np.zeros(16000 * 30, np.float32))


def test_utterances_split_on_pause_and_labels_shift_into_scene_time():
    ws = [dict(start=1.0, end=1.4, text="one"), dict(start=1.5, end=2.5, text="two."),
          dict(start=4.0, end=4.5, text="three"), dict(start=4.6, end=5.6, text="four")]
    c = _clip(ws)
    assert utterances(c, min_dur=0.5) == [(1.0, 2.5), (4.0, 5.6)]
    t = Tile("A", c, (0, 0, 640, 720), windows=[(10.0, 14.0, 3.9)])     # source 3.9-7.9 plays at scene 10-14
    lab = _labels([t], 20.0, dict(scene="s"))
    assert [(w["text"], w["start"]) for w in lab["words"]] == [("three", 10.1), ("four", 10.7)]
    assert lab["speech"]["A"] == [[10.1, 11.7]]


def test_overlap_intervals_from_two_tiles():
    a = _clip([dict(start=0.0, end=3.0, text="a")]); b = _clip([dict(start=0.0, end=3.0, text="b")])
    ta = Tile("A", a, (0, 0, 640, 720), windows=[(0.0, 3.0, 0.0)])
    tb = Tile("B", b, (640, 0, 640, 720), windows=[(2.0, 5.0, 0.0)])
    lab = _labels([ta, tb], 5.0, dict(scene="s"))
    assert lab["overlap"] == [[2.0, 3.0]]


def test_layouts_tile_the_canvas():
    for n in (1, 2, 3):
        assert sum(w * h for _, _, w, h in _layout(n)) == 1280 * 720


def test_silent_participant_does_not_crash_timeline():
    from whosaid.synth.compose import _tile_timeline

    t = Tile("A", _clip([]), (0, 0, 640, 720), windows=[])
    assert _tile_timeline(t, 10.0, "freeze") == [(10.0, 0.0, False)]
