import numpy as np
import pytest

from whosaid import asr as asr_mod
from whosaid import translate as tr
from whosaid.asr import AsrResult, Word
from whosaid.attribute import Chunk, assign_hold, make_chunks, retime_even


class FakeWhisper:
    """Records how the routing code calls Whisper; `detected` is what language detection will say."""

    def __init__(self, monkeypatch, detected=("es", 0.95)):
        self.calls = []
        self.detected = detected
        monkeypatch.setattr(asr_mod, "detect_language", self.detect)
        monkeypatch.setattr(asr_mod, "transcribe_ex", self.transcribe_ex)

    def detect(self, audio, model_size="small", segments=3):
        self.calls.append(("detect",))
        return self.detected

    def transcribe_ex(self, audio, model_size="small", language=None, task="transcribe"):
        self.calls.append(("transcribe", language, task))
        return AsrResult([Word(0.0, 0.4, "hello", translated=(task == "translate"))], language or "xx", 0.9, task)


def test_foreign_speech_is_translated_with_the_detected_language(monkeypatch):
    fw = FakeWhisper(monkeypatch, ("es", 0.95))
    r = tr.transcribe_for_captions(np.zeros(16000, np.float32), "small", target="en")
    assert fw.calls == [("detect",), ("transcribe", "es", "translate")]
    assert r.translated and r.source_lang == "es" and r.words[0].lang == "es" and r.words[0].translated and r.how == "detected"


def test_speech_already_in_the_target_language_is_just_transcribed(monkeypatch):
    fw = FakeWhisper(monkeypatch, ("en", 0.98))
    r = tr.transcribe_for_captions(np.zeros(16000, np.float32), "small", target="en")
    assert fw.calls[-1] == ("transcribe", "en", "transcribe") and not r.translated and not r.words[0].translated


def test_unsure_detection_falls_back_to_the_mix_language(monkeypatch):
    fw = FakeWhisper(monkeypatch, ("de", 0.30))
    r = tr.transcribe_for_captions(np.zeros(16000, np.float32), "small", target="en", fallback="fr")
    assert fw.calls[-1] == ("transcribe", "fr", "translate") and r.how == "fallback" and r.source_lang == "fr"


def test_unsure_detection_without_fallback_lets_whisper_decide_and_still_translates(monkeypatch):
    fw = FakeWhisper(monkeypatch, (None, 0.0))
    r = tr.transcribe_for_captions(np.zeros(16000, np.float32), "small", target="en")
    assert fw.calls[-1] == ("transcribe", None, "translate") and r.how == "unknown"


def test_forced_language_skips_detection(monkeypatch):
    fw = FakeWhisper(monkeypatch)
    r = tr.transcribe_for_captions(np.zeros(16000, np.float32), "small", forced="ja", target="en")
    assert ("detect",) not in fw.calls and fw.calls == [("transcribe", "ja", "translate")] and r.how == "forced"


def test_no_target_is_plain_transcription_with_no_detection(monkeypatch):
    fw = FakeWhisper(monkeypatch)
    r = tr.transcribe_for_captions(np.zeros(16000, np.float32), "small", forced="en")
    assert fw.calls == [("transcribe", "en", "transcribe")] and not r.translated


def test_only_english_is_a_supported_target():
    tr.check_target(None)
    tr.check_target("en")
    with pytest.raises(SystemExit, match="only produces English"):
        tr.check_target("fr")


def words(*texts, t0=0.0, dur=0.4, translated=True):
    return [Word(t0 + i * dur, t0 + (i + 1) * dur, t, translated=translated) for i, t in enumerate(texts)]


def test_retime_even_keeps_the_span_order_and_weights_by_length():
    ws = words("a", "extraordinary")
    ws[0].start, ws[0].end, ws[1].start, ws[1].end = 1.0, 1.0, 1.0, 3.0          # clumped, as Whisper's translated times can be
    c = Chunk(0, ws)
    retime_even([c])
    assert ws[0].start == 1.0 and abs(ws[-1].end - 3.0) < 1e-6
    assert all(a.end <= b.start + 1e-9 for a, b in zip(ws, ws[1:])) and (ws[1].end - ws[1].start) > (ws[0].end - ws[0].start)


def test_retime_leaves_untranslated_chunks_alone():
    ws = words("hello", "there", translated=False)
    before = [(w.start, w.end) for w in ws]
    retime_even([Chunk(0, ws)])
    assert [(w.start, w.end) for w in ws] == before


def test_hold_gives_long_captions_reading_time_but_never_runs_into_the_next_one():
    c1 = Chunk(0, words("this", "is", "quite", "a", "long", "translated", "sentence", "to", "read", dur=0.15))   # 1.35 s spoken
    c2 = Chunk(0, words("next", t0=2.0))
    c3 = Chunk(1, words("other", "speaker", t0=1.0, dur=0.2))
    assign_hold([c1, c2, c3], cps=17.0, min_hold=1.0, max_hold=6.0)
    assert c1.hold_end > c1.words[-1].end + 0.35 and c1.hold_end <= c2.start - 0.05 + 1e-9
    assert c3.hold_end >= c3.start + 1.0                                         # short captions still stay a second
    assert Chunk(0, words("x")).show_end == pytest.approx(0.4 + 0.35)            # default: unchanged behaviour


def test_chunks_default_has_no_hold():
    chunks = make_chunks(words("one", "two", translated=False))
    assert all(c.hold_end is None for c in chunks)


def test_adjacent_captions_of_one_speaker_never_overlap():
    c1 = Chunk(0, words("one", "two", "three"))                        # ends at 1.2
    c2 = Chunk(0, words("four", "five", t0=1.3))                        # starts 0.1 s after c1's last word
    assign_hold([c1, c2])
    assert c1.hold_end <= c2.start - 0.05 + 1e-9 and c1.hold_end >= c1.words[-1].end


def test_translated_mix_words_inside_a_voice_phrase_are_not_duplicated_but_real_gaps_stay():
    from whosaid.attribute import merge_stem_words

    stem = {1: [Word(4.2, 4.8, "I", 1, translated=True), Word(5.0, 5.0, "work", 1, translated=True),
                Word(6.0, 6.0, "Sunday", 1, translated=True), Word(6.1, 6.3, "example", 1, translated=True)]}
    mix = [Word(5.0, 5.0, "work", translated=True), Word(5.8, 5.8, "Sunday", translated=True),      # duplicates of the phrase
           Word(12.0, 12.5, "stranger", translated=True)]                                            # a different, distant speaker
    out = merge_stem_words(mix, stem)
    assert [(w.text, w.speaker) for w in out].count(("work", None)) == 0 and ("Sunday", None) not in [(w.text, w.speaker) for w in out]
    assert ("stranger", None) in [(w.text, w.speaker) for w in out]


def test_untranslated_merge_keeps_strict_overlap_rule():
    from whosaid.attribute import merge_stem_words

    stem = {1: [Word(4.2, 4.8, "hello", 1)]}
    mix = [Word(5.0, 5.4, "near", None)]            # close in time but not overlapping: must stay (old behaviour)
    assert [w.text for w in merge_stem_words(mix, stem)] == ["hello", "near"]
