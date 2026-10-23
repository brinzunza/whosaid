"""Assign transcribed words to face tracks and group them into caption chunks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .asr import Word
from .faces import Track
from .media import ASD_FPS

SPEAK_THRESH = 1.0       # raw LR-ASD logit a face needs to count as speaking at all
TIE = 0.5                # calibrated scores closer than this are a tie -> voice breaks it
MIN_COVERAGE = 0.5       # track must be visible for this fraction of the word
CARRY_GAP = 1.5          # seconds: carry speaker to a word after they left the frame
SWITCH_MARGIN = 1.0      # a new speaker must beat the current one by this many logits (crosstalk / reaction faces)
FILL_GAP = 0.6           # seconds: fill unattributed words between two words of one speaker


@dataclass
class Chunk:
    speaker: int | None
    words: list[Word]
    hold_end: float | None = None      # keep the caption up until here (reading time); None = last word + short linger

    @property
    def show_end(self) -> float:
        return self.hold_end if self.hold_end is not None else self.words[-1].end + 0.35

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


def _smoothed(t: Track, k: int = 5) -> np.ndarray:
    return np.convolve(np.pad(t.score, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def smoothed_scores(tracks: list[Track], asd_only: bool = False) -> dict[int, np.ndarray]:
    """5-frame smoothed speaking scores. After voice-energy fusion Track.score is the fused score; asd_only=True gives the
    detector's own (calibrated) score, which does not depend on whether the extractor found the face's voice."""
    out = {}
    for t in tracks:
        sc = t.score_asd if asd_only and getattr(t, "score_asd", None) is not None else t.score
        if sc is not None:
            out[t.id] = np.convolve(np.pad(sc, 5 // 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return out


def _word_values(words: list[Word], tracks: list[Track], scores) -> list[dict[int, float]]:
    out = []
    for w in words:
        f0, f1 = int(w.start * ASD_FPS), max(int(w.end * ASD_FPS), int(w.start * ASD_FPS) + 1)
        vals = {}
        for t in tracks:
            if t.id not in scores:
                continue
            lo, hi = max(f0, t.start), min(f1, t.end + 1)
            if hi - lo >= MIN_COVERAGE * (f1 - f0):
                vals[t.id] = float(scores[t.id][lo - t.start : hi - t.start].mean())
        out.append(vals)
    return out


def attribute_words(words: list[Word], tracks: list[Track], thresh: float = SPEAK_THRESH,
                    voice_tracks: dict[int, list[int]] | None = None, baseline_from: list[Word] | None = None) -> None:
    """Pick, per word, the face that is speaking.

    LR-ASD scores have a per-face bias (a face with an expressive mouth is 'lit' even when someone else talks),
    so faces are compared on *calibrated* scores: raw minus that face's own median over the whole speech.
    A face must also clear a raw threshold to count as speaking at all (else the word is off-screen).
    Near-ties are broken by the word's voice cluster, if one is matched to a face.
    """
    scores = smoothed_scores(tracks)                      # ranking: fused with voice energy when available
    elig_scores = smoothed_scores(tracks, asd_only=True)  # eligibility: the detector's own score
    by_id = {t.id: t for t in tracks}
    vals_per_word = _word_values(words, tracks, scores)
    elig_per_word = _word_values(words, tracks, elig_scores)
    base_vals = vals_per_word if baseline_from is None else _word_values(baseline_from, tracks, scores)
    base = {}
    for t in tracks:
        v = [vals[t.id] for vals in base_vals if t.id in vals]
        base[t.id] = float(np.median(v)) if v else 0.0

    cur = None
    for w, vals, evals in zip(words, vals_per_word, elig_per_word):
        cand = {tid: v - base[tid] for tid, v in vals.items() if evals.get(tid, -9.0) > thresh}
        if not cand:
            w.speaker = None
            continue
        ranked = sorted(cand, key=cand.get, reverse=True)
        best = ranked[0]
        close = [t for t in ranked if cand[best] - cand[t] < TIE]
        if len(close) > 1 and voice_tracks and w.voice is not None:
            mine = [t for t in close if t in voice_tracks.get(w.voice, [])]
            if len(mine) == 1:
                best = mine[0]
        if cur in cand and cur != best and cand[best] - cand[cur] < SWITCH_MARGIN:
            best = cur  # not a clear change of speaker
        others = [cand[t] for t in cand if t != best]
        w.margin = cand[best] - (max(others) if others else 0.0)
        w.speaker = cur = best

    # speaker left the frame mid-sentence: no face to score, so carry their last words forward
    prev = None
    for w in words:
        if w.speaker is not None:
            prev = w
        elif prev is not None and w.start - prev.end <= CARRY_GAP:
            t = by_id[prev.speaker]
            if int(w.start * ASD_FPS) > t.end:
                w.speaker = prev.speaker
                prev = w
    # fill short unattributed islands between two words by the same speaker
    for i in range(1, len(words) - 1):
        a, w, b = words[i - 1], words[i], words[i + 1]
        if w.speaker is None and a.speaker is not None and a.speaker == b.speaker:
            if w.start - a.end <= FILL_GAP and b.start - w.end <= FILL_GAP:
                w.speaker = a.speaker
    smooth_flips(words)


def smooth_flips(words: list[Word], max_run: int = 2, weak: float = 2.0) -> None:
    """A short run flanked on both sides by the same other speaker, and backed by only weak evidence, is a flip."""
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j].speaker == words[i].speaker:
            j += 1
        if 0 < i and j < len(words) and j - i <= max_run and words[i - 1].speaker == words[j].speaker and all(w.margin < weak for w in words[i:j]):
            for k in range(i, j):
                words[k].speaker = words[i - 1].speaker
        i = j


def _speech_spans(words: list[Word], max_gap: float = 0.8) -> list[tuple[float, float]]:
    """Time spans of phrases: consecutive words (sorted by start) with gaps <= max_gap."""
    spans: list[list[float]] = []
    for w in sorted(words, key=lambda w: w.start):
        if spans and w.start - spans[-1][1] <= max_gap:
            spans[-1][1] = max(spans[-1][1], w.end)
        else:
            spans.append([w.start, w.end])
    return [(a, b) for a, b in spans]


def merge_stem_words(mix_words: list[Word], stem_words: dict[int, list[Word]], min_overlap: float = 0.3) -> list[Word]:
    """Words from separated voices are attributed by construction (speaker = that face). Words from the full mix are
    kept only where no face's voice produced speech, and they are off-screen by definition: a visible face's voice was
    isolated and did not say them."""
    claimed = sorted((w for ws in stem_words.values() for w in ws), key=lambda w: w.start)
    out = list(claimed)
    translated = any(w.translated for w in claimed) or any(w.translated for w in mix_words)
    spans = _speech_spans(claimed) if translated else []
    for w in mix_words:
        if translated:
            # Whisper's times for translated words are approximate (some have zero duration), so word-by-word overlap
            # misses real duplicates. A mix word inside a phrase some face's voice produced (+-0.4 s) is covered.
            mid = (w.start + w.end) / 2
            is_covered = any(a - 0.4 <= mid <= b + 0.4 for a, b in spans)
        else:
            dur = max(w.end - w.start, 1e-3)
            covered = sum(max(0.0, min(w.end, c.end) - max(w.start, c.start)) for c in claimed if c.start < w.end and c.end > w.start)
            is_covered = covered / dur >= min_overlap
        if not is_covered:
            w.speaker, w.source = None, "mix"
            out.append(w)
    out.sort(key=lambda w: w.start)
    return out


def make_chunks(words: list[Word], max_words: int = 6, max_pause: float = 0.7, max_dur: float = 3.0) -> list[Chunk]:
    """Group each speaker's own words into caption chunks. Speakers are chunked independently so two people talking
    at once get two simultaneous captions instead of one chopped-up interleaving."""
    by_speaker: dict = {}
    for w in words:
        by_speaker.setdefault(w.speaker, []).append(w)
    chunks: list[Chunk] = []
    for spk, ws in by_speaker.items():
        cur: Chunk | None = None
        for w in ws:
            split = (
                cur is None
                or (spk is None and cur.words[-1].voice != w.voice)
                or w.start - cur.words[-1].end > max_pause
                or len(cur.words) >= max_words
                or w.end - cur.start > max_dur
                or (len(cur.words) >= 3 and cur.words[-1].text[-1] in ".?!")
            )
            if split:
                cur = Chunk(spk, [w])
                chunks.append(cur)
            else:
                cur.words.append(w)
    chunks.sort(key=lambda c: c.start)
    return chunks


def retime_even(chunks: list[Chunk], min_word: float = 0.06) -> None:
    """Whisper's word times for *translated* text are only approximate (and clump). Inside each translated chunk, keep the
    chunk's first start and last end and spread the words over it in proportion to their length."""
    for c in chunks:
        ws = c.words
        if len(ws) < 2 or not any(w.translated for w in ws):
            continue
        t0, t1 = ws[0].start, ws[-1].end
        span = max(t1 - t0, min_word * len(ws))
        weights = [max(len(w.text), 2) for w in ws]
        scale = span / sum(weights)
        t = t0
        for w, wt in zip(ws, weights):
            d = max(wt * scale, min_word)
            w.start, w.end = t, t + d
            t += d


def assign_hold(chunks: list[Chunk], cps: float = 17.0, min_hold: float = 1.0, max_hold: float = 6.0) -> None:
    """Reading time: a caption stays up for len/cps seconds (at least min_hold, at most max_hold after it starts) even if it
    was spoken faster, but never into the next caption of the same speaker."""
    by_spk: dict = {}
    for c in chunks:
        by_spk.setdefault(c.speaker, []).append(c)
    for cs in by_spk.values():
        cs.sort(key=lambda c: c.start)
        for i, c in enumerate(cs):
            chars = sum(len(w.text) for w in c.words) + len(c.words) - 1
            end = c.words[-1].end + 0.35
            want = min(max(c.start + max(chars / cps, min_hold), end), c.start + max_hold)
            if i + 1 < len(cs):                      # never overlap this speaker's next caption (not even the short linger)
                want = min(want, max(cs[i + 1].start - 0.05, c.words[-1].end))
            c.hold_end = want
