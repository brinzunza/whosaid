"""Word-level transcription with faster-whisper: plain, language-detected, or translated to English."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str  # stripped
    speaker: int | None = None  # track id, None = off-screen / unknown
    voice: int | None = None    # voice cluster id (same voice = same id)
    margin: float = 0.0         # calibrated score lead of the chosen face over the runner-up
    source: str = "mix"         # "mix" (transcribed from the full audio) or "stem" (from one face's separated voice)
    lang: str | None = None     # language the speech was in (when known)
    translated: bool = False    # text is Whisper's English translation; its word times are approximate


@dataclass
class AsrResult:
    words: list[Word]
    language: str | None = None        # language Whisper decided on / was told
    language_prob: float = 0.0
    task: str = "transcribe"


_MODELS: dict = {}


def _model(model_size: str):
    from faster_whisper import WhisperModel

    if model_size not in _MODELS:
        _MODELS[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _MODELS[model_size]


def _as_input(audio):
    return audio if not isinstance(audio, str) and not hasattr(audio, "__fspath__") else str(audio)


def detect_language(audio, model_size: str = "small", segments: int = 3) -> tuple[str | None, float]:
    """Language of the *speech* in a float32 16 kHz array (silence is skipped by VAD). (None, 0.0) if there is none."""
    try:
        lang, prob, _ = _model(model_size).detect_language(
            audio=audio, vad_filter=True, language_detection_segments=segments)
        return lang, float(prob)
    except Exception:       # no speech after VAD, audio too short, ...
        return None, 0.0


def transcribe_ex(audio, model_size: str = "small", language: str | None = None, task: str = "transcribe") -> AsrResult:
    """audio: wav path or float32 mono 16 kHz array. task='translate' makes Whisper output English text."""
    segments, info = _model(model_size).transcribe(
        _as_input(audio), word_timestamps=True, vad_filter=True, language=language, task=task,
        condition_on_previous_text=False,
    )
    words = []
    for seg in segments:
        for w in seg.words or []:
            t = w.word.strip()
            if t:
                words.append(Word(float(w.start), float(w.end), t, translated=(task == "translate")))
    return AsrResult(words, getattr(info, "language", language), float(getattr(info, "language_probability", 0.0)), task)


def transcribe(audio, model_size: str = "small", language: str | None = None) -> list[Word]:
    """audio: wav path or float32 mono 16 kHz array."""
    return transcribe_ex(audio, model_size, language).words
