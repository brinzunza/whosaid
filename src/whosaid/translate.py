"""Captions in English for speech in other languages (step 1: Whisper's built-in translate task).

Everything before transcription (faces, speaking detection, voice extraction) is language-blind. The language is
decided *per audio stream*: each face's extracted voice is detected and transcribed on its own, so two people speaking
different languages, even over each other, each get the right language. Translation is only to English for now
(that is all Whisper's translate task does).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import asr
from .asr import AsrResult

MIN_LANG_PROB = 0.6     # below this the detected language is not trusted (short or noisy streams misfire)
SUPPORTED_TARGETS = ("en",)


@dataclass
class CaptionAsr:
    words: list
    source_lang: str | None      # language of the speech
    language_prob: float
    translated: bool
    how: str                     # "forced" | "detected" | "fallback" | "unknown"


def transcribe_for_captions(audio, model_size: str, forced: str | None = None, target: str | None = None,
                            fallback: str | None = None) -> CaptionAsr:
    """Detect the speech language (unless forced), then transcribe, or translate to `target` when it differs.

    forced   a language code the user insisted on (skips detection)
    fallback language to use when detection is unsure (e.g. the whole mix's language for a short voice)"""
    if forced:
        lang, prob, how = forced, 1.0, "forced"
    elif target is None:
        lang, prob, how = None, 0.0, "unknown"           # plain transcription: let Whisper decide internally
    else:
        lang, prob = asr.detect_language(audio, model_size)
        how = "detected"
        if lang is None or prob < MIN_LANG_PROB:
            lang, how = (fallback, "fallback") if fallback else (None, "unknown")
    translate = target is not None and (lang is None or lang != target)
    res: AsrResult = asr.transcribe_ex(audio, model_size, language=lang, task="translate" if translate else "transcribe")
    source = lang or res.language
    for w in res.words:
        w.lang = source
        w.translated = translate
    return CaptionAsr(res.words, source, prob if lang else res.language_prob, translate, how)


def check_target(target: str | None) -> None:
    if target is not None and target not in SUPPORTED_TARGETS:
        raise SystemExit(f"--translate-to {target!r}: Whisper's translate task only produces English; "
                         f"supported targets: {', '.join(SUPPORTED_TARGETS)}")
