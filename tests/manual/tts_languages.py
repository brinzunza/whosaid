"""Manual plumbing check (needs macOS `say` and the Whisper model): synthetic speech in several languages -> language
detection -> translation to English, checked against keywords of the known meaning.

    uv run python tests/manual/tts_languages.py [whisper-size]

This tests the routing and that translation produces the right *content*. It says little about quality on real
(noisy, spontaneous) speech; for that, use real clips."""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from whosaid.media import read_wav
from whosaid.translate import transcribe_for_captions

CASES = [  # voice, expected language, sentence, any-of keyword groups that must appear in the English
    ("Eddy (French (France))", "fr", "Bonjour, je voudrais réserver une table pour deux personnes ce soir.", [["table"], ["two"], ["tonight", "evening"]]),
    ("Eddy (Spanish (Spain))", "es", "Buenos días, ¿dónde está la estación de tren más cercana?", [["train"], ["station"]]),
    ("Eddy (German (Germany))", "de", "Das Wetter ist heute sehr schön, deshalb gehen wir im Park spazieren.", [["weather"], ["park"], ["walk"]]),
    ("Eddy (Italian (Italy))", "it", "Vorrei un caffè e un cornetto, per favore.", [["coffee"], ["croissant", "pastry", "cornetto", "roll"]]),
    ("Eddy (Japanese (Japan))", "ja", "今日は天気がいいので、公園に散歩に行きます。", [["weather", "nice", "sunny", "good"], ["park"], ["walk", "stroll"]]),
    ("Eddy (Korean (South Korea))", "ko", "저는 내일 친구와 함께 영화를 보러 갈 거예요.", [["movie", "film"], ["friend"], ["tomorrow"]]),
    ("Eddy (Portuguese (Brazil))", "pt", "Eu gostaria de comprar uma passagem de ônibus para o Rio de Janeiro.", [["ticket", "bus"], ["rio"]]),
    ("Yelda", "tr", "Merhaba, bugün hava çok güzel, bu yüzden parkta yürüyüş yapmak istiyorum.", [["weather", "day", "nice", "beautiful"], ["park"], ["walk", "stroll"]]),
    ("Yelda", "tr", "Yarın sabah arkadaşımla buluşup birlikte kahvaltı yapacağız.", [["tomorrow"], ["friend"], ["breakfast"]]),
    ("Yelda", "tr", "Affedersiniz, en yakın eczane nerede ve saat kaça kadar açık?", [["pharmacy", "drugstore", "chemist"], ["where", "nearest", "closest"], ["open"]]),
    ("Eddy (English (US))", "en", "Good morning, could you tell me where the nearest train station is?", [["train"], ["station"]]),
]


def main(size="small"):
    print(f"whisper {size}\n")
    ok_lang = ok_content = 0
    for voice, lang, text, groups in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            aiff, wav = Path(tmp) / "s.aiff", Path(tmp) / "s.wav"
            try:
                subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print(f"[{lang}] voice '{voice}' not installed, skipped")
                continue
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(wav)], check=True)
            audio = read_wav(wav).astype(np.float32) / 32768.0
        t0 = time.time()
        r = transcribe_for_captions(audio, size, target="en")
        english = " ".join(w.text for w in r.words)
        hit = sum(any(k in english.lower() for k in g) for g in groups) / len(groups)
        ok_lang += r.source_lang == lang
        ok_content += hit >= 0.67
        print(f"[{lang}] detected {r.source_lang} ({r.language_prob:.2f}, {r.how}), translated={r.translated}, content {hit:.0%}, {time.time() - t0:.1f}s\n      -> {english}")
    print(f"\nlanguage correct: {ok_lang}/{len(CASES)}   content mostly right (>=2/3 keyword groups): {ok_content}/{len(CASES)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "small")
