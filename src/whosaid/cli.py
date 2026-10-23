import argparse
from pathlib import Path

from .pipeline import run


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "synth":
        from .synth.cli import main as synth_main

        return synth_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "fuse":
        from .fusion_train import main as fuse_main

        return fuse_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        from .sync import main as sync_main

        return sync_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "storage":
        from .disk import report

        return print(report())
    if len(sys.argv) > 1 and sys.argv[1] == "asd-head":
        from .asd_head import main as head_main

        return head_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "asd":
        from .asd_train import main as asd_main

        return asd_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "decide":
        from .decide import main as decide_main

        return decide_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        from .bench import main as bench_main

        return bench_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        from .inspect import main as inspect_main

        return inspect_main(sys.argv[2:])
    p = argparse.ArgumentParser(prog="whosaid", description="Burn speaker-locked captions into a video.")
    p.add_argument("input")
    p.add_argument("-o", "--output", help="default: <input>.captioned.mp4")
    p.add_argument("--whisper-model", default="small", help="tiny/base/small/medium/large-v3 (default: small)")
    p.add_argument("--language", default=None, help="force a language code, e.g. en (default: auto-detect)")
    p.add_argument("--debug", action="store_true", help="draw face boxes with live speaking scores")
    p.add_argument("--no-json", action="store_true", help="don't write the .json with tracks and word attribution")
    p.add_argument("--workdir", help="cache dir for heavy stages (default: <output>.work)")
    p.add_argument("--fresh", action="store_true", help="ignore cached stages")
    p.add_argument("--separate", choices=["auto", "on", "off"], default="auto",
                   help="extract each face's voice before transcribing (auto: when two faces are ever on screen together)")
    p.add_argument("--caption-style", choices=["outline", "box"], default="outline",
                   help="outline: small outlined text, words appear as spoken (default); box: dark box with the whole line")
    p.add_argument("--asd-model", help="speaking-detector weights (default: pretrained LR-ASD)")
    p.add_argument("--asd-calib", choices=["none", "model", "match", "face"], default="none",
                   help="calibrate speaking scores: model = validation-derived scale; match = same, at the pretrained detector's operating point; face = model + per-face valley shift")
    p.add_argument("--fuse", help="fuse extracted-voice energy into the speaking score: a fusion json, or 'on' for models/fusion.json")
    p.add_argument("--translate-to", default=None, metavar="LANG",
                   help="caption in this language when the speech is in another (only 'en' for now). Each voice's language is detected separately")
    p.add_argument("--static-filter", choices=["on", "off"], default="off",
                   help="ignore faces that never move or change their mouth (photos, posters, frozen overlays)")
    p.add_argument("--no-voice", action="store_true", help="skip voice clustering (per-word face scores only)")
    a = p.parse_args()
    src = Path(a.input)
    out = Path(a.output) if a.output else src.with_suffix(".captioned.mp4")
    run(src, out, a.whisper_model, a.language, a.debug, not a.no_json, use_voices=not a.no_voice, workdir=a.workdir, fresh=a.fresh, separate=a.separate, caption_style=a.caption_style, asd_model=a.asd_model, asd_calib=a.asd_calib, fuse=a.fuse, translate=a.translate_to, static_filter=a.static_filter == "on")


if __name__ == "__main__":
    main()
