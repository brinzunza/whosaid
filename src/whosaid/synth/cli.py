import argparse
import json
from pathlib import Path

from .compose import generate, load_library


def main(argv):
    p = argparse.ArgumentParser(prog="whosaid synth", description="Generate synthetic multi-person scenes with labels.")
    p.add_argument("--lib", default="data/lib", help="prepared library (see synth.library.prepare)")
    p.add_argument("--clips", default="data/single")
    p.add_argument("--persons", default="data/persons.json", help='{"clip": "person"} and "splits" {"train": [...persons]}')
    p.add_argument("--out", default="data/synth")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits", default="test", help="comma list of splits to generate, cycled")
    p.add_argument("--mouthing-p", type=float, default=0.3, help="share of scenes whose listeners keep mouthing (hard negatives)")
    a = p.parse_args(argv)
    cfg = json.loads(Path(a.persons).read_text())
    lib = load_library(a.lib, a.clips, cfg["persons"], cfg.get("audio_only", []))
    splits = {s: cfg["splits"][s] for s in a.splits.split(",")}
    generate(a.out, lib, splits, a.n, a.seed, mouthing_p=a.mouthing_p)
