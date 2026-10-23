"""Debug views over a cached analysis: per-word evidence table and a score timeline plot."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .attribute import SPEAK_THRESH, _word_values, smoothed_scores
from .media import ASD_FPS
from .pipeline import analyze, attribute


def evidence_table(a) -> str:
    """One line per word: raw smoothed ASD per face, calibrated per face, voice, and the final decision."""
    scores = smoothed_scores(a.tracks)
    vals = _word_values(a.words, a.tracks, scores)
    base = {t.id: float(np.median([v[t.id] for v in vals if t.id in v] or [0.0])) for t in a.tracks}
    ids = [t.id for t in a.tracks]
    head = f"{'t':>6} {'word':<14}" + "".join(f"  raw{i:>2} cal{i:>2}" for i in ids) + "  voice  margin  -> speaker"
    out = [head, "baselines (per-face median raw score): " + ", ".join(f"T{i}={base[i]:+.2f}" for i in ids)]
    for w, v in zip(a.words, vals):
        cols = ""
        for i in ids:
            if i in v:
                mark = "*" if v[i] > SPEAK_THRESH else " "
                cols += f"  {v[i]:+5.1f}{mark}{v[i] - base[i]:+5.1f}"
            else:
                cols += "      -     -  "
        spk = "off-screen" if w.speaker is None else f"T{w.speaker}"
        out.append(f"{w.start:6.1f} {w.text:<14}{cols}  {str(w.voice):>5}  {w.margin:6.1f}  {spk}")
    return "\n".join(out)


def score_labels(a, labels: list[dict]) -> str:
    """labels: [{"start": s, "end": s, "speaker": track id | null}] -> word-level accuracy + confusion + mismatches."""
    rows, wrong = [], []
    for w in a.words:
        mid = (w.start + w.end) / 2
        truth = next((l["speaker"] for l in labels if l["start"] <= mid < l["end"]), "unlabeled")
        if truth == "unlabeled":
            continue
        rows.append((truth, w.speaker))
        if truth != w.speaker:
            wrong.append(w)
    if not rows:
        return "no words fall inside labeled intervals"
    acc = sum(t == p for t, p in rows) / len(rows)
    conf: dict = {}
    for t, p in rows:
        conf[(t, p)] = conf.get((t, p), 0) + 1
    name = lambda x: "off-screen" if x is None else f"T{x}"
    out = [f"word-level speaker accuracy: {acc:.1%} ({len(rows)} labeled words)", "confusion (truth -> predicted):"]
    out += [f"  {name(t):>10} -> {name(p):<10} {n}" for (t, p), n in sorted(conf.items(), key=lambda kv: -kv[1])]
    out += ["mismatches: " + ", ".join(f"{w.start:.1f}s '{w.text}'(pred {name(w.speaker)})" for w in wrong[:40])]
    return "\n".join(out)


def timeline_plot(a, path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#e0a800", "#17a2b8", "#e5534b", "#3fb950", "#a371f7", "#f0883e"]
    scores = smoothed_scores(a.tracks)
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    ax = axes[0]
    for t in a.tracks:
        x = (t.frames) / ASD_FPS
        ax.plot(x, scores[t.id], color=colors[t.id % 6], lw=1.2, label=f"face T{t.id}")
    ax.axhline(SPEAK_THRESH, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("ASD logit (smoothed)")
    ax.legend(loc="upper right")
    ax.set_title("face speaking scores; dashed = raw threshold")
    ax = axes[1]
    ax.set_ylabel("voice")
    for w in a.words:
        if w.voice is not None:
            ax.barh(0, w.end - w.start, left=w.start, color=plt.cm.tab10(w.voice % 10), height=0.8)
    ax.set_yticks([])
    ax = axes[2]
    ax.set_ylabel("attributed")
    for w in a.words:
        col = "#999999" if w.speaker is None else colors[w.speaker % 6]
        ax.barh(0, w.end - w.start, left=w.start, color=col, height=0.8)
    ax.set_yticks([])
    ax.set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return Path(path)


def main(argv):
    import argparse

    p = argparse.ArgumentParser(prog="whosaid inspect", description="Debug view of a cached analysis.")
    p.add_argument("input")
    p.add_argument("--workdir", required=True, help="work dir from a previous run (e.g. out/argument.work)")
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--separate", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--plot", help="write a timeline PNG here")
    p.add_argument("--labels", help="JSON list of {start,end,speaker} ground-truth intervals; prints accuracy")
    p.add_argument("--table", action="store_true", help="print the per-word evidence table")
    a = p.parse_args(argv)
    an = analyze(a.input, a.workdir, a.whisper_model, use_voices=not a.no_voice, separate=a.separate)
    attribute(an)
    if a.labels:
        import json

        print(score_labels(an, json.loads(Path(a.labels).read_text())))
    if a.table or not (a.plot or a.labels):
        print(evidence_table(an))
    if a.plot:
        print("wrote", timeline_plot(an, a.plot))
