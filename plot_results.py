"""Quality vs measured bit rate, one line per quantizer.

Reads the last line of every runs/*/eval.jsonl. Compare methods at matched
entropy rate (x axis), not at matched codebook size: two codecs with K=256
can use very different numbers of bits once entropy coded.

    python plot_results.py --runs runs --out results.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = [("lsd_db", "log-spectral distance (dB), lower is better"),
           ("speaker_sim", "speaker similarity, higher is better"),
           ("wer_vs_original", "WER vs original transcript, lower is better"),
           ("envelope", "envelope error, lower is better")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results.png")
    a = p.parse_args()

    rows = []
    for f in sorted(Path(a.runs).glob("*/eval.jsonl")):
        last = f.read_text().strip().splitlines()[-1]
        r = json.loads(last)
        r["run"] = f.parent.name
        rows.append(r)
    if not rows:
        raise SystemExit(f"no eval.jsonl under {a.runs}")

    by_q = defaultdict(list)
    for r in rows:
        by_q[r["quantizer"]].append(r)
    metrics = [(k, t) for k, t in METRICS if any(k in r for r in rows)]

    print(f"{'run':24s} {'quantizer':8s} {'nominal':>8s} {'entropy':>8s} "
          + " ".join(f"{k:>16s}" for k, _ in metrics))
    for r in sorted(rows, key=lambda r: (r["quantizer"], r["entropy_bps"])):
        print(f"{r['run']:24s} {r['quantizer']:8s} {r['nominal_bps']:8.0f} {r['entropy_bps']:8.0f} "
              + " ".join(f"{r.get(k, float('nan')):16.4f}" for k, _ in metrics))

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8), squeeze=False)
    markers = {"fuzzy": "o", "vq": "s", "fsq": "^"}
    for ax, (k, title) in zip(axes[0], metrics):
        for i, r in enumerate(sorted(rows, key=lambda r: r["run"])):
            if k in r:
                ax.plot(r["entropy_bps"], r[k], markers.get(r["quantizer"], "o"),
                        color=f"C{i}", ms=8, label=r["run"])
        ax.set_xlabel("entropy rate (bit/s)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("saved", a.out)


if __name__ == "__main__":
    main()
