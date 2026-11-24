"""Click robustness of three interval envelopes: peak, quantile, granular.

For each clip:
    1. compute each envelope on the clean signal,
    2. add short clicks, compute again,
    3. distortion = mean |upper_clicked - upper_clean| + |lower_clicked - lower_clean|,
       divided by the mean width of the clean envelope.
Also reported: how much of each clean window the envelope covers, and its
mean width relative to the peak envelope (a narrower interval says more).

    python envelope_experiment.py --synthetic
    python envelope_experiment.py --data_dir /kaggle/working/LibriSpeech/dev-clean
"""

import argparse
import json

import numpy as np

from gsc.envelope import add_clicks, granular_envelope, peak_envelope, quantile_envelope


def coverage(x, lo, up, w):
    n = len(lo)
    X = x[: n * w].reshape(n, w)
    return ((X >= lo[:, None]) & (X <= up[:, None])).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--clips", type=int, default=100)
    p.add_argument("--window", type=int, default=320)
    p.add_argument("--clicks", type=int, default=20, help="clicks per 4 s clip")
    p.add_argument("--quantile", type=float, default=0.05)
    p.add_argument("--plot", default="envelope_example.png")
    p.add_argument("--out", default="envelope_results.json")
    a = p.parse_args()

    if a.synthetic:
        from gsc.data import Synthetic
        ds = Synthetic(n=a.clips, segment=64000, seed=3, fixed=True)
    else:
        from gsc.data import AudioFolder
        ds = AudioFolder(a.data_dir, "all", segment=64000, max_files=a.clips)

    methods = {
        "peak": lambda s: peak_envelope(s, a.window),
        "quantile": lambda s: quantile_envelope(s, a.window, a.quantile),
        "granular": lambda s: granular_envelope(s, a.window),
    }
    stats = {k: {"distortion": [], "coverage": [], "width": []} for k in methods}
    rng = np.random.default_rng(0)
    for i in range(len(ds)):
        x = ds[i][0].numpy().astype(np.float64)
        y, _ = add_clicks(x, a.clicks, rng=rng)
        peak_width = np.mean(np.subtract(*peak_envelope(x, a.window)[::-1]))
        for k, f in methods.items():
            lo, up = f(x)
            lo2, up2 = f(y)
            width = np.mean(up - lo)
            stats[k]["distortion"].append(
                (np.mean(np.abs(up2 - up)) + np.mean(np.abs(lo2 - lo))) / (width + 1e-12))
            stats[k]["coverage"].append(coverage(x, lo, up, a.window))
            stats[k]["width"].append(width / (peak_width + 1e-12))

    summary = {k: {m: [float(np.mean(v)), float(np.std(v))] for m, v in d.items()}
               for k, d in stats.items()}
    summary["setup"] = {"clips": len(ds), "window": a.window, "clicks": a.clicks,
                        "data": "synthetic" if a.synthetic else a.data_dir}
    print(json.dumps(summary, indent=2))
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = ds[0][0].numpy().astype(np.float64)
        y, pos = add_clicks(x, a.clicks, rng=1)
        t = np.arange(len(x)) / 16000
        tw = (np.arange(len(x) // a.window) + 0.5) * a.window / 16000
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for ax, sig, title in [(axes[0], x, "clean"), (axes[1], y, "with clicks")]:
            ax.plot(t, sig, color="0.75", lw=0.5, label="signal")
            for k, c in [("peak", "C3"), ("quantile", "C2"), ("granular", "C0")]:
                lo, up = methods[k](sig)
                ax.plot(tw, up, color=c, lw=1.2, label=k)
                ax.plot(tw, lo, color=c, lw=1.2)
            ax.set_title(title)
            ax.set_ylabel("amplitude")
        axes[0].legend(loc="upper right", ncol=4, fontsize=8)
        axes[1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=130)
        print("saved", a.plot)


if __name__ == "__main__":
    main()
