"""Audio datasets.

AudioFolder : every .wav/.flac under a folder (e.g. LibriSpeech). Files are
              split into train/eval by a hash of the file path, so the split
              is stable across runs and machines.
Synthetic   : speech-like test signals (harmonic voiced parts with an
              amplitude envelope, noise bursts, silence). Only for checking
              that the code runs; no real results should come from it.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SR = 16000


def _is_eval(path, eval_fraction):
    h = int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16)
    return (h % 1000) < eval_fraction * 1000


def load_audio(path, sr=SR):
    import soundfile as sf
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if fs != sr:
        from scipy.signal import resample_poly
        g = math.gcd(fs, sr)
        x = resample_poly(x, sr // g, fs // g).astype(np.float32)
    return x


class AudioFolder(Dataset):
    def __init__(self, root, split="train", segment=SR, eval_fraction=0.02,
                 max_files=None, eval_offset=SR // 2):
        files = sorted(p for ext in ("*.wav", "*.flac") for p in Path(root).rglob(ext))
        if split == "all":  # every file, fixed crops (a separate eval folder)
            self.files = files
        else:
            want_eval = split == "eval"
            self.files = [f for f in files
                          if _is_eval(f.relative_to(root), eval_fraction) == want_eval]
        if max_files and len(self.files) > max_files:
            if split == "train":
                self.files = self.files[:max_files]
            else:  # evenly spaced over the sorted list, so every speaker folder is sampled
                idx = np.linspace(0, len(self.files) - 1, max_files).round().astype(int)
                self.files = [self.files[i] for i in idx]
        if not self.files:
            raise FileNotFoundError(f"no {split} audio found under {root}")
        self.split, self.segment, self.eval_offset = split, segment, eval_offset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = load_audio(self.files[i])
        n = self.segment
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        if self.split == "train":
            # torch's generator is seeded differently in each loader worker
            start = int(torch.randint(0, len(x) - n + 1, (1,)))
        else:  # fixed crop, skipping the leading silence
            start = min(self.eval_offset, len(x) - n)
        x = x[start:start + n]
        peak = np.abs(x).max()
        if peak > 0:  # loudness normalization to a fixed peak
            x = 0.9 * x / peak
        return torch.from_numpy(x.copy())[None]


class Synthetic(Dataset):
    def __init__(self, n=256, segment=SR, seed=0, fixed=False):
        self.n, self.segment, self.seed, self.fixed = n, segment, seed, fixed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 100003 + i if self.fixed else None)
        n, t = self.segment, np.arange(self.segment) / SR
        x = np.zeros(n, np.float32)
        pos = 0
        while pos < n:
            length = int(rng.uniform(0.08, 0.3) * SR)
            seg = slice(pos, min(n, pos + length))
            tt = t[seg] - t[pos]
            kind = rng.choice(["voiced", "noise", "silence"], p=[0.6, 0.2, 0.2])
            if kind == "voiced":
                f0 = rng.uniform(90, 250) * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 4) * tt))
                phase = 2 * np.pi * np.cumsum(f0) / SR
                formant = rng.uniform(400, 2500)
                sig = sum(np.exp(-((k * f0.mean() - formant) / 800) ** 2) * np.sin(k * phase)
                          for k in range(1, 30))
            elif kind == "noise":
                sig = rng.standard_normal(len(tt)) * 0.3
            else:
                sig = rng.standard_normal(len(tt)) * 0.003
            env = np.sin(np.pi * np.linspace(0, 1, len(tt))) ** 2
            x[seg] = (sig * env * rng.uniform(0.2, 1.0)).astype(np.float32)
            pos += length
        x = 0.9 * x / (np.abs(x).max() + 1e-8)
        return torch.from_numpy(x)[None]
