"""Evaluation: rate and quality on a fixed eval set.

Rate    : nominal bits/s, entropy bits/s, codes used per group.
Quality : multi-resolution STFT loss, log-spectral distance (dB),
          envelope error, and, when priors are given,
          speaker cosine similarity and the word error rate of the decoded
          speech against the ASR transcript of the original speech.
          (The reference is the original's transcript, not the ground-truth
          text, so WER measures what the codec destroys, not ASR mistakes.)
"""

from pathlib import Path

import torch

from .losses import EnvelopeLoss, MultiResSTFTLoss, _stft_mag
from .priors import word_error_rate
from .rate import IndexHistogram

SR = 16000


def log_spectral_distance(y_hat, y, n_fft=512, hop=128):
    a = 20 * torch.log10(_stft_mag(y_hat.squeeze(1), n_fft, hop))
    b = 20 * torch.log10(_stft_mag(y.squeeze(1), n_fft, hop))
    return ((a - b) ** 2).mean(1).sqrt().mean(-1)  # RMS over freq, mean over time


@torch.no_grad()
def evaluate(model, loader, device, content=None, speaker=None, sample_dir=None,
             n_samples=4, **qkw):
    model.eval()
    stft_loss, env_loss = MultiResSTFTLoss(), EnvelopeLoss()
    hist = IndexHistogram(model.groups, model.codebook_size)
    sums = {"stft": 0.0, "envelope": 0.0, "lsd_db": 0.0, "speaker_sim": 0.0, "wer": 0.0,
            "content_kl": 0.0}
    n_batches, n_items, saved = 0, 0, 0
    for x in loader:
        x = x.to(device)
        y, q = model(x, **qkw)
        hist.update(q["indices"])
        sums["stft"] += stft_loss(y, x).item()
        sums["envelope"] += env_loss(y, x).item()
        sums["lsd_db"] += log_spectral_distance(y, x).sum().item()
        if speaker is not None:
            sums["speaker_sim"] += speaker.similarity(y, x).sum().item()
        if content is not None:
            sums["content_kl"] += content.kl(y, x).item()
            refs, hyps = content.transcribe(x), content.transcribe(y)
            sums["wer"] += sum(word_error_rate(r, h) for r, h in zip(refs, hyps))
        if sample_dir is not None and saved < n_samples:
            import soundfile as sf
            Path(sample_dir).mkdir(parents=True, exist_ok=True)
            for i in range(min(n_samples - saved, x.shape[0])):
                sf.write(f"{sample_dir}/{saved:02d}_original.wav", x[i, 0].cpu().numpy(), SR)
                sf.write(f"{sample_dir}/{saved:02d}_decoded.wav", y[i, 0].cpu().numpy(), SR)
                saved += 1
        n_batches += 1
        n_items += x.shape[0]

    frames_per_sec = SR / model.hop
    out = {
        "quantizer": model.quantizer_name,
        "nominal_bps": model.nominal_bits_per_frame() * frames_per_sec,
        "entropy_bps": hist.entropy_bits_per_frame() * frames_per_sec,
        "codes_used": hist.codes_used(),
        "stft": sums["stft"] / n_batches,
        "envelope": sums["envelope"] / n_batches,
        "lsd_db": sums["lsd_db"] / n_items,
    }
    if speaker is not None:
        out["speaker_sim"] = sums["speaker_sim"] / n_items
    if content is not None:
        out["content_kl"] = sums["content_kl"] / n_batches
        out["wer_vs_original"] = sums["wer"] / n_items
    return out
