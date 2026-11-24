"""Training criteria.

Waveform-level terms (no pretrained model needed):
    MultiResSTFTLoss : spectral convergence + log-magnitude L1 at several
                       resolutions. It ignores phase, in line with the draft's
                       point that small phase changes are not heard.
    EnvelopeLoss     : L1 between smoothed log-energy profiles (draft Fig. 8).
                       Differentiable, and avoids the Hilbert-envelope issue
                       raised in the draft review.

Perceptual priors (pretrained, frozen; see priors.py):
    content : KL( p(S) || p(S_hat) ) between frame-level CTC posteriors.
              This is the draft's multinomial-prior KL, Eq. (1).
    speaker : 1 - cos( e(S), e(S_hat) ) between speaker embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stft_mag(x, n_fft, hop):
    win = torch.hann_window(n_fft, device=x.device)
    s = torch.stft(x, n_fft, hop, window=win, return_complex=True)
    return s.abs().clamp_min(1e-7)


class MultiResSTFTLoss(nn.Module):
    def __init__(self, resolutions=((512, 128), (1024, 256), (256, 64))):
        super().__init__()
        self.resolutions = resolutions

    def forward(self, y_hat, y):
        y_hat, y = y_hat.squeeze(1), y.squeeze(1)
        total = 0.0
        for n_fft, hop in self.resolutions:
            a, b = _stft_mag(y_hat, n_fft, hop), _stft_mag(y, n_fft, hop)
            sc = torch.linalg.norm(b - a, dim=(1, 2)) / torch.linalg.norm(b, dim=(1, 2))
            mag = F.l1_loss(torch.log(a), torch.log(b))
            total = total + sc.mean() + mag
        return total / len(self.resolutions)


def log_energy(x, win=320, hop=160, eps=1e-5):
    """x: (B, 1, N) -> (B, frames) smoothed log-energy in dB."""
    e = F.avg_pool1d(x ** 2, win, hop)
    return 10 * torch.log10(e + eps).squeeze(1)


class EnvelopeLoss(nn.Module):
    def __init__(self, win=320, hop=160):
        super().__init__()
        self.win, self.hop = win, hop

    def forward(self, y_hat, y):
        return F.l1_loss(log_energy(y_hat, self.win, self.hop),
                         log_energy(y, self.win, self.hop)) / 10.0
