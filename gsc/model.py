"""Fully convolutional speech codec: raw waveform in, raw waveform out.

Encoder: strided 1-D convolutions with dilated residual units. With
strides (2, 4, 5, 8) one latent frame covers 320 samples, so 16 kHz speech
gives 50 frames per second. At 4 groups x 256 codes (32 bits per frame)
the nominal rate is 1.6 kbit/s.

Decoder: the mirror image with transposed convolutions.
"""

import math

import torch
import torch.nn as nn

from .quantizers import build_quantizer


class ResUnit(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(ch, ch, 7, dilation=dilation, padding=3 * dilation),
            nn.ELU(),
            nn.Conv1d(ch, ch, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class EncBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.net = nn.Sequential(
            ResUnit(cin, 1), ResUnit(cin, 3), ResUnit(cin, 9),
            nn.ELU(),
            nn.Conv1d(cin, cout, 2 * stride, stride=stride, padding=stride // 2 + stride % 2),
        )

    def forward(self, x):
        return self.net(x)


class DecBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            nn.ConvTranspose1d(cin, cout, 2 * stride, stride=stride,
                               padding=stride // 2 + stride % 2, output_padding=stride % 2),
            ResUnit(cout, 1), ResUnit(cout, 3), ResUnit(cout, 9),
        )

    def forward(self, x):
        return self.net(x)


class Codec(nn.Module):
    def __init__(self, quantizer="fuzzy", channels=32, strides=(2, 4, 5, 8),
                 latent_dim=64, groups=4, codebook_size=256, code_dim=8, fsq_levels=None):
        super().__init__()
        self.strides = tuple(strides)
        self.hop = 1
        for s in strides:
            self.hop *= s
        chs = [channels * 2 ** i for i in range(len(strides) + 1)]

        enc = [nn.Conv1d(1, chs[0], 7, padding=3)]
        for i, s in enumerate(strides):
            enc.append(EncBlock(chs[i], chs[i + 1], s))
        enc += [nn.ELU(), nn.Conv1d(chs[-1], latent_dim, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)

        dec = [nn.Conv1d(latent_dim, chs[-1], 7, padding=3)]
        for i, s in reversed(list(enumerate(strides))):
            dec.append(DecBlock(chs[i + 1], chs[i], s))
        dec += [nn.ELU(), nn.Conv1d(chs[0], 1, 7, padding=3), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)

        self.quantizer = build_quantizer(quantizer, latent_dim, groups, codebook_size, code_dim,
                                         fsq_levels)
        self.quantizer_name = quantizer
        self.groups = groups
        self.codebook_size = getattr(self.quantizer, "K", codebook_size)

    def nominal_bits_per_frame(self):
        return self.groups * math.log2(self.codebook_size)

    def encode(self, x):
        """x: (B, 1, N) with N a multiple of hop -> latent (B, D, N/hop)."""
        return self.encoder(x)

    def forward(self, x, **qkw):
        n = x.shape[-1]
        pad = (-n) % self.hop
        if pad:
            x = nn.functional.pad(x, (0, pad))
        z = self.encode(x)
        q = self.quantizer(z, **qkw)
        y = self.decoder(q["z_q"])[..., :n]
        q["z"] = z
        return y, q
