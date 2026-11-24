"""Bit rate from the empirical entropy of the transmitted symbols.

Each group's index stream is treated as an independent source. Its entropy
H_g (bits per symbol) is the rate an ideal entropy coder (e.g. arithmetic
coding) would reach without context modeling. Sum over groups, times frames
per second, gives bits per second. The nominal rate G * log2(K) is the
fixed-length cost without any entropy coding.
"""

import torch


class IndexHistogram:
    def __init__(self, groups, codebook_size):
        self.counts = torch.zeros(groups, codebook_size, dtype=torch.float64)

    def update(self, indices):
        """indices: (B, G, T)."""
        g = indices.shape[1]
        flat = indices.detach().cpu().permute(1, 0, 2).reshape(g, -1)
        for i in range(g):
            self.counts[i] += torch.bincount(flat[i], minlength=self.counts.shape[1]).double()

    def entropy_bits_per_frame(self):
        p = self.counts / self.counts.sum(1, keepdim=True).clamp_min(1)
        h = -(p * torch.log2(p.clamp_min(1e-300))).sum(1)
        return h.sum().item()

    def codes_used(self):
        return (self.counts > 0).sum(1).tolist()
