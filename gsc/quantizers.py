"""Three quantizers with the same interface, so the codec body stays fixed.

Every quantizer splits the latent z (B, D, T) into G groups. Each group is
coded with one index from a codebook of size K, so the nominal rate is
G * log2(K) bits per frame for all three methods.

    FSQ   : finite scalar quantization (Mentzer et al., 2023). Each group has
            d dims with L levels, L**d = K. No codebook, rounding with a
            straight-through gradient.
    VQ    : standard vector quantization (van den Oord et al., 2017) with
            codebook and commitment losses and a straight-through gradient.
    Fuzzy : fuzzy prototype quantizer. Memberships come from the fuzzy
            c-means formula, decoding is the c-means degranulation
            (a membership-weighted mean of prototypes). During training the
            output moves from the soft value to the hard prototype, and the
            fuzzifier m moves toward 1, so at the end the code is discrete.

forward(z) returns a dict:
    z_q      : quantized latent, same shape as z
    indices  : (B, G, T) long, the transmitted symbols
    aux_loss : scalar, the method's own training loss (0 for FSQ)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_groups(z, groups):
    """(B, D, T) -> (B, G, T, D/G)."""
    b, d, t = z.shape
    return z.view(b, groups, d // groups, t).permute(0, 1, 3, 2)


def _merge_groups(z):
    """(B, G, T, d) -> (B, G*d, T)."""
    b, g, t, d = z.shape
    return z.permute(0, 1, 3, 2).reshape(b, g * d, t)


class FSQ(nn.Module):
    def __init__(self, latent_dim, groups=4, codebook_size=256):
        super().__init__()
        # choose d dims of L levels per group with L**d == K
        options = [(d, round(codebook_size ** (1 / d))) for d in range(1, 9)]
        options = [(d, L) for d, L in options if L >= 2 and L ** d == codebook_size]
        if not options:
            raise ValueError(f"codebook_size {codebook_size} is not L**d")
        # prefer a few dims with several levels, e.g. 256 -> 4 dims x 4 levels
        d, L = min(options, key=lambda p: abs(p[1] - 4))
        self.groups, self.dims, self.levels, self.K = groups, d, L, codebook_size
        self.proj_in = nn.Conv1d(latent_dim, groups * d, 1)
        self.proj_out = nn.Conv1d(groups * d, latent_dim, 1)
        half = (L - 1) / 2
        offset = 0.5 if L % 2 == 0 else 0.0
        self.register_buffer("half_range", torch.tensor(half))
        self.register_buffer("offset", torch.tensor(offset))
        self.register_buffer("shift", torch.tensor(math.atanh(offset / half)))
        self.register_buffer("radix", L ** torch.arange(d))

    def forward(self, z, **_):
        h = self.proj_in(z)
        bounded = torch.tanh(h + self.shift) * self.half_range - self.offset
        rounded = bounded + (torch.round(bounded) - bounded).detach()
        # integer level per dim, 0..L-1
        lv = (torch.round(bounded) + self.offset + self.half_range).long()
        lv = _split_groups(lv, self.groups)  # (B, G, T, d)
        indices = (lv * self.radix).sum(-1)
        z_q = self.proj_out(rounded / (self.half_range + self.offset))
        return {"z_q": z_q, "indices": indices, "aux_loss": z.new_zeros(())}


class _CodebookBase(nn.Module):
    def __init__(self, latent_dim, groups, codebook_size, code_dim):
        super().__init__()
        self.groups, self.K, self.code_dim = groups, codebook_size, code_dim
        self.proj_in = nn.Conv1d(latent_dim, groups * code_dim, 1)
        self.proj_out = nn.Conv1d(groups * code_dim, latent_dim, 1)
        self.codebook = nn.Parameter(torch.randn(groups, codebook_size, code_dim) * 0.1)
        self.register_buffer("usage", torch.zeros(groups, codebook_size))

    def _sq_dist(self, x):
        """x: (B, G, T, d) -> squared distances (B, G, T, K)."""
        cb = self.codebook  # (G, K, d)
        x2 = (x ** 2).sum(-1, keepdim=True)
        c2 = (cb ** 2).sum(-1)[None, :, None, :]
        xc = torch.einsum("bgtd,gkd->bgtk", x, cb)
        return (x2 - 2 * xc + c2).clamp_min(0.0)

    def _gather(self, idx):
        """idx: (B, G, T) -> prototypes (B, G, T, d)."""
        g = torch.arange(self.groups, device=idx.device)[None, :, None]
        return self.codebook[g, idx]

    @torch.no_grad()
    def _track_usage(self, idx):
        counts = F.one_hot(idx, self.K).sum(dim=(0, 2)).float()  # (G, K)
        self.usage.mul_(0.99).add_(counts, alpha=0.01)

    @torch.no_grad()
    def reset_dead_codes(self, z, threshold=1e-3):
        """Move rarely used prototypes onto random encoder outputs.

        Applied the same way to VQ and Fuzzy, so neither gets an advantage.
        Returns the number of codes reset.
        """
        x = _split_groups(self.proj_in(z), self.groups)  # (B, G, T, d)
        x = x.permute(1, 0, 2, 3).reshape(self.groups, -1, self.code_dim)
        n_reset = 0
        for g in range(self.groups):
            dead = (self.usage[g] < threshold).nonzero().flatten()
            if len(dead) == 0:
                continue
            pick = torch.randint(0, x.shape[1], (len(dead),), device=z.device)
            self.codebook.data[g, dead] = x[g, pick]
            self.usage[g, dead] = 1.0 / self.K
            n_reset += len(dead)
        return n_reset


class VQ(_CodebookBase):
    def __init__(self, latent_dim, groups=4, codebook_size=256, code_dim=8, beta=0.25):
        super().__init__(latent_dim, groups, codebook_size, code_dim)
        self.beta = beta

    def forward(self, z, **_):
        x = _split_groups(self.proj_in(z), self.groups)
        idx = self._sq_dist(x).argmin(-1)
        q = self._gather(idx)
        aux = F.mse_loss(q, x.detach()) + self.beta * F.mse_loss(x, q.detach())
        q_st = x + (q - x).detach()
        if self.training:
            self._track_usage(idx)
        return {"z_q": self.proj_out(_merge_groups(q_st)), "indices": idx, "aux_loss": aux}


class FuzzyVQ(_CodebookBase):
    """Fuzzy prototype quantizer (granulation + degranulation).

    Granulation (fuzzy c-means memberships, fuzzifier m > 1):
        u_i = 1 / sum_j (d_i / d_j)^(2/(m-1)),  d_i = ||x - v_i||
    Using squared distances D_i = d_i^2 this is
        u_i = softmax_i( -log(D_i) / (m - 1) ).
    Degranulation (c-means reconstruction):
        x_hat = sum_i u_i^m v_i / sum_i u_i^m
    Training loss: the c-means objective J = sum_i u_i^m D_i, which pulls the
    prototypes toward the data and the data toward the prototypes.

    Two schedule values are passed in at each step:
        m     : fuzzifier, annealed from m_start toward m_end (close to 1)
        alpha : 0 gives the soft output, 1 gives the hard prototype in the
                forward pass. The gradient always goes through the soft path,
                so no straight-through trick is needed for the encoder.
    """

    def __init__(self, latent_dim, groups=4, codebook_size=256, code_dim=8, eps=1e-6):
        super().__init__(latent_dim, groups, codebook_size, code_dim)
        self.eps = eps

    def memberships(self, sq_dist, m):
        return torch.softmax(-torch.log(sq_dist + self.eps) / (m - 1.0), dim=-1)

    def forward(self, z, m=1.5, alpha=1.0, **_):
        x = _split_groups(self.proj_in(z), self.groups)
        D = self._sq_dist(x)
        u = self.memberships(D, m)
        um = u ** m
        w = um / um.sum(-1, keepdim=True)
        soft = torch.einsum("bgtk,gkd->bgtd", w, self.codebook)
        idx = D.argmin(-1)
        hard = self._gather(idx)
        if self.training:
            out = soft + alpha * (hard - soft).detach()
            self._track_usage(idx)
        else:
            out = hard  # the decoder only ever receives the transmitted symbol
        aux = (um * D).sum(-1).mean()  # c-means objective J
        return {"z_q": self.proj_out(_merge_groups(out)), "indices": idx, "aux_loss": aux,
                "memberships": u}


def build_quantizer(name, latent_dim, groups, codebook_size, code_dim=8):
    name = name.lower()
    if name == "fsq":
        return FSQ(latent_dim, groups, codebook_size)
    if name == "vq":
        return VQ(latent_dim, groups, codebook_size, code_dim)
    if name == "fuzzy":
        return FuzzyVQ(latent_dim, groups, codebook_size, code_dim)
    raise ValueError(name)
