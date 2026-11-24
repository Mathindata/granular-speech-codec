"""Checks that run on CPU without downloads. Run: python -m pytest -q tests"""

import math

import pytest
import torch

from gsc.model import Codec
from gsc.quantizers import FuzzyVQ
from gsc.rate import IndexHistogram
from gsc.priors import word_error_rate


@pytest.mark.parametrize("qname", ["fsq", "vq", "fuzzy"])
@pytest.mark.parametrize("n", [16000, 16123])
def test_shapes_and_indices(qname, n):
    m = Codec(qname, channels=8, latent_dim=16, groups=4, codebook_size=256, code_dim=4)
    x = torch.randn(2, 1, n) * 0.1
    y, q = m(x, m=1.5, alpha=0.5)
    assert y.shape == x.shape
    frames = math.ceil(n / m.hop)
    assert q["indices"].shape == (2, 4, frames)
    assert 0 <= q["indices"].min() and q["indices"].max() < 256


@pytest.mark.parametrize("qname", ["fsq", "vq", "fuzzy"])
def test_gradient_reaches_encoder(qname):
    m = Codec(qname, channels=8, latent_dim=16, groups=2, codebook_size=16, code_dim=4)
    x = torch.randn(2, 1, 3200) * 0.1
    y, q = m(x, m=1.5, alpha=1.0)
    ((y - x) ** 2).mean().add(q["aux_loss"]).backward()
    g = m.encoder[0].weight.grad
    assert g is not None and g.abs().sum() > 0


def test_fcm_membership_formula():
    torch.manual_seed(0)
    fq = FuzzyVQ(latent_dim=4, groups=1, codebook_size=5, code_dim=3)
    D = torch.rand(1, 1, 7, 5) + 0.1  # squared distances
    mval = 1.7
    u = fq.memberships(D, mval)
    d = D.sqrt()
    explicit = 1.0 / ((d[..., :, None] / d[..., None, :]) ** (2 / (mval - 1))).sum(-1)
    assert torch.allclose(u, explicit, atol=1e-5)
    assert torch.allclose(u.sum(-1), torch.ones(1, 1, 7))


def test_fuzzy_becomes_hard_and_eval_uses_prototype():
    torch.manual_seed(0)
    fq = FuzzyVQ(latent_dim=4, groups=1, codebook_size=8, code_dim=4)
    D = torch.rand(1, 1, 5, 8) + 0.1
    u = fq.memberships(D, 1.01)
    assert torch.equal(u.argmax(-1), D.argmin(-1))
    assert (u.max(-1).values > 0.99).all()
    fq.eval()
    z = torch.randn(1, 4, 5)
    out = fq(z, m=1.5, alpha=0.0)  # schedule values must not matter in eval
    proto = fq.codebook[0, out["indices"][0, 0]]  # (T, d)
    expected = fq.proj_out(proto.T[None])
    assert torch.allclose(out["z_q"], expected, atol=1e-6)


def test_fsq_index_is_a_bijection():
    m = Codec("fsq", channels=8, latent_dim=16, groups=1, codebook_size=256)
    fsq = m.quantizer
    levels = torch.cartesian_prod(*[torch.arange(L) for L in fsq.levels])
    idx = (levels * fsq.radix).sum(-1)
    assert len(idx.unique()) == 256 and idx.max() == 255


def test_entropy():
    h = IndexHistogram(2, 16)
    h.update(torch.arange(16).repeat(3, 2, 2))  # (3, 2, 32), uniform
    assert abs(h.entropy_bits_per_frame() - 2 * 4.0) < 1e-9
    h2 = IndexHistogram(1, 16)
    h2.update(torch.zeros(1, 1, 10, dtype=torch.long))
    assert h2.entropy_bits_per_frame() == 0.0


def test_wer():
    assert word_error_rate("a b c", "a b c") == 0
    assert word_error_rate("a b c", "a x c") == pytest.approx(1 / 3)
    assert word_error_rate("a b c", "a c") == pytest.approx(1 / 3)


def test_priors_with_tiny_random_models():
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, WavLMConfig, WavLMForXVector
    from gsc.priors import ContentPrior, SpeakerPrior
    small = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                 intermediate_size=64, conv_dim=(16, 16), conv_stride=(5, 4),
                 conv_kernel=(10, 8), num_conv_pos_embeddings=16,
                 num_conv_pos_embedding_groups=2)
    cp = ContentPrior(model=Wav2Vec2ForCTC(Wav2Vec2Config(vocab_size=12, **small)))
    sp = SpeakerPrior(model=WavLMForXVector(WavLMConfig(
        tdnn_dim=(16, 16), tdnn_kernel=(3, 1), tdnn_dilation=(1, 1), xvector_output_dim=8,
        **small)))
    x = torch.randn(2, 1, 8000)
    y = (x + 0.3 * torch.randn_like(x)).requires_grad_(True)
    kl = cp.kl(y, x)
    assert kl.item() >= -1e-6
    assert cp.kl(x, x).item() == pytest.approx(0, abs=1e-5)
    dist = sp.distance(y, x)
    (kl + dist).backward()
    assert y.grad is not None and y.grad.abs().sum() > 0
    assert all(p.grad is None for p in cp.parameters())  # priors stay frozen
    assert len(cp.transcribe(x)) == 2


def test_fsq_mixed_levels():
    m = Codec("fsq", channels=8, latent_dim=16, groups=2, fsq_levels=[8, 6, 5])
    fsq = m.quantizer
    assert m.codebook_size == 240
    levels = torch.cartesian_prod(*[torch.arange(L) for L in fsq.levels])
    idx = (levels * fsq.radix).sum(-1)
    assert len(idx.unique()) == 240 and idx.max() == 239
    x = torch.randn(2, 1, 6400) * 0.5
    y, q = m(x)
    assert q["indices"].max() < 240 and q["indices"].min() >= 0
    # every level of every dim is reachable: push values to both extremes
    h = torch.linspace(-6, 6, 400)[None, None, :].repeat(1, 6, 1)
    b = torch.tanh(h + fsq.shift) * fsq.half_range - fsq.offset
    lv = (torch.round(b) - fsq.lowest)
    for c, L in enumerate([8, 6, 5] * 2):
        assert set(lv[0, c].long().tolist()) == set(range(L))


def test_eval_files_spread(tmp_path):
    import numpy as np, soundfile as sf
    from gsc.data import AudioFolder
    for spk in range(5):
        (tmp_path / f"s{spk}").mkdir()
        for k in range(10):
            sf.write(tmp_path / f"s{spk}" / f"{k}.flac", np.zeros(20000, np.float32), 16000)
    ds = AudioFolder(tmp_path, "all", max_files=5)
    assert sorted(f.parent.name for f in ds.files) == [f"s{i}" for i in range(5)]
