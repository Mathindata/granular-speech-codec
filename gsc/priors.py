"""Frozen pretrained models used as perceptual priors and as evaluation tools.

ContentPrior : wav2vec 2.0 with a CTC head ("facebook/wav2vec2-base-960h").
               Gives a probability vector over characters for every 20 ms
               frame, i.e. a membership vector per frame.
SpeakerPrior : WavLM x-vector model ("microsoft/wavlm-base-plus-sv").
               Gives one speaker embedding per clip.

Both need internet once to download weights (Kaggle: turn Internet on).
Gradients flow through the models to the decoded speech; the model weights
themselves stay frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize(x):
    """Per-utterance zero mean, unit variance, as the HF feature extractors do."""
    x = x.squeeze(1)
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-7)


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class ContentPrior(nn.Module):
    def __init__(self, name="facebook/wav2vec2-base-960h", model=None, processor=None):
        super().__init__()
        if model is None:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            model = Wav2Vec2ForCTC.from_pretrained(name)
            processor = Wav2Vec2Processor.from_pretrained(name)
        self.model = _freeze(model)
        self.processor = processor

    def train(self, mode=True):  # keep the frozen model in eval mode
        super().train(mode)
        self.model.eval()
        return self

    def log_probs(self, x):
        return F.log_softmax(self.model(_normalize(x)).logits, dim=-1)

    def kl(self, y_hat, y):
        """KL( p(y) || p(y_hat) ), averaged over frames. Draft Eq. (1)."""
        with torch.no_grad():
            lp = self.log_probs(y)
        lq = self.log_probs(y_hat)
        return (lp.exp() * (lp - lq)).sum(-1).mean()

    @torch.no_grad()
    def transcribe(self, x):
        ids = self.log_probs(x).argmax(-1)
        if self.processor is None:
            return [" ".join(map(str, r.tolist())) for r in ids]
        return self.processor.batch_decode(ids)


class SpeakerPrior(nn.Module):
    def __init__(self, name="microsoft/wavlm-base-plus-sv", model=None):
        super().__init__()
        if model is None:
            from transformers import WavLMForXVector
            model = WavLMForXVector.from_pretrained(name)
        self.model = _freeze(model)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def embed(self, x):
        return F.normalize(self.model(_normalize(x)).embeddings, dim=-1)

    def distance(self, y_hat, y):
        with torch.no_grad():
            e = self.embed(y)
        return (1 - (self.embed(y_hat) * e).sum(-1)).mean()

    @torch.no_grad()
    def similarity(self, y_hat, y):
        return (self.embed(y_hat) * self.embed(y)).sum(-1)


def word_error_rate(ref, hyp):
    """Word-level edit distance / number of reference words."""
    r, h = ref.split(), hyp.split()
    if not r:
        return float(len(h) > 0)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)
