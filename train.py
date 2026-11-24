"""Train one codec.

Examples
--------
Quick check on synthetic signals (CPU, about a minute):
    python train.py --synthetic --steps 200 --quantizer fuzzy --out runs/smoke

LibriSpeech on a Kaggle GPU:
    python train.py --data_dir /kaggle/working/LibriSpeech/train-clean-100 \
        --quantizer fuzzy --steps 40000 --w_content 1 --w_speaker 1 --out runs/fuzzy_256

Run the same command with --quantizer vq and --quantizer fsq for the baselines.
Change --codebook_size (16, 64, 256, 1024) to sweep the rate.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gsc.data import AudioFolder, Synthetic
from gsc.evaluation import evaluate
from gsc.losses import EnvelopeLoss, MultiResSTFTLoss
from gsc.model import Codec


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir")
    p.add_argument("--eval_dir", help="defaults to a held-out 2%% of data_dir")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="runs/run")
    p.add_argument("--quantizer", choices=["fsq", "vq", "fuzzy"], default="fuzzy")
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--code_dim", type=int, default=8)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--segment", type=float, default=1.0, help="training clip, seconds")
    p.add_argument("--eval_segment", type=float, default=4.0)
    p.add_argument("--eval_items", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=2)
    # loss weights
    p.add_argument("--w_stft", type=float, default=1.0)
    p.add_argument("--w_env", type=float, default=1.0)
    p.add_argument("--w_aux", type=float, default=1.0)
    p.add_argument("--w_content", type=float, default=0.0)
    p.add_argument("--w_speaker", type=float, default=0.0)
    # curriculum, as in the draft's training procedure (section 3.4)
    p.add_argument("--content_start", type=int, default=0, help="step the content prior turns on")
    p.add_argument("--speaker_start", type=int, default=0, help="step the speaker prior turns on")
    # fuzzy schedule
    p.add_argument("--m_start", type=float, default=1.5)
    p.add_argument("--m_end", type=float, default=1.05)
    p.add_argument("--anneal_frac", type=float, default=0.5,
                   help="fraction of steps over which m and alpha reach their final values")
    p.add_argument("--reset_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def schedule(step, total, a):
    """Fuzzifier m and hardness alpha at a given step."""
    r = min(1.0, step / max(1, a.anneal_frac * total))
    m = a.m_start + r * (a.m_end - a.m_start)
    return {"m": m, "alpha": r}


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = 16000

    if a.synthetic:
        train_set = Synthetic(n=10**6, segment=int(a.segment * sr))
        eval_set = Synthetic(n=a.eval_items, segment=int(a.eval_segment * sr), seed=1, fixed=True)
    else:
        train_set = AudioFolder(a.data_dir, "train", int(a.segment * sr))
        if a.eval_dir:
            eval_set = AudioFolder(a.eval_dir, "all", int(a.eval_segment * sr), max_files=a.eval_items)
        else:
            eval_set = AudioFolder(a.data_dir, "eval", int(a.eval_segment * sr), max_files=a.eval_items)
    train_loader = DataLoader(train_set, a.batch, shuffle=not a.synthetic,
                              num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    eval_loader = DataLoader(eval_set, 8, shuffle=False, num_workers=a.workers)

    model = Codec(a.quantizer, a.channels, latent_dim=a.latent_dim, groups=a.groups,
                  codebook_size=a.codebook_size, code_dim=a.code_dim).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, "
          f"nominal rate {model.nominal_bits_per_frame() * sr / model.hop:.0f} bit/s")

    content = speaker = None
    if a.w_content > 0:
        from gsc.priors import ContentPrior
        content = ContentPrior().to(device)
    if a.w_speaker > 0:
        from gsc.priors import SpeakerPrior
        speaker = SpeakerPrior().to(device)

    opt = torch.optim.AdamW(model.parameters(), a.lr, betas=(0.8, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05)
    stft_loss, env_loss = MultiResSTFTLoss(), EnvelopeLoss()
    log = open(out / "train_log.jsonl", "a")

    step, t0, it = 0, time.time(), iter(train_loader)
    model.train()
    while step < a.steps:
        try:
            x = next(it)
        except StopIteration:
            it = iter(train_loader)
            x = next(it)
        x = x.to(device)
        qkw = schedule(step, a.steps, a)
        y, q = model(x, **qkw)

        terms = {"stft": stft_loss(y, x), "env": env_loss(y, x), "aux": q["aux_loss"]}
        loss = a.w_stft * terms["stft"] + a.w_env * terms["env"] + a.w_aux * terms["aux"]
        if content is not None and step >= a.content_start:
            terms["content"] = content.kl(y, x)
            loss = loss + a.w_content * terms["content"]
        if speaker is not None and step >= a.speaker_start:
            terms["speaker"] = speaker.distance(y, x)
            loss = loss + a.w_speaker * terms["speaker"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1

        reset = 0
        if (a.quantizer != "fsq" and step % a.reset_every == 0 and step < 0.8 * a.steps):
            reset = model.quantizer.reset_dead_codes(q["z"].detach())

        if step % a.log_every == 0 or step == 1:
            rec = {"step": step, "loss": loss.item(), **{k: v.item() for k, v in terms.items()},
                   **qkw, "reset": reset, "sec": round(time.time() - t0, 1)}
            print(json.dumps(rec))
            log.write(json.dumps(rec) + "\n")
            log.flush()

        if step % a.eval_every == 0 or step == a.steps:
            res = evaluate(model, eval_loader, device, content, speaker,
                           sample_dir=out / f"samples_{step}", **schedule(a.steps, a.steps, a))
            res["step"] = step
            print("EVAL", json.dumps(res))
            with open(out / "eval.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")
            torch.save({"model": model.state_dict(), "args": vars(a), "step": step},
                       out / "checkpoint.pt")
            model.train()


if __name__ == "__main__":
    main()
