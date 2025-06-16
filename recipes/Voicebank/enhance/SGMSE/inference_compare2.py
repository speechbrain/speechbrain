#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Pytorch-Lightning SGMSE vs. SpeechBrain port on one utterance,
emitting detailed statistics after every processing step – including the
raw UNet/score-network output (NEW).

SpeechBrain side now runs through ScoreModel.enhance().
"""

# ───────────── CONFIG ──────────────────────────────────────────────
PL_CKPT  = "/export/home/1rochdi/sgmse/checkpoints/m1.ckpt"
SB_DIR   = "/export/home/1rochdi/speechbrain/results/SGMSE/save/" \
           "run_2025-05-26_22-21-09/CKPT+2025-05-28+14-26-09+00"
SB_SCORE = "score_model_patched.ckpt"
SB_YAML  = "/export/home/1rochdi/speechbrain/recipes/Voicebank/" \
           "enhance/SGMSE/hparams.yaml"

TEST_WAV = "/data/datasets/noisy-vctk-16k/noisy_testset_wav_16k/p257_191.wav"
OUT_DIR  = "/export/home/1rochdi/speechbrain/results/SGMSE/compare_inference"
# ───────────────────────────────────────────────────────────────────

import sys, os, warnings, math, json, pathlib
sys.path.insert(0, "/export/home/1rochdi/sgmse")        # original repo

import torch, soundfile as sf, numpy as np
from hyperpyyaml import load_hyperpyyaml
from speechbrain.lobes.models.sgmse.util.other import pad_spec
from sgmse.model import ScoreModel
from librosa import resample
from speechbrain import Stage                               # only for type
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── helper utils ─────────────────────────────────────────────
def tensor_stats(t):
    t = t.detach()
    if torch.is_complex(t):
        t = t.abs()
    return tuple(float(x) for x in (t.min(), t.max(), t.mean(), t.std()))

def report(name, A, B):
    amin, amax, amean, astd = tensor_stats(A)
    bmin, bmax, bmean, bstd = tensor_stats(B)
    diff  = (A - B).abs()
    dmax  = float(diff.max())
    dmean = float(diff.mean())
    rel   = dmax / (abs(amax) + 1e-12)

    print(f"{name:<10}  {tuple(A.shape)!s:<22} {str(A.dtype)[6:]:<6} "
          f"{amin:+.2e}/{amax:+.2e}  {amean:+.2e}±{astd:.2e}   | "
          f"{bmin:+.2e}/{bmax:+.2e}  {bmean:+.2e}±{bstd:.2e}   "
          f"Δmax={dmax:.2e}  Δmean={dmean:.2e}  rel={rel:5.1%}")

def get_window(kind, n):
    if kind == "sqrthann":
        return torch.sqrt(torch.hann_window(n, periodic=True))
    return torch.hann_window(n, periodic=True)

# ─── 1. load checkpoints ─────────────────────────────────────
print("→ loading Lightning checkpoint")
pl = ScoreModel.load_from_checkpoint(PL_CKPT,
                                     map_location=device).eval().to(device)

print("→ loading SpeechBrain checkpoint")
with open(SB_YAML) as f:
    hps = load_hyperpyyaml(f)
sbm = hps["modules"]["score_model"].to(device)

ckpt = torch.load(os.path.join(SB_DIR, SB_SCORE), map_location=device)
state = ckpt.get("state_dict", ckpt)
missing, _ = sbm.load_state_dict(state, strict=False)
assert not missing, f"SB model missing keys: {missing}"
if "ema" in ckpt:
    sbm.ema.load_state_dict(ckpt["ema"]); sbm.store_ema()

for m in (pl, sbm):
    m.t_eps = 0.03;  m.sde.N = 30;  m.eval()

# ─── 2. common spec helpers ─────────────────────────────────
n_fft, hop, sr  = hps["n_fft"], hps["hop_length"], hps["sample_rate"]
w               = get_window(hps["window_type"], n_fft).to(device)
trans, fac      = hps["transform_type"], hps["spec_factor"]
exp_e           = hps.get("spec_abs_exponent", 1.0)
pad_mode        = "reflection"

def fwd(z):
    if trans == "exponent":
        if exp_e != 1: z = z.abs() ** exp_e * torch.exp(1j*z.angle())
        return z * fac
    if trans == "log":
        return torch.log1p(z.abs()) * torch.exp(1j*z.angle()) * fac
    return z

def back(z):
    if trans == "exponent":
        z = z / fac
        if exp_e != 1: z = z.abs()**(1/exp_e) * torch.exp(1j*z.angle())
        return z
    if trans == "log":
        return torch.expm1((z/fac).abs()) * torch.exp(1j*z.angle())
    return z

# ─── 3. load noisy utterance ────────────────────────────────
wav, sr_orig = sf.read(TEST_WAV)
if sr_orig != sr:
    wav = resample(wav, sr_orig, sr)
y_wav = torch.tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)
norm  = y_wav.abs().max();  y_wav = y_wav / norm

# ─── 4. helper: PL sampler (unchanged) ──────────────────────
def sample_pl(model, Yp):
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    sampler = model.get_pc_sampler("reverse_diffusion", "ald",
                                   y=Yp, N=model.sde.N,
                                   corrector_steps=1, snr=0.5,
                                   intermediate=False)
    out_pad, _ = sampler()
    return out_pad

# ─── 5. shared preprocessing up to padded spec ──────────────
store = {}
store["Y"]       = torch.stft(y_wav, n_fft, hop, center=True,
                              return_complex=True, window=w)
store["Ytf"]     = fwd(store["Y"])
store["Y4"]      = store["Ytf"].unsqueeze(1)
store["Ypad_in"] = pad_spec(store["Y4"], mode=pad_mode)  # (B,1,F_pad,T_pad)

# --- sanity-check raw ScoreModel output ---------------------
with torch.no_grad():
    t_test = torch.full((store["Ypad_in"].size(0),), 0.5,
                        dtype=torch.float32, device=device)
    x_t = y = store["Ypad_in"]

    pl_raw = pl (x_t, y, t_test)
    sb_raw = sbm(x_t, y, t_test)
    print("median magnitude ratio =", (pl_raw.abs() /
                                       (sb_raw.abs()+1e-12)).median())

store["raw_net_pl"] = pl_raw
store["raw_net_sb"] = sb_raw

# ─── 6. full sampling / enhancement ─────────────────────────
with torch.no_grad():
    # PL: direct sampler (as before)
    store["pl_out_pad"] = sample_pl(pl, store["Ypad_in"])

    # SB: go through ScoreModel.enhance()
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    store["sb_out_pad"] = sbm.enhance(
        store["Ypad_in"],
        sampler_type=getattr(sbm.sde, "sampler_type", "pc"),
        predictor="reverse_diffusion",
        corrector="ald",
        N=sbm.sde.N, corrector_steps=1, snr=0.5
    )

# ─── 7. back-to-wave -------------------------------------------------
F, T = store["Y"].shape[-2:]
store["pl_sample"] = store["pl_out_pad"][:, :, :F, :T].squeeze(1)
store["sb_sample"] = store["sb_out_pad"][:, :, :F, :T].squeeze(1)

store["pl_cplx"] = back(store["pl_sample"])
store["sb_cplx"] = back(store["sb_sample"])

store["pl_wav"] = torch.istft(store["pl_cplx"], n_fft, hop, center=True,
                              window=w, length=y_wav.shape[-1]) * norm
store["sb_wav"] = torch.istft(store["sb_cplx"], n_fft, hop, center=True,
                              window=w, length=y_wav.shape[-1]) * norm

# ─── 8. detailed stats print-out ------------------------------------
print("\n   name        shape                  dtype   "
      "PL-stats(min/max)   mean±std    |   SB-stats   mean±std      Δ-stats")
print("─"*130)

def rpt(lbl, a, b): report(lbl, store[a], store[b])
rpt("Y",        "Y",          "Y")
rpt("Ytf",      "Ytf",        "Ytf")
rpt("Y4",       "Y4",         "Y4")
rpt("Ypad_in",  "Ypad_in",    "Ypad_in")
rpt("RawNet",   "raw_net_pl", "raw_net_sb")
rpt("Samp_pad", "pl_out_pad", "sb_out_pad")
rpt("Sample",   "pl_sample",  "sb_sample")
rpt("iSTFT",    "pl_cplx",    "sb_cplx")
rpt("Waveform", "pl_wav",     "sb_wav")

# ─── 9. write wavs ---------------------------------------------------
sf.write(f"{OUT_DIR}/enh_pl.wav", store["pl_wav"].cpu().numpy().squeeze(), sr)
sf.write(f"{OUT_DIR}/enh_sb.wav", store["sb_wav"].cpu().numpy().squeeze(), sr)
print(f"\nEnhanced wavs saved to {OUT_DIR}")