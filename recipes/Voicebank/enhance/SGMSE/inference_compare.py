#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Pytorch-Lightning SGMSE vs. SpeechBrain port on one utterance,
emitting detailed statistics after every processing step – including the
raw UNet/score-network output (NEW).
"""

# ───────────── CONFIG ──────────────────────────────────────────────
PL_CKPT  = "/export/home/1rochdi/sgmse/checkpoints/m1.ckpt"
SB_DIR   = "/export/home/1rochdi/speechbrain/results/SGMSE/save/run_2025-05-26_22-21-09/CKPT+2025-05-28+14-26-09+00"
SB_SCORE = "score_model_patched.ckpt"
SB_YAML  = "/export/home/1rochdi/speechbrain/recipes/Voicebank/enhance/SGMSE/hparams.yaml"

TEST_WAV = "/data/datasets/noisy-vctk-16k/noisy_testset_wav_16k/p257_191.wav"
OUT_DIR  = "/export/home/1rochdi/speechbrain/results/SGMSE/compare_inference"
# ───────────────────────────────────────────────────────────────────

# ensure original repo is importable ────────────────────────────────
import sys, os, warnings, pathlib, math, json
sys.path.insert(0, "/export/home/1rochdi/sgmse")

import torch, soundfile as sf, numpy as np
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.lobes.models.sgmse.util.other import pad_spec
from sgmse.model import ScoreModel
from librosa import resample
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── helper functions ─────────────────────────────────────────────
def tensor_stats(t):
    """Return (min,max,mean,std) on magnitude if complex."""
    t = t.detach()
    if torch.is_complex(t):
        t = t.abs()
    return tuple(float(x) for x in (t.min(), t.max(), t.mean(), t.std()))

def report(name, A, B):
    """Print detailed stats for PL tensor A vs SB tensor B."""
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
    return (torch.sqrt(torch.hann_window(n, periodic=True))
            if kind == "sqrthann" else torch.hann_window(n, periodic=True))

# ─── 1. load checkpoints ──────────────────────────────────────────
print("→ loading Lightning checkpoint")
pl = ScoreModel.load_from_checkpoint(PL_CKPT, map_location=device).eval().to(device)

print("→ loading SpeechBrain checkpoint")
with open(SB_YAML) as f:
    hps = load_hyperpyyaml(f)
sbm = hps["modules"]["score_model"].to(device)

ckpt = torch.load(os.path.join(SB_DIR, SB_SCORE), map_location=device)
state = ckpt.get("state_dict", ckpt)
missing, _ = sbm.load_state_dict(state, strict=False)
assert not missing, f"SB model missing keys: {missing}"
if "ema" in ckpt:
    sbm.ema.load_state_dict(ckpt["ema"])
    sbm.store_ema()

for m in (pl, sbm):
    m.t_eps = 0.03
    m.sde.N = 30
    m.eval()

# ─── 2. common audio → spec preprocessing ────────────────────────
n_fft, hop, sr = hps["n_fft"], hps["hop_length"], hps["sample_rate"]
w              = get_window(hps["window_type"], n_fft).to(device)
trans, fac     = hps["transform_type"], hps["spec_factor"]
exp_e          = hps.get("spec_abs_exponent", 1.0)
pad_mode       = "reflection"

def fwd(z):
    if trans == "exponent":
        if exp_e != 1:
            z = z.abs() ** exp_e * torch.exp(1j * z.angle())
        return z * fac
    if trans == "log":
        return torch.log1p(z.abs()) * torch.exp(1j * z.angle()) * fac
    return z

def back(z):
    if trans == "exponent":
        z = z / fac
        if exp_e != 1:
            z = z.abs() ** (1 / exp_e) * torch.exp(1j * z.angle())
        return z
    if trans == "log":
        return torch.expm1((z / fac).abs()) * torch.exp(1j * z.angle())
    return z

# ─── 3. load & normalise utterance ────────────────────────────────
wav, sr_orig = sf.read(TEST_WAV)
if sr_orig != sr:
    wav = resample(wav, orig_sr=sr_orig, target_sr=sr)
y = torch.tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)
norm = y.abs().max()
y = y / norm

# ─── 4. identical sampler helper ─────────────────────────────────
def sample(model, Yp):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    sampler = model.get_pc_sampler(
        predictor_name="reverse_diffusion",
        corrector_name="ald",
        y=Yp,
        N=model.sde.N,
        corrector_steps=1,
        snr=0.5,
        intermediate=False,
    )
    s_pad, _ = sampler()
    return s_pad

# ─── 5. forward path for both models ─────────────────────────────
store = {}
store["Y"]       = torch.stft(
    y, n_fft, hop, center=True, return_complex=True, window=w
)
store["Ytf"]     = fwd(store["Y"])
store["Y4"]      = store["Ytf"].unsqueeze(1)
store["Ypad_in"] = pad_spec(store["Y4"], mode=pad_mode)

# --- direct score-network check ----------------------------------
with torch.no_grad():
    # arbitrary diffusion time in (0, 1)
    t_test = torch.full(
        (store["Ypad_in"].size(0),),   # one τ per batch element
        0.5,
        dtype=torch.float32,
        device=device,
    )

    x_t = store["Ypad_in"]            # use the padded spec itself as x_t
    y    = store["Ypad_in"]           # conditioning mixture (same here)

    # Lightning vs. SpeechBrain
    pl_raw = pl (x_t, y, t_test)      # ← PL implementation
    sb_raw = sbm(x_t, y, t_test)      # ← SB port
    scale = (pl_raw.abs() / (sb_raw.abs() + 1e-12)).median()
    print("median magnitude ratio =", scale)

store["raw_net_pl"] = pl_raw
store["raw_net_sb"] = sb_raw

with torch.no_grad():
    # ... your existing code to build x_t, y, t_test ...

    # 1) compute “raw” ScoreModel output (this is score = c_skip*x_t + c_out*F)
    pl_raw = pl(x_t, y, t_test)
    sb_raw = sbm(x_t, y, t_test)
    print("median(|pl_raw|/|sb_raw|) =", (pl_raw.abs()/(sb_raw.abs()+1e-12)).median())

    # 2) …but let’s grab F = the DNN’s “un-scaled” output BEFORE c_out or c_skip is applied.
    #    We know that in forward(), the first line under “ncsnpp_v2” is:
    #        F = self.dnn(self._c_in(t)*x_t, self._c_in(t)*y, t)
    #    So let’s recompute that exact same “F” for both models:
    c_in_pl = pl._c_in(t_test)       # shape (batch,1,1,1)
    c_in_sb = sbm._c_in(t_test)
    # feed into the DNN itself (bypass any c_out or c_skip)
    F_pl = pl.dnn(c_in_pl * x_t, c_in_pl * y, t_test)
    F_sb = sbm.dnn(c_in_sb * x_t, c_in_sb * y, t_test)

    # Compare these two “F”s directly:
    diff = (F_pl - F_sb).abs()
    print("   * After DNN (before any c_out/c_skip):")
    print("     max|F_pl - F_sb| =", float(diff.max()), 
          "  mean|F_pl - F_sb| =", float(diff.mean()),
          "  relΔ =", float(diff.max()) / (float(F_pl.abs().max())+1e-12))

    # 3) Now also check “F” after dividing by network_scaling, if that branch was active:
    if pl.network_scaling == "1/sigma":
        std = pl.sde._std(t_test)[:,None,None,None]
        F_pl_scaled = F_pl / std
        F_sb_scaled = F_sb / std   # sbm._std(t) should be identical if both use same SDE
    elif pl.network_scaling == "1/t":
        F_pl_scaled = F_pl / t_test[:,None,None,None]
        F_sb_scaled = F_sb / t_test[:,None,None,None]
    else:
        F_pl_scaled = F_pl
        F_sb_scaled = F_sb

    diff2 = (F_pl_scaled - F_sb_scaled).abs()
    print("   * After network_scaling (same [F/std or F/t] if applicable):")
    print("     max|scaledF_pl - scaledF_sb| =", float(diff2.max()),
          "  mean|scaledF_pl - scaledF_sb| =", float(diff2.mean()))

    # 4) Finally check “score = c_skip*x_t + c_out*scaledF”:
    c_skip_pl = pl._c_skip(t_test)
    c_out_pl  = pl._c_out(t_test)
    c_skip_sb = sbm._c_skip(t_test)
    c_out_sb  = sbm._c_out(t_test)

    score_pl = c_skip_pl * x_t + c_out_pl * F_pl_scaled
    score_sb = c_skip_sb * x_t + c_out_sb * F_sb_scaled
    dscore = (score_pl - score_sb).abs()
    print("   * After c_skip + c_out (this should exactly match pl_raw / sb_raw):")
    print("     max|score_pl - score_sb| =", float(dscore.max()),
          "  mean|score_pl - score_sb| =", float(dscore.mean()),
          "  relΔ =", float(dscore.max())/(float(score_pl.abs().max()) + 1e-12))
# ---------------------------------------------------------------

F, T = store["Y"].shape[-2:]

with torch.no_grad():
    store["pl_out_pad"] = sample(pl, store["Ypad_in"])
    store["sb_out_pad"] = sample(sbm, store["Ypad_in"])

store["pl_sample"] = store["pl_out_pad"][:, :, :F, :T].squeeze(1)
store["sb_sample"] = store["sb_out_pad"][:, :, :F, :T].squeeze(1)

store["pl_cplx"] = back(store["pl_sample"])
store["sb_cplx"] = back(store["sb_sample"])

store["pl_wav"] = (
    torch.istft(store["pl_cplx"], n_fft, hop, center=True, window=w, length=y.shape[-1]) * norm
)
store["sb_wav"] = (
    torch.istft(store["sb_cplx"], n_fft, hop, center=True, window=w, length=y.shape[-1]) * norm
)

# ─── 6. detailed comparison print-out ────────────────────────────
print(
    "\n   name        shape                  dtype   "
    "PL-stats(min/max)   mean±std    |   SB-stats   mean±std      Δ-stats"
)
print("─" * 130)

def report_pair(label, a_key, b_key):
    report(label, store[a_key], store[b_key])

report_pair("Y",          "Y",             "Y")
report_pair("Ytf",        "Ytf",           "Ytf")
report_pair("Y4",         "Y4",            "Y4")
report_pair("Ypad_in",    "Ypad_in",       "Ypad_in")
# ▸▸▸ NEW row with raw-network diff
report_pair("RawNet",     "raw_net_pl",    "raw_net_sb")
# ▸▸▸ END NEW
report_pair("Samp_pad",   "pl_out_pad",    "sb_out_pad")
report_pair("Sample",     "pl_sample",     "sb_sample")
report_pair("iSTFT",      "pl_cplx",       "sb_cplx")
report_pair("Waveform",   "pl_wav",        "sb_wav")

# ─── 7. write wavs ────────────────────────────────────────────────
sf.write(os.path.join(OUT_DIR, "enh_pl.wav"), store["pl_wav"].squeeze().cpu().numpy(), sr)
sf.write(os.path.join(OUT_DIR, "enh_sb.wav"), store["sb_wav"].squeeze().cpu().numpy(), sr)
print(f"\nEnhanced wavs saved to {OUT_DIR}")