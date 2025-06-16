#!/usr/bin/env python
import os, torch, soundfile as sf, numpy as np
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.lobes.models.sgmse.util.other import pad_spec

# ───────────────────────────────────────────────────────────────
# ★★  SET THESE FOUR PATHS  ★★
GOOD_CKPT = "/export/home/1rochdi/speechbrain/results/SGMSE/save/run_2025-04-07_21-44-45/CKPT+2025-04-09+13-34-18+00/score_model.ckpt"
GOOD_YAML = "/export/home/1rochdi/speechbrain/recipes/Voicebank/enhance/SGMSE/hparams.yaml"      # the YAML you trained with

BAD_CKPT  = "/export/home/1rochdi/speechbrain/results/SGMSE/save/run_2025-04-20_20-25-48/CKPT+2025-04-22+11-59-54+00/score_model.ckpt"
BAD_YAML  = "/export/home/1rochdi/speechbrain/recipes/Voicebank/enhance/SGMSE/hparams.yaml"       # YAML of the noisy run

TEST_WAV  = "/data/datasets/noisy-vctk-16k/noisy_testset_wav_16k/p257_107.wav"            # any 1-second clip
# ───────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHORT_LEN = 16000         # 1 s at 16 kHz is enough for inspection
torch.set_printoptions(precision=3, sci_mode=True)

# --------------------------------------------------------------
def load_model(ckpt_path, yaml_path):
    with open(yaml_path) as f:
        hp = load_hyperpyyaml(f)

    model = hp["modules"]["score_model"].to(DEVICE)
    raw   = torch.load(ckpt_path, map_location=DEVICE)

    # weights
    net_state = raw.get("state_dict") or raw.get("score_model") or raw
    model.load_state_dict(net_state, strict=False)

    # EMA (if present)
    has_ema = "ema" in raw
    if has_ema:
        model.ema.load_state_dict(raw["ema"])
        model.ema.copy_to(model.dnn.parameters())

    model.eval()
    return model, hp, raw, has_ema

def global_info(name, raw, has_ema):
    print(f"\n▶ {name.upper()} CHECKPOINT INFO")
    print("  keys :", list(raw.keys())[:10], "…")
    print("  current_epoch :", raw.get("current_epoch"))
    print("  global_step   :", raw.get("global_step"))
    if has_ema:
        print("  EMA num_updates :", raw['ema']['num_updates'])
    else:
        print("  EMA absent")

def stat_table(name, mod_good, mod_bad):
    print(f"\n▶ WEIGHT-WISE STATISTICS ({name})")
    print(f"{'layer':45s}  {'µg':>8}  {'σg':>8} | {'µb':>8}  {'σb':>8} |  ||Δ||₂")
    for (n_g, p_g), (n_b, p_b) in zip(mod_good.named_parameters(),
                                      mod_bad.named_parameters()):
        if p_g.ndim == 0:
            continue
        μg, σg = p_g.mean().item(), p_g.std().item()
        μb, σb = p_b.mean().item(), p_b.std().item()
        diff   = (p_g - p_b).pow(2).sum().sqrt().item()
        print(f"{n_g:45s}  {μg:+8.2e} {σg:+8.2e} | {μb:+8.2e} {σb:+8.2e} | {diff:+8.2e}")

def hparam_diff(raw_g, raw_b):
    print("\n▶ H-PARAMETERS DIFFERENCE (only differing keys)")
    hp_g = raw_g.get("hyper_parameters") or raw_g.get("hparams") or {}
    hp_b = raw_b.get("hyper_parameters") or raw_b.get("hparams") or {}
    for k in sorted(set(hp_g)|set(hp_b)):
        if hp_g.get(k) != hp_b.get(k):
            print(f"{k:25s}: good={hp_g.get(k)!r}  bad={hp_b.get(k)!r}")

# --------------------------------------------------------------
def quick_forward(model, hp, tag):
    wav, sr = sf.read(TEST_WAV); wav = wav[:SHORT_LEN]
    wav = torch.tensor(wav, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    win = torch.sqrt(torch.hann_window(hp["n_fft"], periodic=True)).to(DEVICE) \
          if hp["window_type"]=="sqrthann" else torch.hann_window(hp["n_fft"], periodic=True).to(DEVICE)

    spec = torch.stft(wav, n_fft=hp["n_fft"], hop_length=hp["hop_length"],
                      center=True, return_complex=True, window=win)

    ttype, fac = hp["transform_type"], hp["spec_factor"]
    exp = hp.get("spec_abs_exponent", 1.0)
    if ttype=="exponent":
        if exp!=1.0: spec = spec.abs()**exp * torch.exp(1j*spec.angle())
        spec = spec*fac
    elif ttype=="log":
        spec = torch.log1p(spec.abs())*torch.exp(1j*spec.angle())*fac

    spec4 = pad_spec(spec.unsqueeze(1))

    with torch.no_grad():
        out_pad = model.enhance(spec4, N=10)      # quick sampler
    out = out_pad.squeeze(1)[:, :spec.shape[-2], :spec.shape[-1]]

    if ttype=="exponent":
        out = out/fac
        if exp!=1.0: out = out.abs()**(1/exp)*torch.exp(1j*out.angle())
    elif ttype=="log":
        out = out/fac
        out = torch.expm1(out.abs())*torch.exp(1j*out.angle())

    wav_hat = torch.istft(out, n_fft=hp["n_fft"], hop_length=hp["hop_length"],
                          center=True, window=win, length=wav.shape[-1])

    print(f"[{tag}]  |spec in|²={spec.abs().pow(2).mean():.3e}  "
          f"|spec out|²={out.abs().pow(2).mean():.3e}  "
          f"wav pow={wav_hat.pow(2).mean():.3e}")

# --------------------------------------------------------------
def main():
    good, hp_g, raw_g, ema_g = load_model(GOOD_CKPT, GOOD_YAML)
    bad,  hp_b, raw_b, ema_b = load_model(BAD_CKPT,  BAD_YAML)

    global_info("good", raw_g, ema_g)
    global_info("bad",  raw_b, ema_b)

    stat_table("μ/σ & diff", good, bad)
    hparam_diff(raw_g, raw_b)

    print("\n▶ QUICK FORWARD-PASS ENERGY")
    quick_forward(good, hp_g, "GOOD")
    quick_forward(bad,  hp_b, "BAD")

if __name__ == "__main__":
    main()