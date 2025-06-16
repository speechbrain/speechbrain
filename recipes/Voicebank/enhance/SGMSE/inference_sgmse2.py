import os, torch, soundfile as sf, numpy as np
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.lobes.models.sgmse.util.other import pad_spec

# -----------------------
# Configuration variables
# -----------------------
CHECKPOINT_FOLDER = "/export/home/1rochdi/speechbrain/results/SGMSE/save/run_2025-06-12_15-29-13/CKPT+2025-06-15+10-23-22+00"
SCORE_MODEL_FILE  = "score_model.ckpt"   # checkpoint to load
CONFIG_PATH       = "/export/home/1rochdi/speechbrain/recipes/Voicebank/enhance/SGMSE/hparams.yaml"
NUM_FILES         = 10

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def dbg(tag, x):
    x = x.detach().cpu()
    shape = tuple(x.shape)

    if torch.is_complex(x):
        mag = x.abs()
        mn, mx = mag.min(), mag.max()
        print(f"{tag:<18s} shape={shape!s:<15}  (complex) |mag| min={mn:+.3e}  max={mx:+.3e}")
    else:
        mn, mx = x.min(), x.max()
        print(f"{tag:<18s} shape={shape!s:<15}  min={mn:+.3e}  max={mx:+.3e}")

def get_stft_window(t, n):
    if t == "sqrthann":
        return torch.sqrt(torch.hann_window(n, periodic=True))
    if t == "hann":
        return torch.hann_window(n, periodic=True)
    raise NotImplementedError(t)

def do_stft(sig, w, n_fft, hop):
    return torch.stft(sig, n_fft=n_fft, hop_length=hop, center=True,
                      return_complex=True, window=w)

def do_istft(spec, w, n_fft, hop, length=None):
    return torch.istft(spec, n_fft=n_fft, hop_length=hop, center=True,
                       window=w, length=length)

def spec_fwd(spec, ttype, fac, e):
    if ttype == "exponent":
        if e != 1.0:
            mag = spec.abs() ** e
            spec = mag * torch.exp(1j * spec.angle())
        return spec * fac
    if ttype == "log":
        mag = torch.log1p(spec.abs())
        spec = mag * torch.exp(1j * spec.angle())
        return spec * fac
    return spec

def spec_back(spec, ttype, fac, e):
    if ttype == "exponent":
        spec = spec / fac
        if e != 1.0:
            mag = spec.abs() ** (1.0 / e)
            spec = mag * torch.exp(1j * spec.angle())
        return spec
    if ttype == "log":
        spec = spec / fac
        mag  = torch.expm1(spec.abs())
        return mag * torch.exp(1j * spec.angle())
    return spec

# ------------------------------------------------------------
def main():
    # ---------- hparams ----------
    with open(CONFIG_PATH) as f:
        hparams = load_hyperpyyaml(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = hparams["modules"]["score_model"].to(device)

    # ---------- checkpoint ----------
    ckpt_path = os.path.join(CHECKPOINT_FOLDER, SCORE_MODEL_FILE)
    raw = torch.load(ckpt_path, map_location=device)

    state = raw["state_dict"] if "state_dict" in raw else \
            raw["score_model"] if "score_model" in raw else raw

    miss, unexp = model.load_state_dict(state, strict=False)
    tot = sum(p.numel() for p in model.parameters())
    ok  = tot - sum(np.prod(model.state_dict()[k].shape) for k in miss)

    print(f"\n✓ loaded {ok}/{tot} parameters")
    print("   missing :", len(miss))
    print("   unexpected :", len(unexp))

    if "ema" in raw:
        model.ema.load_state_dict(raw["ema"])
        model.ema.copy_to(model.dnn.parameters())
        print("✓ EMA swapped in (decay =", model.ema.decay, ")")

    model.eval()

    # ---- quick weight sanity -------------------------------------------------
    first_w = next(p for p in model.parameters() if p.requires_grad)
    print("First weight tensor mean/std:",
        first_w.mean().item(), first_w.std().item())
    # --------------------------------------------------------------------------

    # ---------- STFT config ----------
    n_fft           = hparams["n_fft"]
    hop_length      = hparams["hop_length"]
    window_type     = hparams["window_type"]
    transform_type  = hparams["transform_type"]
    spec_factor     = hparams["spec_factor"]
    spec_exp        = hparams.get("spec_abs_exponent", 1.0)
    sr              = hparams["sample_rate"]
    window          = get_stft_window(window_type, n_fft).to(device)

    enhanced_dir = hparams["inference_folder"]
    os.makedirs(enhanced_dir, exist_ok=True)

    # ---------- test set ----------
    test_json = hparams["test_annotation"]
    test_set = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=test_json,
        replacements={"data_root": hparams["data_folder"]},
        output_keys=["id", "noisy_wav", "clean_wav"],
    )

    # ---------- loop ----------
    for i, sample in enumerate(test_set):
        if i >= NUM_FILES:
            break
        uid   = sample["id"]
        nfile = sample["noisy_wav"]; cfile = sample["clean_wav"]

        y_wav, sr_n = sf.read(nfile); x_wav, sr_c = sf.read(cfile)
        assert sr_n == sr_c == sr, "sample-rate mismatch"

        # ---------- 1. torch & norm ----------
        y = torch.tensor(y_wav, dtype=torch.float32, device=device).unsqueeze(0)
        x = torch.tensor(x_wav, dtype=torch.float32, device=device).unsqueeze(0)

        norm_mode = hparams.get("normalize", "noisy")
        normfac   = y.abs().max() if norm_mode=="noisy" else \
                    x.abs().max() if norm_mode=="clean" else 1.0
        y = y / normfac
        dbg(f"[{uid}] y_norm", y)

        # ---------- 2. STFT & transform ----------
        Y = do_stft(y, window, n_fft, hop_length)
        Y = spec_fwd(Y, transform_type, spec_factor, spec_exp)
        Y4 = Y.unsqueeze(1)
        F_orig, T_orig = Y.shape[-2:]

        Y_pad = pad_spec(Y4)
        dbg(f"[{uid}] Y_pad", Y_pad)

        # ---------- 3. model.enhance ----------
        with torch.no_grad():
            samp_pad = model.enhance(
                Y_pad,
                sampler_type=getattr(model.sde, "sampler_type", "pc"),
                predictor="reverse_diffusion",
                corrector="ald",
                N=hparams.get("inference_N", 30),
                corrector_steps=hparams.get("corrector_steps", 1),
                snr=hparams.get("snr", 0.5),
            )
        dbg(f"[{uid}] sample_pad", samp_pad)

        samp = samp_pad[:, :, :F_orig, :T_orig].squeeze(1)
        dbg(f"[{uid}] sample", samp)

        # ---------- 4. back transform & iSTFT ----------
        samp_cplx = spec_back(samp, transform_type, spec_factor, spec_exp)
        wav = do_istft(samp_cplx, window, n_fft, hop_length, length=y.shape[-1])
        wav = wav * normfac
        dbg(f"[{uid}] enh_wav", wav)

        enh = wav.squeeze().cpu().numpy()

        # ---------- write wavs ----------
        sf.write(os.path.join(enhanced_dir, f"{uid}_enhanced.wav"), enh, sr)
        sf.write(os.path.join(enhanced_dir, f"{uid}_clean.wav"),   x_wav, sr)
        sf.write(os.path.join(enhanced_dir, f"{uid}_noisy.wav"),   y_wav, sr)

        print(f"→ {uid} done\n")

    print("=== inference finished ===")

if __name__ == "__main__":
    main()