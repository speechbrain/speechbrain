import os
import torch
import soundfile as sf
from hyperpyyaml import load_hyperpyyaml

from speechbrain.lobes.models.sgmse.util.other import pad_spec
import speechbrain as sb

# -----------------------
# Configuration variables
# -----------------------
CHECKPOINT_FOLDER = "/export/home/1rochdi/speechbrain/recipes/Voicebank/enhance/SGMSE/results/SGMSE/save/CKPT+2025-03-12+14-24-36+00"
SCORE_MODEL_FILE = "score_model.ckpt"  # Contains raw model weights
CONFIG_PATH = "/export/home/1rochdi/speechbrain/recipes/Voicebank/enhance/SGMSE/hparams.yaml"
NUM_FILES = 10  # Number of test files to process

def get_stft_window(window_type, n_fft):
    """Build a window tensor for STFT."""
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(n_fft, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(n_fft, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")

def do_stft(sig, window, n_fft, hop_length):
    """Compute the short-time Fourier transform (STFT)."""
    return torch.stft(
        sig,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
        return_complex=True,
        window=window,
    )

def do_istft(spec, window, n_fft, hop_length, length=None):
    """Compute the inverse STFT."""
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
        window=window,
        length=length,
    )

def spec_fwd(spec_cplx, transform_type, factor, exponent):
    """Forward spectral transform (e.g. log, exponent) on the complex spectrogram."""
    if transform_type == "exponent":
        if exponent != 1.0:
            mag = spec_cplx.abs() ** exponent
            phase = spec_cplx.angle()
            spec_cplx = mag * torch.exp(1j * phase)
        spec_cplx *= factor

    elif transform_type == "log":
        mag = torch.log1p(spec_cplx.abs())
        phase = spec_cplx.angle()
        spec_cplx = mag * torch.exp(1j * phase)
        spec_cplx *= factor

    elif transform_type == "none":
        pass

    return spec_cplx

def spec_back(spec_cplx, transform_type, factor, exponent):
    """Inverse spectral transform to revert the forward transform."""
    if transform_type == "exponent":
        spec_cplx = spec_cplx / factor
        if exponent != 1.0:
            mag = spec_cplx.abs() ** (1.0 / exponent)
            phase = spec_cplx.angle()
            spec_cplx = mag * torch.exp(1j * phase)

    elif transform_type == "log":
        spec_cplx = spec_cplx / factor
        mag = torch.expm1(spec_cplx.abs())
        phase = spec_cplx.angle()
        spec_cplx = mag * torch.exp(1j * phase)

    elif transform_type == "none":
        pass

    return spec_cplx

def main():
    # Load hyperparameters from the YAML config file.
    with open(CONFIG_PATH) as fin:
        hparams = load_hyperpyyaml(fin)

    # Instantiate the ScoreModel and load checkpoint.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = hparams["modules"]["score_model"].to(device)

    # Build the path to score_model.ckpt.
    score_model_path = os.path.join(CHECKPOINT_FOLDER, SCORE_MODEL_FILE)
    if not os.path.isfile(score_model_path):
        raise FileNotFoundError(f"Could not find {score_model_path}")

    # Load raw model weights.
    state_dict = torch.load(score_model_path, map_location=device)
    # Try loading directly, if necessary, try state_dict["score_model"]
    try:
        model.load_state_dict(state_dict)
        print(f"Loaded raw state_dict from {score_model_path}")
    except Exception:
        model.load_state_dict(state_dict["score_model"])
        print(f"Loaded 'score_model' key from {score_model_path}")

    model.eval()
    print(f"Model weights loaded from {score_model_path}")

    # Prepare STFT window and transforms from your config.
    n_fft = hparams["n_fft"]
    hop_length = hparams["hop_length"]
    window_type = hparams["window_type"]
    transform_type = hparams["transform_type"]
    spec_factor = hparams["spec_factor"]
    spec_abs_exponent = hparams.get("spec_abs_exponent", 1.0)
    sample_rate = hparams["sample_rate"]
    window = get_stft_window(window_type, n_fft).to(device)

    # Create output folder for enhanced waveforms.
    enhanced_folder = hparams["enhanced_folder"]
    os.makedirs(enhanced_folder, exist_ok=True)

    # Get the test annotation file from hparams.
    test_json = hparams["test_annotation"]

    # Load test set from json
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=test_json,
        replacements={"data_root": hparams["data_folder"]},
        output_keys=["id", "noisy_wav", "clean_wav"]
    )

    # Enhance each file (up to NUM_FILES)
    for i, sample in enumerate(test_data):
        if i >= NUM_FILES:
            break

        uttid = sample["id"]
        noisy_file = sample["noisy_wav"]
        clean_file = sample["clean_wav"]

        # Read the audio
        y_wav, sr_noisy = sf.read(noisy_file)
        x_wav, sr_clean = sf.read(clean_file)
        assert sr_noisy == sr_clean == sample_rate, "Mismatched sample rates."

        # Convert to torch tensor 
        y_wav_torch = torch.tensor(y_wav, dtype=torch.float32, device=device).unsqueeze(0)

        # Normalize
        norm_factor = y_wav_torch.abs().max().item()
        if norm_factor < 1e-8:
            norm_factor = 1.0
        y_wav_torch = y_wav_torch / norm_factor

        # Compute STFT
        y_stft = do_stft(y_wav_torch, window, n_fft, hop_length)  # (1, F, T) complex

        # Apply forward spectral transform
        y_stft = spec_fwd(y_stft, transform_type, spec_factor, spec_abs_exponent)

        # Add channel dimension (B=1, C=1, F, T)
        y_stft = y_stft.unsqueeze(1)

        # Pad to match the backbone's expected input shape.
        y_stft = pad_spec(y_stft)

        # Enhance using the model's enhance() method.
        with torch.no_grad():
            enh_stft = model.enhance(y_stft, N=model.sde.N)  # (1,1,F,T)

        # Remove channel dimension.
        enh_stft = enh_stft.squeeze(1)  # (1, F, T)

        # Apply inverse spectral transform.
        enh_stft = spec_back(enh_stft, transform_type, spec_factor, spec_abs_exponent)

        # Compute inverse STFT.
        enh_wav_torch = do_istft(enh_stft, window, n_fft, hop_length, length=None)

        # Renormalize.
        enh_wav_torch = enh_wav_torch * norm_factor
        enh_wav = enh_wav_torch.squeeze().cpu().numpy()

        # Write out the results.
        enh_name = f"{uttid}_enhanced.wav"
        clean_name = f"{uttid}_clean.wav"
        enh_path = os.path.join(enhanced_folder, enh_name)
        clean_path = os.path.join(enhanced_folder, clean_name)

        sf.write(enh_path, enh_wav, sample_rate)
        sf.write(clean_path, x_wav, sample_rate)
        print(f"Enhanced {uttid}: wrote {enh_path} and {clean_path}")

    print("Inference finished.")

if __name__ == "__main__":
    main()