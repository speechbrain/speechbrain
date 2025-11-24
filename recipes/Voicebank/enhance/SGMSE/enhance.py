"""
Single-file or batch speech enhancement with SGMSE.
Single file:
python enhance.py --run_dir /path/to/run  noisy.wav

Whole directory:
python enhance.py --run_dir /path/to/run  /path/to/noisy_dir
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from train import SGMSEBrain

from speechbrain.utils.checkpoints import Checkpointer


# Helpers
def is_audio_file(path):
    return path.suffix.lower() in {".wav", ".flac", ".ogg"}


def collect_audio_files(src):
    return [p for p in src.iterdir() if p.is_file() and is_audio_file(p)]


def main():
    parser = argparse.ArgumentParser(
        description="Run SGMSE enhancement (torchaudio I/O)"
    )
    parser.add_argument(
        "--run_dir",
        "-r",
        type=Path,
        required=True,
        help="Path to the trained run directory (the folder that "
        "contains hyperparams.yaml and checkpoints/).",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a noisy audio file OR a directory of audio files.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        sys.exit(f"--run_dir '{run_dir}' does not exist.")

    hparams_file = run_dir / "hyperparams.yaml"
    checkpoints_dir = run_dir / "checkpoints"

    with open(hparams_file, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)

    target_sr = hparams["sample_rate"]
    inference_dir = Path(run_dir / "enhanced_inference")
    inference_dir.mkdir(parents=True, exist_ok=True)

    modules = hparams["modules"]
    brain = SGMSEBrain(
        modules=modules,
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer=Checkpointer(
            checkpoints_dir=checkpoints_dir,
            recoverables={"score_model": modules["score_model"]},
        ),
    )
    brain.setup_inference()  # loads latest checkpoint, ema ...

    # Enhancement routine
    def enhance_file(noisy_path, dst_dir):
        wav, sr = torchaudio.load(noisy_path)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)

        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        with torch.no_grad():
            wav = wav.to(brain.device)
            enhanced = brain.enhance(wav).cpu()

        out_path = dst_dir / f"{noisy_path.stem}_enhanced{noisy_path.suffix}"
        torchaudio.save(out_path.as_posix(), enhanced, target_sr, format="wav")
        return out_path

    src = args.input.expanduser().resolve()

    if src.is_file():
        if not is_audio_file(src):
            sys.exit(f"{src} is not a supported audio file.")
        out_path = enhance_file(src, inference_dir)
        print(f"Enhanced file written to {out_path}")

    elif src.is_dir():
        files = collect_audio_files(src)
        if not files:
            sys.exit(f"{src} contains no enhanceable audio files.")

        batch_out_dir = inference_dir / f"{src.name}_enhanced"
        batch_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Enhancing {len(files)} file(s) > {batch_out_dir}")
        for idx, fpath in enumerate(files, 1):
            out_path = enhance_file(fpath, batch_out_dir)
            print(f"[{idx}/{len(files)}] > {out_path}")
    else:
        sys.exit(f"{src} is neither a file nor a directory.")


if __name__ == "__main__":
    main()
