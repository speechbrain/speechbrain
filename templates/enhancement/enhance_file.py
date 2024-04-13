import argparse

from speechbrain.inference.enhancement import SpectralMaskEnhancement

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("noisy_file")
    parser.add_argument("save_directory")
    parser.add_argument("--enhanced_file", default="enhanced.wav")
    args = parser.parse_args()

    enhancer = SpectralMaskEnhancement.from_hparams(
        source=".",
        hparams_file="inference.yaml",
        savedir=args.save_directory,
    )
    enhancer.enhance_file(args.noisy_file, args.enhanced_file)
