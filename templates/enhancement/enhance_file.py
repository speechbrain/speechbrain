#!/usr/bin/env/python3
"""Run this script to perform inference with a trained model.

To run this script, execute the following (for example) on the command line:
> python enhance_file.py noisy_file.wav results/4234/save

NOTE: If you changed the hparams in train.yaml, especially w.r.t STFT or
the enhancement model, make sure the hparams in inference.yaml are the same.

Authors
 * Peter Plantinga 2024
"""
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
