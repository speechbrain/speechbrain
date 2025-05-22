#!/usr/bin/env/python3
"""Run this script to perform inference with a trained model.

To run this script, execute the following (for example) on the command line:
> python verify_speaker.py sample1.wav sample2.wav results/4234/save

NOTE: If you changed the hparams in train.yaml, especially w.r.t STFT or
the verification model, make sure the hparams in inference.yaml are the same.

Authors
 * Peter Plantinga 2024
"""
import argparse

from speechbrain.inference.speaker import SpeakerRecognition

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample1")
    parser.add_argument("sample2")
    parser.add_argument("save_directory")
    args = parser.parse_args()

    verifier = SpeakerRecognition.from_hparams(
        source=".",
        hparams_file="inference.yaml",
        savedir=args.save_directory,
    )
    score, prediction = verifier.verify_files(args.sample1, args.sample2)
    if prediction:
        print("Model predicts SAME speaker")
    else:
        print("Model predicts DIFFERENT speakers")
