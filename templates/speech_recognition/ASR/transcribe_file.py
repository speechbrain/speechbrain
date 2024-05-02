#!/usr/bin/env/python3
"""Run this script to perform inference with a trained model.

To run this script, execute the following (for example) on the command line:
> python transcribe_file.py speech_file.wav results/4234/save

NOTE: If you changed the hparams in train.yaml, especially w.r.t the ASR model,
make sure the hparams in inference.yaml are the same.

Authors
 * Peter Plantinga 2024
"""
import argparse
import os

from speechbrain.inference.ASR import EncoderDecoderASR


def link_file(filename, source_dir, target_dir):
    """Create a symbolic link for file between two directories

    Arguments
    ---------
    filename : str
        The name of the file to link
    source_dir : str
        The directory containing the source file
    target_dir : str
        The directory to put the link into
    """
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)
    if os.path.lexists(target_path):
        os.remove(target_path)
    os.symlink(source_path, target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("speech_file")
    parser.add_argument("save_directory")
    args = parser.parse_args()

    # Link ASR and normalizer in save folder alongside LM and tokenizer.
    # This is done so that trained models and pretrained models are available
    # in the same directory to be loaded by the inference class.
    source_dir = os.path.abspath(args.save_directory)
    target_dir = os.path.dirname(source_dir)
    link_file("model.ckpt", source_dir, target_dir)
    link_file("normalizer.ckpt", source_dir, target_dir)

    transcriber = EncoderDecoderASR.from_hparams(
        source=".",
        hparams_file="inference.yaml",
        savedir=target_dir,
    )
    text = transcriber.transcribe_file(args.speech_file)

    print(f"Text present in file {args.speech_file}:")
    print(text)
