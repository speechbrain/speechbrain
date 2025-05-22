import os
import sys
import speechbrain as sb
import numpy as np
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

from sgmse_brain import SGMSEBrain 

import torch
import numpy as np
import torch.nn.functional as F
import speechbrain as sb

def pad_or_crop_waveform(sig, target_len, random_crop=True):
    """
    Pads or crops `sig` to have exactly `target_len` samples.
    """
    orig_len = sig.shape[-1]
    # Pad if too short
    if orig_len < target_len:
        needed = target_len - orig_len
        left_pad = needed // 2
        right_pad = needed - left_pad
        sig = F.pad(sig, (left_pad, right_pad), mode="constant")
    # Crop if too long
    elif orig_len > target_len:
        if random_crop:
            start = np.random.randint(0, orig_len - target_len)
        else:
            start = (orig_len - target_len) // 2
        sig = sig[..., start:start+target_len]
    return sig


def dataio_prep(hparams):
    """
    Prepare the datasets in SpeechBrain style. In addition to reading and padding/cropping, we now:
      - Read the audio files,
      - Normalize both noisy and clean signals with a common factor (based on hparams["normalize"]),
      - And (later in the Brain) the STFT and spec transformations will be applied.
    """
    target_len = hparams["num_samples"]     
    random_crop = hparams["random_crop"]

    # First, read the raw audio and pad/crop.
    @sb.utils.data_pipeline.takes("noisy_wav")
    @sb.utils.data_pipeline.provides("noisy_sig")
    def noisy_pipeline(noisy_wav):
        sig = sb.dataio.dataio.read_audio(noisy_wav)
        sig = pad_or_crop_waveform(sig, target_len, random_crop)
        return sig

    @sb.utils.data_pipeline.takes("clean_wav")
    @sb.utils.data_pipeline.provides("clean_sig")
    def clean_pipeline(clean_wav):
        sig = sb.dataio.dataio.read_audio(clean_wav)
        sig = pad_or_crop_waveform(sig, target_len, random_crop)
        return sig

    # normalize both signals by the same factor.
    @sb.utils.data_pipeline.takes("noisy_sig", "clean_sig")
    @sb.utils.data_pipeline.provides("noisy_sig", "clean_sig")
    def normalization_pipeline(noisy_sig, clean_sig):
        norm_mode = hparams.get("normalize", "noisy")
        if norm_mode == "noisy":
            normfac = noisy_sig.abs().max()
        elif norm_mode == "clean":
            normfac = clean_sig.abs().max()
        elif norm_mode == "not":
            normfac = 1.0
        else:
            raise ValueError("Invalid normalization mode")
        return noisy_sig / normfac, clean_sig / normfac

    # Build datasets
    datasets = {}
    for split in ["train", "valid", "test"]:
        json_path = hparams[f"{split}_annotation"]
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[noisy_pipeline, clean_pipeline, normalization_pipeline],
            output_keys=["id", "noisy_sig", "clean_sig"],
        )

    # Possibly sort or shuffle
    if hparams["sorting"] == "ascending" or hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="length", reverse=hparams["sorting"] == "descending"
        )
        hparams["train_dataloader_opts"]["shuffle"] = False

    return datasets


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    
    from voicebank_prepare import prepare_voicebank
    sb.utils.distributed.run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create datasets
    datasets = dataio_prep(hparams)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides
    )

    sgmse_brain = SGMSEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Train
    sgmse_brain.fit(
        epoch_counter=sgmse_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Evaluate
    sgmse_brain.evaluate(
        test_set=datasets["test"],
        max_key="valid_loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
