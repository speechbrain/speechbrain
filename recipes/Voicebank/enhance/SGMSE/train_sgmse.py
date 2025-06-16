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
    seg_frames   = hparams["segment_frames"] 
    n_fft        = hparams["n_fft"]
    hop_length   = hparams["hop_length"]
    target_len   = (seg_frames - 1) * hop_length  
    normalize    = hparams.get("normalize", "noisy")
    data_folder  = hparams["data_folder"]

    random_crop_train = hparams.get("random_crop_valid", True)
    random_crop_valid = hparams.get("random_crop_valid", False)
    random_crop_test  = hparams.get("random_crop_test",  False)

    def build_pipeline(random_crop_flag):
        @sb.utils.data_pipeline.takes("noisy_wav")
        @sb.utils.data_pipeline.provides("noisy_sig")
        def noisy_pipeline(noisy_wav):
            sig = sb.dataio.dataio.read_audio(noisy_wav)
            sig = pad_or_crop_waveform(sig, target_len, random_crop_flag)
            return sig

        @sb.utils.data_pipeline.takes("clean_wav")
        @sb.utils.data_pipeline.provides("clean_sig")
        def clean_pipeline(clean_wav):
            sig = sb.dataio.dataio.read_audio(clean_wav)
            sig = pad_or_crop_waveform(sig, target_len, random_crop_flag)
            return sig

        @sb.utils.data_pipeline.takes("noisy_sig", "clean_sig")
        @sb.utils.data_pipeline.provides("noisy_sig", "clean_sig")
        def norm_pipeline(noisy_sig, clean_sig):
            if normalize == "noisy":
                fac = noisy_sig.abs().max()
            elif normalize == "clean":
                fac = clean_sig.abs().max()
            else: # "not"
                fac = 1.0
            return noisy_sig / fac, clean_sig / fac

        return [noisy_pipeline, clean_pipeline, norm_pipeline]

    # create datasets 
    datasets = {}
    for split, rc in zip(
        ["train", "valid", "test"],
        [random_crop_train, random_crop_valid, random_crop_test],
    ):
        pipelines = build_pipeline(rc)
        json_path = hparams[f"{split}_annotation"]
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=pipelines,
            output_keys=["id", "noisy_sig", "clean_sig"],
        )

    # optional length sorting
    if hparams["sorting"] in ("ascending", "descending"):
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
