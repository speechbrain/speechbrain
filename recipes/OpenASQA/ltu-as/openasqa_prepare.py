"""
openasqa data preparation

Authors
 * Yingzhi Wang 2024
"""
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
import sys
import torch
import torchaudio
import numpy as np
import json
import shutil
import os
import logging
from hyperpyyaml import load_hyperpyyaml
logger = logging.getLogger(__name__)

# to be added
OPENASQA_CLASSIFICATION_JSON_URL = ""
OPENASQA_ALL_JSON_URL = ""
PRETRAINED_TLTR_URL = ""


def prepare_openasqa(
    whisper_feature_folder,
    pretrained_tltr_path,
    classification_json,
    all_json,
    whisper_model,
    average_pooling,
    audioset_folder=None,
    vggsound_folder=None,
    fsd50k_folder=None,
    audiocaps_folder=None,
    clotho_folder=None,
    iemocap_folder=None,
    libritts_folder=None,
    voxceleb2_folder=None,
    mosei_folder=None,
    fma_folder=None,
):
    """
    To be done
    download json files, tltr pretrained weights
    modfify audio path
    extract whisper features
    For the ready-to-use json ["dataset"] IEMOCAP and FMA should br lower-cased,
    audioset should be as_
    all the audioset based datasets should be merged
    """

    # Checks if this phase is already done (if so, skips it)
    if skip(classification_json, all_json):
        logger.info("Preparation completed in previous run, skipping.")
        return

    logger.info(
        f"Creating {classification_json}, {OPENASQA_ALL_JSON_URL}"
    )

    download_file(OPENASQA_CLASSIFICATION_JSON_URL, classification_json)
    download_file(OPENASQA_ALL_JSON_URL, all_json)
    download_file(PRETRAINED_TLTR_URL, pretrained_tltr_path)

    for file in [classification_json, all_json]:
        with open(file, 'r') as f:
            data = json.load(f)

        subsets = []
        if audioset_folder is not None:
            os.mkdir(os.path.join(whisper_feature_folder, "audioset"))
            subsets.append("as_")
        if vggsound_folder is not None:
            os.mkdir(os.path.join(whisper_feature_folder, "vgg-sound"))
            subsets.append("vggsound")
        if fsd50k_folder is not None:
            os.mkdir(os.path.join(whisper_feature_folder, "fsd50k"))
            subsets.append("fsd50k")
        if audiocaps_folder is not None:
            os.mkdir(os.path.join(whisper_feature_folder, "audiocaps"))
            subsets.append("audiocaps")
        if clotho_folder is not None:
            os.mkdir(os.path.join(whisper_feature_folder, "clotho"))
            subsets.append("clotho")
        if iemocap_folder is not None:
            os.mkdir(os.path.join(whisper_feature_folder, "iemocap"))
            subsets.append("iemocap")
        if libritts_folder is not None:
            os.mkdir(os.path.join(whisper_feature_folder, "libritts"))
            subsets.append("libritts")
        if voxceleb2_folder is not None:
            os.mkdir(os.path.join(whisper_feature_folder, "voxceleb2"))
            subsets.append("voxceleb")
        if mosei_folder is not None:
            os.mkdir(os.path.join(whisper_feature_folder, "mosei"))
            subsets.append("mosei")
        if fma_folder is not None:
            os.mkdir(os.path.join(whisper_feature_folder, "fma"))
            subsets.append("fma")

        assert len(subsets) > 0, "At least one dataset should be provided"

        new_dict = {}

        for key in data.keys():
            audio_id = data[key]["audio_id"]
            dataset = data[key]["dataset"]
            feature_path = data[key]["feature_path"]

            for subset in subsets:
                if subset in dataset:
                    if subset == "as_":
                        audio_id = audio_id.format(audioset_folder=audioset_folder)
                    elif subset == "vggsound":
                        audio_id = audio_id.format(vggsound_folder=vggsound_folder)
                    elif subset == "fsd50k":
                        audio_id = audio_id.format(fsd50k_folder=fsd50k_folder)
                    elif subset == "audiocaps":
                        audio_id = audio_id.format(audiocaps_folder=audiocaps_folder)
                    elif subset == "clotho":
                        audio_id = audio_id.format(clotho_folder=clotho_folder)
                    elif subset == "iemocap":
                        audio_id = audio_id.format(iemocap_folder=iemocap_folder)
                    elif subset == "libritts":
                        audio_id = audio_id.format(libritts_folder=libritts_folder)
                    elif subset == "voxceleb":
                        audio_id = audio_id.format(voxceleb2_folder=voxceleb2_folder)
                    elif subset == "mosei":
                        audio_id = audio_id.format(mosei_folder=mosei_folder)
                    elif subset == "fma":
                        audio_id = audio_id.format(fma_folder=fma_folder)
                    break

            if not os.path.exists(audio_id):
                logger.info(f"{audio_id} does not exist, please check the audio path")
                continue
            
            feature_path.format(whisper_feature_folder=whisper_feature_folder)
            extract_whisper_features(
                whisper_model,
                average_pooling,
                audio_id,
                feature_path,
            )
            # update new dictionary with selected items
            new_dict[key] = data[key]
        
        with open(file, 'w') as fout:
            json.dump(new_dict,fout)
    

def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def extract_whisper_features(model, pooling, audio_path, outputpath, sample_rate=16000):
    print("extracting whisper features")
    info = sb.dataio.dataio.read_audio_info(audio_path)
    sig = sb.dataio.dataio.read_audio(audio_path)

    if len(sig.shape) > 1:
        sig = torch.mean(sig, dim=1)

    resampled = torchaudio.transforms.Resample(
        info.sample_rate,
        sample_rate,
    )(sig)

    resampled = resampled.unsqueeze(0).to(torch.device("cuda"))
    output = model(resampled)[1: ]
    output = output.squeeze()
    output = pooling(output)
    output = output.half()
    output = output.detach().cpu().numpy()

    np.savez_compressed(outputpath, output)

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    #  Load pretrained whisper
    hparams["whisper"] = hparams["whisper"].to(
        device=run_opts["device"]
    )

    # whisper pad/trim all the audios to 10 seconds
    hparams["whisper"]._n_samples = hparams["sample_rate"]
    chunked_embed_positions_weight = torch.nn.Parameter(hparams["whisper"].model.encoder.embed_positions.weight[:500, :])
    hparams["whisper"].model.encoder.embed_positions.weight = chunked_embed_positions_weight
    
    prepare_openasqa(
        hparams["whisper_feature_folder"],
        hparams["pretrained_tltr_path"],
        hparams["classification_json"],
        hparams["all_json"],
        hparams["whisper_model"],
        hparams["average_pooling"],
        hparams["audioset_folder"],
        hparams["vggsound_folder"],
        hparams["fsd50k_folder"],
        hparams["audiocaps_folder"],
        hparams["clotho_folder"],
        hparams["iemocap_folder"],
        hparams["libritts_folder"],
        hparams["voxceleb2_folder"],
        hparams["mosei_folder"],
        hparams["fma_folder"],
    )