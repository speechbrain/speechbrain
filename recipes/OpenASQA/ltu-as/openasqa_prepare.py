"""
openasqa data preparation

Authors
 * Yingzhi Wang 2024
"""

import json
import logging
import os
import sys

import numpy as np
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import download_file

logger = logging.getLogger(__name__)

# to be added
PRETRAINED_TLTR_URL = "https://www.dropbox.com/scl/fi/nciysewp1cedc3ob8etqe/large-v1_ori.pth?rlkey=2ekg4x0wpqlzxt4it92kqas7c&st=l25hte4d&dl=0"

OPENASQA_CLASSIFICATION_JSON_URL = "https://www.dropbox.com/scl/fi/2h2pr4afssdeta6akywkg/classification.json?rlkey=gex199fa6wtbcjzonhbluzmn1&st=tdyyg801&dl=0"
OPENASQA_ALL_JSON_URL = "https://www.dropbox.com/scl/fi/6ycu9muolep2ox7wbi690/all.json?rlkey=iqxkkffreop903bnz23tlhdue&st=7kbkumdk&dl=0"

EVAL_ESC50_JSON_URL = "https://www.dropbox.com/scl/fi/ffn8k59rl77qjckr5clh4/eval_esc50.json?rlkey=gwc3qv6ve5g5n3jlc1g69y7t5&st=9scedbpo&dl=0"
EVAL_IEMOCAP_JSON_URL = "https://www.dropbox.com/scl/fi/9vhlkz9ly9tot07tpf479/eval_iemocap_emo.json?rlkey=rtcomi3n5c1r4djszzcynjo2q&st=tvvlwct8&dl=0"
EVAL_VOXCELEB_GENDER_JSON_URL = "https://www.dropbox.com/scl/fi/mkntiibo2c5rvrwts11wo/eval_voxceleb_gender.json?rlkey=zzsodtjtq3l3eut1qcv44zy7m&st=qhj2gnh2&dl=0"
EVAL_VOXCELEB_AGE_JSON_URL = "https://www.dropbox.com/scl/fi/8qijm9swdrgpqi09wg66k/eval_voxceleb_age.json?rlkey=8ln1as7qz5ic16xbw4qjkhw46&st=vvbv5svm&dl=0"
EVAL_LIBRISPEECH_TEST_CLEAN_JSON_URL = "https://www.dropbox.com/scl/fi/qd69q7m835u83b34a3djk/eval_librispeech_asr.json?rlkey=vog1m4m6mrhusa6zwzhxqfq6v&st=td53ckoe&dl=0"


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
    Prepare the necessary traning data and pre-trained model weights.

    Arguments
    ---------
    whisper_feature_folder : str
        path to save the pre-extracted whisper features.
    pretrained_tltr_path : str
        path to save the downloaded tltr model weights, a tltr pre-trained on
        AudioSet is used for training ltu-as.
    classification_json : str
        path to save the training annotations for classification tasks, this
        json file is used for stage 1 and 2.
    all_json : str
        path to save the training annotations for all the tasks, this json
        file is only used for stage 3.
    whisper_model :
        the whisper model used to extract acoustic features.
    average_pooling :
        average pooling function to reduct audio emd size.
    audioset_folder : str
        path to the AudioSet dataset.
    vggsound_folder : str
        path to the VGG-SOUND dataset.
    fsd50k_folder : str
        path to the FSD50k dataset.
    audiocaps_folder : str
        path to the AudioCaps dataset.
    clotho_folder : str
        path to the Clotho dataset.
    iemocap_folder : str
        path to the IEMOCAP dataset.
    libritts_folder : str
        path to the LibriTTS dataset.
    voxceleb2_folder : str
        path to the Voxceleb2 dataset.
    mosei_folder : str
        path to the CMU-MOSEI dataset.
    fma_folder : str
        path to the FMA dataset.
    """

    # Checks if this phase is already done (if so, skips it)
    if skip(classification_json, all_json):
        logger.info("Training preparation completed in previous run, skipping.")
        return

    logger.info(f"Creating {classification_json}, {all_json}")

    download_file(PRETRAINED_TLTR_URL, pretrained_tltr_path)
    download_file(OPENASQA_CLASSIFICATION_JSON_URL, classification_json)
    download_file(OPENASQA_ALL_JSON_URL, all_json)

    if not os.path.exists(whisper_feature_folder):
        os.makedirs(whisper_feature_folder)

    subsets = {
        "audioset": audioset_folder,
        "vggsound": audioset_folder,
        "fsd50k": fsd50k_folder,
        "audiocaps": audiocaps_folder,
        "clotho": clotho_folder,
        "iemocap": iemocap_folder,
        "libritts": libritts_folder,
        "voxceleb2": voxceleb2_folder,
        "mosei": mosei_folder,
        "fma": fma_folder,
    }

    for set in list(subsets.keys()):
        if subsets[set] is not None:
            os.makedirs(os.path.join(whisper_feature_folder, set))
        else:
            del subsets[set]

    assert len(subsets) > 0, "At least one dataset should be provided"
    logger.info(f"The provided datasets are {list(subsets.keys())}")

    for file in [classification_json, all_json]:
        with open(file, "r") as f:
            data = json.load(f)

        new_dict = {}

        for key in data.keys():
            audio_id = data[key]["audio_id"]
            feature_path = data[key]["feature_path"]

            for set in subsets.keys():
                if set in audio_id:
                    audio_id = audio_id.format(data_folder=subsets[set])
                    break

            if not os.path.exists(audio_id):
                logger.info(
                    f"{audio_id} does not exist, please check the audio path"
                )
                continue

            feature_path.format(whisper_feature_folder=whisper_feature_folder)
            if not os.path.exists(feature_path):
                extract_whisper_features(
                    whisper_model,
                    average_pooling,
                    audio_id,
                    feature_path,
                )
            # update new dictionary with selected items
            new_dict[key] = data[key]

        with open(file, "w") as fout:
            json.dump(new_dict, fout)


def prepare_openasqa_eval(
    whisper_feature_folder,
    eval_esc50_json,
    eval_iemocap_emo_json,
    eval_voxceleb_gender_json,
    eval_voxceleb_age_json,
    eval_librispeech_asr_json,
    valid_json,
    whisper_model,
    average_pooling,
    esc50_folder=None,
    iemocap_folder=None,
    voxceleb2_test_folder=None,
    librispeech_test_clean_folder=None,
):
    """
    Prepare the necessary evaluation data.

    Arguments
    ---------
    whisper_feature_folder : str
        path to save the pre-extracted whisper features for evaluation.
    eval_esc50_json : str
        path to save the evaluation annotation for ESC50 dataset.
    eval_iemocap_emo_json : str
        path to save the evaluation annotation for IEMOCAP test set.
    eval_voxceleb_gender_json : str
        path to save the evaluation annotation for Voxceleb2 test set (gender).
    eval_voxceleb_age_json : str
        path to save the evaluation annotation for Voxceleb2 test set (age).
    eval_librispeech_asr_json : str
        path to save the evaluation annotation for LibriSpeech test-clean set.
    valid_json : str
        path to save a tiny validation set used to follow the training process.
    whisper_model :
        the whisper model used to extract acoustic features.
    average_pooling :
        average pooling function to reduct audio emd size.
    esc50_folder : str
        path to the ESC50 dataset.
    iemocap_folder : str
        path to the IEMOCAP dataset.
    voxceleb2_test_folder : str
        path to the Voxceleb2-test dataset.
    librispeech_test_clean_folder : str
        path to the LibriSpeech test-clean dataset.
    """

    if skip(valid_json):
        logger.info(
            "Evaluation preparation completed in previous run, skipping."
        )
        return

    if not os.path.exists(whisper_feature_folder):
        os.makedirs(whisper_feature_folder)

    subsets = {
        "esc50": {
            "folder": esc50_folder,
            "annotation": eval_esc50_json,
            "url": EVAL_ESC50_JSON_URL,
        },
        "iemocap": {
            "folder": iemocap_folder,
            "annotation": eval_iemocap_emo_json,
            "url": EVAL_IEMOCAP_JSON_URL,
        },
        "voxceleb_gender": {
            "folder": voxceleb2_test_folder,
            "annotation": eval_voxceleb_gender_json,
            "url": EVAL_VOXCELEB_GENDER_JSON_URL,
        },
        "voxceleb_age": {
            "folder": voxceleb2_test_folder,
            "annotation": eval_voxceleb_age_json,
            "url": EVAL_VOXCELEB_AGE_JSON_URL,
        },
        "librispeech": {
            "folder": librispeech_test_clean_folder,
            "annotation": eval_librispeech_asr_json,
            "url": EVAL_LIBRISPEECH_TEST_CLEAN_JSON_URL,
        },
    }

    valid_dict = {}

    for set in subsets.keys():
        if subsets[set]["folder"] is not None:
            annotation_json = subsets[set]["annotation"]
            logger.info(f"Creating {annotation_json}")
            download_file(subsets[set]["url"], annotation_json)
            os.makedirs(
                os.path.join(whisper_feature_folder, "eval", set.split("_")[0])
            )  # create one folder for voxceleb_gender and voxceleb_age
            with open(annotation_json, "r") as f:
                data = json.load(f)

                new_dict = {}
                for key in data.keys():
                    audio_id = data[key]["audio_id"]
                    feature_path = data[key]["feature_path"]

                    audio_id = audio_id.format(
                        data_folder=subsets[set]["folder"]
                    )

                    if not os.path.exists(audio_id):
                        logger.info(
                            f"{audio_id} does not exist, please check the audio path"
                        )
                        continue

                    feature_path.format(
                        whisper_feature_folder=whisper_feature_folder
                    )
                    if not os.path.exists(feature_path):
                        extract_whisper_features(
                            whisper_model,
                            average_pooling,
                            audio_id,
                            feature_path,
                        )
                    # update new dictionary with selected items
                    new_dict[key] = data[key]

            # Add only one last item of each dataset into valid set
            valid_dict[key] = data[key]

            with open(annotation_json, "w") as fout:
                json.dump(new_dict, fout)
            logger.info(f"{annotation_json} successfully created.")

    with open(valid_json, "w") as fout:
        json.dump(valid_dict, fout)
        logger.info("A tiny valid set is also successfully created.")


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


def extract_whisper_features(
    model, pooling, audio_path, outputpath, sample_rate=16000
):
    """
    Extract whisper feature via a whisper model.

    Arguments
    ---------
    model :
        whisper model used for extraction
    pooling :
        average pooling function to reduct audio emd size.
    audio_path : str
        path to the target audio.
    outputpath : str
        path to a numpy file for saving the features.
    sample_rate : str
        resample the target audio to a designed sampling rate before extraction.
    """

    info = sb.dataio.dataio.read_audio_info(audio_path)
    sig = sb.dataio.dataio.read_audio(audio_path)

    if len(sig.shape) > 1:
        sig = torch.mean(sig, dim=1)

    resampled = torchaudio.transforms.Resample(
        info.sample_rate,
        sample_rate,
    )(sig)

    resampled = resampled.unsqueeze(0).to(torch.device("cuda"))
    output = model(resampled)[1:]
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
    hparams["whisper"] = hparams["whisper"].to(device=run_opts["device"])

    # whisper pad/trim all the audios to 10 seconds
    hparams["whisper"]._n_samples = hparams["sample_rate"]
    chunked_embed_positions_weight = torch.nn.Parameter(
        hparams["whisper"].model.encoder.embed_positions.weight[:500, :]
    )
    hparams["whisper"].model.encoder.embed_positions.weight = (
        chunked_embed_positions_weight
    )

    prepare_openasqa(
        whisper_feature_folder=hparams["whisper_feature_folder"],
        pretrained_tltr_path=hparams["pretrained_tltr_path"],
        classification_json=hparams["classification_json"],
        all_json=hparams["all_json"],
        whisper_model=hparams["whisper_model"],
        average_pooling=hparams["average_pooling"],
        audioset_folder=hparams["audioset_folder"],
        vggsound_folder=hparams["vggsound_folder"],
        fsd50k_folder=hparams["fsd50k_folder"],
        audiocaps_folder=hparams["audiocaps_folder"],
        clotho_folder=hparams["clotho_folder"],
        iemocap_folder=hparams["iemocap_folder"],
        libritts_folder=hparams["libritts_folder"],
        voxceleb2_folder=hparams["voxceleb2_folder"],
        mosei_folder=hparams["mosei_folder"],
        fma_folder=hparams["fma_folder"],
    )

    prepare_openasqa_eval(
        whisper_feature_folder=hparams["whisper_feature_folder"],
        eval_esc50_json=hparams["eval_esc50_json"],
        eval_iemocap_emo_json=hparams["eval_iemocap_emo_json"],
        eval_voxceleb_gender_json=hparams["eval_voxceleb_gender_json"],
        eval_voxceleb_age_json=hparams["eval_voxceleb_age_json"],
        eval_librispeech_asr_json=hparams["eval_librispeech_asr_json"],
        valid_json=hparams["valid_json"],
        whisper_model=hparams["whisper_model"],
        average_pooling=hparams["average_pooling"],
        esc50_folder=hparams["esc50_folder"],
        eval_iemocap_folder=hparams["eval_iemocap_folder"],
        voxceleb2_test_folder=hparams["voxceleb2_test_folder"],
        librispeech_test_clean_folder=hparams["librispeech_test_clean_folder"],
    )
