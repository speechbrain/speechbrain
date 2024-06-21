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
PRETRAINED_TLTR_URL = ""

OPENASQA_CLASSIFICATION_JSON_URL = ""
OPENASQA_ALL_JSON_URL = ""

EVAL_ESC50_JSON_URL = ""
EVAL_IEMOCAP_JSON_URL = ""
EVAL_VOXCELEB_GENDER_JSON_URL = ""
EVAL_VOXCELEB_AGE_JSON_URL = ""
EVAL_LIBRISPEECH_TEST_CLEAN_JSON_URL = ""


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
        logger.info("Training preparation completed in previous run, skipping.")
        return

    logger.info(
        f"Creating {classification_json}, {all_json}"
    )

    download_file(PRETRAINED_TLTR_URL, pretrained_tltr_path)
    download_file(OPENASQA_CLASSIFICATION_JSON_URL, classification_json)
    download_file(OPENASQA_ALL_JSON_URL, all_json)

    if not os.path.exists(whisper_feature_folder):
        os.makedirs(whisper_feature_folder)

    subsets = []
    if audioset_folder is not None:
        os.makedirs(os.path.join(whisper_feature_folder, "audioset"))
        subsets.append("audioset")
    if vggsound_folder is not None:
        os.makedirs(os.path.join(whisper_feature_folder, "vgg-sound"))
        subsets.append("vggsound")
    if fsd50k_folder is not None:
        os.makedirs(os.path.join(whisper_feature_folder, "fsd50k"))
        subsets.append("fsd50k")
    if audiocaps_folder is not None:
        os.makedirs(os.path.join(whisper_feature_folder, "audiocaps"))
        subsets.append("audiocaps")
    if clotho_folder is not None:
        os.makedirs(os.path.join(whisper_feature_folder, "clotho"))
        subsets.append("clotho")
    if iemocap_folder is not None:
        os.makedirs(os.path.join(whisper_feature_folder, "iemocap"))
        subsets.append("iemocap")
    if libritts_folder is not None:
        os.makedirs(os.path.join(whisper_feature_folder, "libritts"))
        subsets.append("libritts")
    if voxceleb2_folder is not None:
        os.makedirs(os.path.join(whisper_feature_folder, "voxceleb2"))
        subsets.append("voxceleb")
    if mosei_folder is not None:
        os.makedirs(os.path.join(whisper_feature_folder, "mosei"))
        subsets.append("mosei")
    if fma_folder is not None:
        os.makedirs(os.path.join(whisper_feature_folder, "fma"))
        subsets.append("fma")

    assert len(subsets) > 0, "At least one dataset should be provided"

    for file in [classification_json, all_json]:
        with open(file, 'r') as f:
            data = json.load(f)

        new_dict = {}

        for key in data.keys():
            audio_id = data[key]["audio_id"]
            feature_path = data[key]["feature_path"]

            if "audioset_folder" in audio_id:
                audio_id = audio_id.format(audioset_folder=audioset_folder)
            elif "vggsound_folder" in audio_id:
                audio_id = audio_id.format(vggsound_folder=vggsound_folder)
            elif "fsd50k_folder" in audio_id:
                audio_id = audio_id.format(fsd50k_folder=fsd50k_folder)
            elif "audiocaps_folder" in audio_id:
                audio_id = audio_id.format(audiocaps_folder=audiocaps_folder)
            elif "clotho_folder" in audio_id:
                audio_id = audio_id.format(clotho_folder=clotho_folder)
            elif "iemocap_folder" in audio_id:
                audio_id = audio_id.format(iemocap_folder=iemocap_folder)
            elif "libritts_folder" in audio_id:
                audio_id = audio_id.format(libritts_folder=libritts_folder)
            elif "voxceleb2_folder" in audio_id:
                audio_id = audio_id.format(voxceleb2_folder=voxceleb2_folder)
            elif "mosei_folder" in audio_id:
                audio_id = audio_id.format(mosei_folder=mosei_folder)
            elif "fma_folder" in audio_id:
                audio_id = audio_id.format(fma_folder=fma_folder)

            if not os.path.exists(audio_id):
                logger.info(f"{audio_id} does not exist, please check the audio path")
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
        
        with open(file, 'w') as fout:
            json.dump(new_dict,fout)


def prepare_openasqa_eval(
    eval_whisper_feature_folder,
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
    To be done
    download json files, tltr pretrained weights
    For the ready-to-use json ["dataset"] IEMOCAP and FMA should br lower-cased,
    """

    if skip(valid_json):
        logger.info("Evaluation preparation completed in previous run, skipping.")
        return

    if not os.path.exists(eval_whisper_feature_folder):
        os.makedirs(eval_whisper_feature_folder)


    valid_dict = {}

    if esc50_folder is not None:
        logger.info(f"Creating {eval_esc50_json}")
        download_file(EVAL_ESC50_JSON_URL, eval_esc50_json)
        os.makedirs(os.path.join(eval_whisper_feature_folder, "esc50"))
        with open(eval_esc50_json, 'r') as f:
            data = json.load(f)
        
            new_dict = {}
            for key in data.keys():
                audio_id = data[key]["audio_id"]
                feature_path = data[key]["feature_path"]

                audio_id = audio_id.format(esc50_folder=esc50_folder)

                if not os.path.exists(audio_id):
                    logger.info(f"{audio_id} does not exist, please check the audio path")
                    continue
                
                feature_path.format(eval_whisper_feature_folder=eval_whisper_feature_folder)
                if not os.path.exists(feature_path):
                    extract_whisper_features(
                        whisper_model,
                        average_pooling,
                        audio_id,
                        feature_path,
                    )
                # update new dictionary with selected items
                new_dict[key] = data[key]

        # Add the last item of each dataset into valid set
        valid_dict[key] = data[key]

        with open(eval_esc50_json, 'w') as fout:
            json.dump(new_dict,fout)
        logger.info(f"{eval_esc50_json} successfully created.")


    if iemocap_folder is not None:
        logger.info(f"Creating {eval_iemocap_emo_json}")
        download_file(EVAL_IEMOCAP_JSON_URL, eval_iemocap_emo_json)
        os.makedirs(os.path.join(eval_whisper_feature_folder, "iemocap"))
        with open(eval_iemocap_emo_json, 'r') as f:
            data = json.load(f)
        
            new_dict = {}
            for key in data.keys():
                audio_id = data[key]["audio_id"]
                feature_path = data[key]["feature_path"]

                audio_id = audio_id.format(iemocap_folder=iemocap_folder)

                if not os.path.exists(audio_id):
                    logger.info(f"{audio_id} does not exist, please check the audio path")
                    continue
                
                feature_path.format(eval_whisper_feature_folder=eval_whisper_feature_folder)
                if not os.path.exists(feature_path):
                    extract_whisper_features(
                        whisper_model,
                        average_pooling,
                        audio_id,
                        feature_path,
                    )
                # update new dictionary with selected items
                new_dict[key] = data[key]
        
        # Add the last item of each dataset into valid set
        valid_dict[key] = data[key]

        with open(eval_iemocap_emo_json, 'w') as fout:
            json.dump(new_dict,fout)
        logger.info(f"{eval_iemocap_emo_json} successfully created.")
        

    if voxceleb2_test_folder is not None:
        logger.info(f"Creating {eval_voxceleb_gender_json} and {eval_voxceleb_age_json}")
        download_file(EVAL_VOXCELEB_GENDER_JSON_URL, eval_voxceleb_gender_json)
        download_file(EVAL_VOXCELEB_AGE_JSON_URL, eval_voxceleb_age_json)

        os.makedirs(os.path.join(eval_whisper_feature_folder, "voxceleb"))

        for file in [eval_voxceleb_gender_json, eval_voxceleb_age_json]:
            with open(file, 'r') as f:
                data = json.load(f)

                new_dict = {}
                for key in data.keys():
                    audio_id = data[key]["audio_id"]
                    feature_path = data[key]["feature_path"]

                    audio_id = audio_id.format(voxceleb2_test_folder=voxceleb2_test_folder)

                    if not os.path.exists(audio_id):
                        logger.info(f"{audio_id} does not exist, please check the audio path")
                        continue
                    
                    feature_path.format(eval_whisper_feature_folder=eval_whisper_feature_folder)
                    if not os.path.exists(feature_path):
                        extract_whisper_features(
                            whisper_model,
                            average_pooling,
                            audio_id,
                            feature_path,
                        )
                    # update new dictionary with selected items
                    new_dict[key] = data[key]

            # Add the last item of each dataset into valid set
            valid_dict[key] = data[key]

            with open(file, 'w') as fout:
                json.dump(new_dict,fout)
            logger.info(f"{file} successfully created.")
        
    
    if librispeech_test_clean_folder is not None:
        logger.info(f"Creating {eval_librispeech_asr_json}")
        download_file(EVAL_LIBRISPEECH_TEST_CLEAN_JSON_URL, eval_librispeech_asr_json)
        os.makedirs(os.path.join(eval_whisper_feature_folder, "librispeech"))
        with open(eval_librispeech_asr_json, 'r') as f:
            data = json.load(f)
        
            new_dict = {}
            for key in data.keys():
                audio_id = data[key]["audio_id"]
                feature_path = data[key]["feature_path"]

                audio_id = audio_id.format(librispeech_test_clean_folder=librispeech_test_clean_folder)

                if not os.path.exists(audio_id):
                    logger.info(f"{audio_id} does not exist, please check the audio path")
                    continue
                
                feature_path.format(eval_whisper_feature_folder=eval_whisper_feature_folder)
                if not os.path.exists(feature_path):
                    extract_whisper_features(
                        whisper_model,
                        average_pooling,
                        audio_id,
                        feature_path,
                    )
                # update new dictionary with selected items
                new_dict[key] = data[key]
        
        # Add the last item of each dataset into valid set
        valid_dict[key] = data[key]

        with open(eval_librispeech_asr_json, 'w') as fout:
            json.dump(new_dict,fout)
        logger.info(f"{eval_librispeech_asr_json} successfully created.")

    with open(valid_json, 'w') as fout:
        json.dump(valid_dict,fout)
        logger.info(f"A tiny valid set is also successfully created.")


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
        eval_whisper_feature_folder=hparams["eval_whisper_feature_folder"],
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