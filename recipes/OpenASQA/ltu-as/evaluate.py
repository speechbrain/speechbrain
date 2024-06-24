"""
LTU-AS evaluation with 5 different datasets and log their metrics.

Authors
 * Yingzhi Wang 2024
"""

import json
import logging
import re
import sys
from statistics import mean

import jiwer
import torch
from hyperpyyaml import load_hyperpyyaml
from transformers import AutoTokenizer, AutoModelForCausalLM

import speechbrain as sb
from speechbrain.inference.multimodal import LTU_AS
from speechbrain.utils.metric_stats import BinaryMetricStats, ErrorRateStats

logger = logging.getLogger(__name__)


def infer(model, item_dict):
    """
    Inference with pre-extracted whisper features.

    Arguments
    ---------
    model:
        ltu-as pretrained model.
    item_dict: dict
        An inference item that contains audio_id, instruction, input, output.

    Returns
    -------
    response
        Generated hypothesis.
    """
    predicted_words = model.generate_with_raw_audio(
        item_dict["audio_id"], item_dict["instruction"], item_dict["input"]
    )
    print(predicted_words[0])
    return predicted_words


def eval_iemocap_emo(model, hparams):
    """
    IEMOCAP emotion classification evaluation.

    Arguments
    ---------
    model:
        ltu-as pretrained model.
    hparams: dict
        Hyperparameters.

    Returns
    -------
    Accuracy
    """
    with open(hparams["eval_iemocap_emo_json"], "r") as f:
        data = json.load(f)
    good = 0
    for key in data.keys():
        pred = infer(model, data[key])[0]
        pred_emotion = pred.replace("Speech emotion: ", "")[:-1]
        if pred_emotion in data[key]["output"]:
            good += 1
    return good / len(data)


def eval_voxceleb_gender(model, hparams):
    """
    Voxceleb2 test set gender classification evaluation.

    Arguments
    ---------
    model:
        ltu-as pretrained model.
    hparams: dict
        Hyperparameters.

    Returns
    -------
    F1-score
    """
    metric = BinaryMetricStats()

    with open(hparams["eval_voxceleb_gender_json"], "r") as f:
        data = json.load(f)
    for key in data.keys():
        pred = infer(model, data[key])[0]

        if "female" in pred:
            pred_label = 0
        else:
            pred_label = 1
        if "female" in data[key]["output"]:
            real_label = 0
        else:
            real_label = 1
        metric.append(
            [key], torch.tensor([pred_label]), torch.tensor([real_label])
        )
    return metric.summarize()["F-score"]


def eval_voxceleb_age(model, hparams):
    """
    Voxceleb2 test set gender prediction evaluation.

    Arguments
    ---------
    model:
        ltu-as pretrained model.
    hparams: dict
        Hyperparameters.

    Returns
    -------
    Mean Absolute Error
    """

    def extract_and_process_number(string):
        pattern = r"\b(\d+(?:-\d+)?)\b"
        matches = re.findall(pattern, string)

        def process_number_range(number):
            if "-" in number:
                start, end = map(int, number.split("-"))
                return mean([start, end])
            else:
                return int(number)

        processed_result = [process_number_range(n) for n in matches]
        return processed_result

    def age_reg(ref, pred):
        ref = int(ref.split(" ")[-1][:-1])
        if len(extract_and_process_number(pred)) != 0:
            pred = extract_and_process_number(pred)[0]
            return ref, pred
        else:
            return None, None

    with open(hparams["eval_voxceleb_age_json"], "r") as f:
        data = json.load(f)
    all_mae = []
    for key in data.keys():
        pred = infer(model, data[key])[0]
        ref, pred = age_reg(data[key]["output"], pred)
        mae = abs(ref - pred)
        all_mae.append(mae)

    return mean(all_mae)


def eval_librispeech_asr(model, hparams):
    """
    Librispeech test-clean ASR evaluation.

    Arguments
    ---------
    model:
        ltu-as pretrained model.
    hparams: dict
        Hyperparameters.

    Returns
    -------
    Word Error Rate
    """
    metric = ErrorRateStats()

    def process_text(text):
        text = text.lower()
        text = (
            text.split("spoken text: ")[-1].replace("spoken text:","").lstrip()
        )
        return text

    def preprocess_text(cur_trans):
        cur_trans = jiwer.ToUpperCase()(cur_trans)
        cur_trans = jiwer.RemovePunctuation()(cur_trans)
        return cur_trans

    with open(hparams["eval_librispeech_asr_json"], "r") as f:
        data = json.load(f)
    for key in data.keys():
        pred = infer(model, data[key])[0]
        metric.append(
            [key],
            [preprocess_text(process_text(pred))],
            [preprocess_text(process_text(data[key]["output"]))]        
        )
    return metric.summarize()["WER"]


def eval_esc50(model, hparams):
    """
    ESC50 Audio Classification evaluation.
    Ltu-as model outpus an audio description and another llm is used for classification.

    Arguments
    ---------
    model:
        ltu-as pretrained model.
    hparams: dict
        Hyperparameters.

    Returns
    -------
    Accuracy
    """
    model_id = hparams["external_llm"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    external_llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    with open(hparams["eval_esc50_json"], "r") as f:
        data = json.load(f)
    good_count = 0
    for key in data.keys():
        pred = infer(model, data[key])[0]
        label = data[key]["output"]
        pred = pred.replace("Labels: ", "").replace("Audio caption: ", "")

        string = f"Classify the sound <<{pred}>> into one or several of the following labels: [car_horn, rooster, rain, pouring_water, clock_alarm, washing_machine, drinking_sipping, sea_waves, cow, thunderstorm, keyboard_typing, wind, airplane, engine, crickets, vacuum_cleaner, glass_breaking, crying_baby, coughing, chirping_birds, crow, sneezing, laughing, cat, snoring, sheep, door_wood_knock, dog, fireworks, mouse_click, clock_tick, hen, train, door_wood_creaks, water_drops, can_opening, hand_saw, pig, insects, crackling_fire, helicopter, footsteps, clapping, frog, siren, chainsaw, breathing, church_bells, toilet_flush, brushing_teeth], please make sure that: 1. answer with only a list of label(s). 2. multi-label is allowed. 3. Do not create new labels."
        messages = [
            {
                "role": "system",
                "content": "You are a sound classifier who classifies a sound description into a label.",
            },
            {"role": "user", "content": string},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ).to(external_llm.device)

        outputs = external_llm.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1] :]

        output = tokenizer.decode(response, skip_special_tokens=True).replace(
            "assistant\n\n", ""
        )
        output = output.lower()

        if label in output:
            good_count += 1
    return good_count / len(data)


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    ltu_as = LTU_AS.from_hparams(
        source=hparams["inference_folder"],
        run_opts={"device": "cuda"},
    )

    with open(hparams["eval_log"], "w") as f:
        if hparams["eval_iemocap_emo_json"] is not None:
            f.write(
                f"IEMOCAP emotion recognition (Acc) : {eval_iemocap_emo(ltu_as, hparams)}\n"
            )
        if hparams["eval_voxceleb_gender_json"] is not None:
            f.write(
                f"Voxceleb2 test gender classification (F1) : {eval_voxceleb_gender(ltu_as, hparams)}\n"
            )
        if hparams["eval_voxceleb_age_json"] is not None:
            f.write(
                f"Voxceleb2 test age prediction (MAE) : {eval_voxceleb_age(ltu_as, hparams)}\n"
            )
        if hparams["eval_librispeech_asr_json"] is not None:
            f.write(
                f"Librispeech test clean ASR (WER) : {eval_librispeech_asr(ltu_as, hparams)}\n"
            )
        if hparams["eval_esc50_json"] is not None:
            f.write(
                f"ESC50 audio classification (Acc) : {eval_esc50(ltu_as, hparams)}\n"
            )
