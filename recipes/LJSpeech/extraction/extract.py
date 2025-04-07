"""An feature extraction script for LibriTTS

Authors
 * Artem Ploujnikov 2025
"""

import sys
from pathlib import Path

import torchaudio
from hyperpyyaml import load_hyperpyyaml
from torch import nn

import speechbrain as sb
from speechbrain.dataio.batch import PaddedData
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset prep (parsing Librispeech
    from ljspeech_prepare import prepare_ljspeech  # noqa

    # multi-gpu (ddp) save data preparation
    parent_folder = Path(hparams["save_folder"]).parent
    parent_folder.mkdir(exist_ok=True, parents=True)
    run_on_main(
        prepare_ljspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "splits": hparams["splits"],
            "split_ratio": hparams["split_ratio"],
            "seed": hparams["seed"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    modules = nn.ModuleDict(hparams["modules"]).to(run_opts["device"])

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig, rate = torchaudio.load(wav)
        sig = torchaudio.functional.resample(sig, rate, hparams["sample_rate"])
        return sig.squeeze(0)

    @sb.utils.data_pipeline.takes("sig")
    @sb.utils.data_pipeline.provides("audio_features")
    def audio_features_pipeline(sig):
        audio_features = modules.ssl_model(sig.data, sig.lengths)
        return PaddedData(audio_features, sig.lengths)

    @sb.utils.data_pipeline.takes("sig")
    @sb.utils.data_pipeline.provides("audio_tokens", "audio_emb")
    def audio_tokens_pipeline(sig):
        _, audio_emb, audio_tokens = modules.tokens_model.encode(
            sig.data, sig.lengths, **hparams["tokenizer_kwargs"]
        )
        yield PaddedData(audio_tokens, sig.lengths)
        yield PaddedData(audio_emb, sig.lengths)

    data_folder = hparams["data_folder"]
    datasets = []
    for split in ["train", "valid", "test"]:
        json_path = hparams[f"{split}_json"]
        name = Path(json_path).stem
        dataset = DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
        )
        # NOTE: This is only for debugging
        if hparams["data_count"]:
            dataset.data_ids = dataset.data_ids[: hparams["data_count"]]
        datasets.append(dataset)

    merged_data = {
        data_id: dataset.data[data_id]
        for dataset in datasets
        for data_id in dataset.data_ids
    }
    merged_dataset = DynamicItemDataset(merged_data)
    merged_dataset.add_dynamic_item(audio_pipeline)

    logger.info("Extracting features")
    feature_extractor = hparams["feature_extractor"](
        device=run_opts["device"],
        dynamic_items=[
            audio_features_pipeline,
            audio_tokens_pipeline,
        ],
    )
    feature_extractor.set_output_features(hparams["extract_features"])
    feature_extractor.extract(merged_dataset)
    logger.info("Extraction done")
