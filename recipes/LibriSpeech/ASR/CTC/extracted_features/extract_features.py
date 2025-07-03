import speechbrain as sb
from speechbrain.utils.logger import get_logger
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.dataio.dataloader import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import ExitStack
from typing import Dict, Any
import os 
import lilcom
from dataclasses import dataclass
from speechbrain.dataio.feature_io import FeatureStorageConfig, FeatureStorageWriter, create_feature_storage_writers, NumpyHdf5Writer
from hyperpyyaml import load_hyperpyyaml
import sys 
"""
python extract_features.py hparams/extract_ssl_representations.yaml --data_folder=$SLURM_TMPDIR/librispeech/LibriSpeech/ --output_folder $SLURM_TMPDIR/results/extract_ssl_representations/ --batch_size=64


python train_with_wav2vec.py hparams/train_hf_wav2vec.yaml --data_folder=$SLURM_TMPDIR/LibriSpeech/ --output_folder $SCRATCH/results/wav2vec2-base-960h/ --extracted_features_folder $SLURM_TMPDIR/results/extract_ssl_representations/ssl_features --batch_size=32

find . -type f -name '*.tar.gz' -exec tar -xzf {} -C . \;
scp -r $HOME/projects/def-ravanelm/datasets/librispeech/* .


python extract_features.py hparams/extract_ssl_representations.yaml --data_folder=$SLURM_TMPDIR/LibriSpeech/ --output_folder $SCRATCH/extracted_features/wav2vec2-base-960h/ --batch_size=20
python extract_features.py hparams/extract_ssl_representations.yaml --data_folder=$SLURM_TMPDIR/LibriSpeech/ --output_folder $SCRATCH/extracted_features/hubert-large-ll60k/ --batch_size=20

python train_with_wav2vec.py hparams/train_hf_wav2vec.yaml --data_folder=$SLURM_TMPDIR/LibriSpeech/ --output_folder $SCRATCH/results/wav2vec2-base-960h-10-epochs/ --extracted_features_folder $SLURM_TMPDIR/results/extract_ssl_representations/save/ssl_features --batch_size=32 --number_of_epochs 10
"""
logger = get_logger(__name__)
            
@dataclass
class FeatureExtractionConfig:
    utterance_id_key: str = "id"
    ssl_key: str = "ssl_feats"

class ExtractFeatures(sb.core.Brain):
    def __init__(
            self, 
            modules,
            hparams,
            run_opts,
            feature_extraction_config: FeatureExtractionConfig):
        super().__init__(
            modules=modules,
            hparams=hparams,
            run_opts=run_opts,
        )
        self.feature_extraction_config = feature_extraction_config

    def compute_features(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        batch_size = wavs.shape[0]
        
        # extract features
        feats = self.modules.wav2vec2(wavs, wav_lens)

        return [
            {
                self.feature_extraction_config.utterance_id_key: batch.id[i], 
                self.feature_extraction_config.ssl_key: feats[i], 
            } for i in range(batch_size)
        ]

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig"],
    )

    return train_data, valid_data, test_datasets

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech

    sb.utils.distributed.run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(
        hparams
    )

    feature_extractor = ExtractFeatures(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        feature_extraction_config=FeatureExtractionConfig(
            utterance_id_key="id",
            ssl_key="ssl_feats",
        )
    )

    feature_extractor.cache_features(
        hparams["train_feature_storage_writers"], 
        train_data, 
        loader_kwargs=hparams["dataloader_opts"], 
        stage=sb.Stage.TRAIN
    )
    feature_extractor.cache_features(
        hparams["valid_feature_storage_writers"], 
        valid_data, 
        loader_kwargs=hparams["dataloader_opts"], 
        stage=sb.Stage.VALID
    )
    for k, v in test_datasets.items():
        feature_extractor.cache_features(
            hparams["test_feature_storage_writers"][k], 
            v, 
            loader_kwargs=hparams["dataloader_opts"], 
            stage=sb.Stage.TEST
        )

    # from speechbrain.dataio.feature_io import NumpyHdf5Reader
    # reader = NumpyHdf5Reader(os.path.join(hparams["ssl_features_folder"], "train_hdf5_v2_ssl_feats.h5"))
    # for item in train_data:
    #     print(item['id'])
    #     print(reader.read(item['id']))
    #     exit()