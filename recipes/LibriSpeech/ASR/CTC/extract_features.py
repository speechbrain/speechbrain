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
        
        # Add waveform augmentation if specified.
        # if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
        #     wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens) 
        
        feats = self.modules.wav2vec2(wavs, wav_lens)

        return [
            {
                self.feature_extraction_config.utterance_id_key: batch.id[i], 
                self.feature_extraction_config.ssl_key: feats[i], 
            } for i in range(batch_size)
        ]

def dataio_prepare():
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = os.environ["SLURM_TMPDIR"] + "/save"

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.environ["SLURM_TMPDIR"] + "/save/train.csv",
        replacements={"data_root": data_folder},
    )

    datasets = [train_data]

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

    return (
        train_data,
    )

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

    prepare_librispeech(
        data_folder= os.environ["SLURM_TMPDIR"] + "/librispeech/LibriSpeech",
        tr_splits= ['train-clean-100'],
        dev_splits= ['dev-clean'],
        te_splits= ['test-clean'],
        save_folder= os.environ["SLURM_TMPDIR"] + "/save",
        merge_lst= ['train-clean-100'],
        merge_name= "train.csv",
        skip_prep= False,
    )

    train_data, = dataio_prepare()

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
        hparams["feature_storage_writers"], 
        train_data, 
        loader_kwargs={"batch_size": 12}, 
        stage=sb.Stage.TRAIN
    )

    from speechbrain.dataio.feature_io import NumpyHdf5Reader
    reader = NumpyHdf5Reader(os.path.join(hparams["ssl_features_folder"], "train_hdf5_v2_ssl_feats.h5"))
    for item in train_data:
        print(item['id'])
        print(reader.read(item['id']))
        exit()