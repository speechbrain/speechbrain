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

logger = get_logger(__name__)
            
sample_rate = 16000
n_fft = 512
n_mels = 80
win_length = 32
fbank = sb.lobes.features.Fbank(
   sample_rate=sample_rate,
   n_fft=n_fft,
   n_mels=n_mels,
   win_length=win_length
)

@dataclass
class FeatureExtractionConfig:
    utterance_id_key: str = "id"
    filterbank_key: str = "fbanks"

class ExtractFeatures(sb.core.Brain):
    def __init__(self, feature_extraction_config: FeatureExtractionConfig):
        super().__init__()
        self.feature_extraction_config = feature_extraction_config

    def compute_features(self, batch):
        #NOTE: the user will only have to implement this method
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        feats = fbank(wavs)
        batch_size = wavs.shape[0]        

        return [
            {
                self.feature_extraction_config.utterance_id_key: batch.id[i], 
                self.feature_extraction_config.filterbank_key: feats[i], 
            } for i in range(batch_size)
        ]
    
    def cache_features(
        self,
        writers: Dict[str, FeatureStorageWriter],
        dataset,
        loader_kwargs={},
        progressbar=None,
        overwrite=False,
    ):
        #NOTE: will be part of core.py
        assert writers is not None, "Feature cache manager must be provided"
            
        if not (
            isinstance(dataset, DataLoader)
            or isinstance(dataset, LoopedLoader)
        ):
            dataset = self.make_dataloader(
                dataset, stage=sb.Stage.TEST, **loader_kwargs
            )
        
        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()
        feature_types = list(writers.keys())
        
        with ExitStack() as stack:
            open_writers = {ft: stack.enter_context(writers[ft]) for ft in feature_types}
            for data in self._cache_features(
                dataset=dataset,
                enable=enable,
            ):
                for item in data:
                    for ft in feature_types:
                        open_writers[ft].write(item[self.feature_extraction_config.utterance_id_key], item[ft])

    def _cache_features(self, dataset, enable):
        #NOTE: will be part of core.py
        for batch in tqdm(dataset, dynamic_ncols=True, disable=not enable, colour=self.tqdm_barcolor["valid"]):
            extracted_features = self.compute_features(batch)
            yield extracted_features

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
    
    # Define feature configurations
    feature_configs = {
        'fbanks': FeatureStorageConfig(
            name='fbanks',
            dtype=np.float32,
            writer_class=NumpyHdf5Writer,
            # NOTE: mode="x" will fail if the file already exists.
            writer_kwargs={"mode": "x"}
        ),
    }

    # Create writers with one line
    writers = create_feature_storage_writers(
        feature_configs=feature_configs,
        base_path=os.path.join(os.environ['SLURM_TMPDIR'], "save"),
        prefix="train_hdf5_v2_"
    )

    feature_extractor = ExtractFeatures(
        feature_extraction_config=FeatureExtractionConfig(
            utterance_id_key="id",
            filterbank_key="fbanks",
        )
    )
    
    feature_extractor.cache_features(
        writers, train_data, loader_kwargs={"batch_size": 12}, overwrite=True)

    from speechbrain.dataio.feature_io import NumpyHdf5Reader
    reader = NumpyHdf5Reader(os.path.join(os.environ['SLURM_TMPDIR'], "save", "train_hdf5_fbanks.h5"))
    for item in train_data:
        print(item['id'])
        print(reader.read(item['id']))
        exit()