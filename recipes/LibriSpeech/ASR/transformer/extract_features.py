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

logger = get_logger(__name__)


class FeatureStorageReader(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: int = None,
    ) -> np.ndarray:
        ...

class FeatureStorageWriter(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def storage_path(self) -> str:
        ...

    @abstractmethod
    def write(self, key: str, value: np.ndarray) -> str:
        ...
    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        ...


class NumpyHdf5Writer(FeatureStorageWriter):
    name = "numpy_hdf5"

    def __init__(self, storage_path: Path, mode: str = "w", dtype=np.float32, *args, **kwargs):
        """
        :param storage_path: Path under which we'll create the HDF5 file.
            We will add a ``.h5`` suffix if it is not already in ``storage_path``.
        :param mode: Modes supported by h5py:
            w        Create file, truncate if exists (default)
            w- or x  Create file, fail if exists
            a        Read/write if exists, create otherwise
        """
        super().__init__()
        import h5py

        p = Path(storage_path)
        self.storage_path_ = p.with_suffix(
            p.suffix + ".h5" if p.suffix != ".h5" else ".h5"
        )
        print(f"saving to {self.storage_path_}")
        self.hdf = h5py.File(self.storage_path, mode=mode)
        self.dtype = dtype

    @property
    def storage_path(self) -> str:
        return str(self.storage_path_)

    def write(self, key: str, value: np.ndarray) -> str:
        value = value.cpu().numpy().astype(self.dtype)
        self.hdf.create_dataset(key, data=value)
        return key

    def close(self) -> None:
        return self.hdf.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class NumpyHdf5Reader(FeatureStorageReader):

    name = "numpy_hdf5"

    def __init__(self, storage_path: Path, *args, **kwargs):
        super().__init__()
        import h5py
        self.hdf5_file = h5py.File(storage_path, mode="r")

    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: int = None,
    ) -> np.ndarray:
        return self.hdf5_file[key][left_offset_frames:right_offset_frames]

    def close(self) -> None:
        return self.hdf5_file.close()

class LilcomHdf5Reader(FeatureStorageReader):
    """
    Reads lilcom-compressed numpy arrays from a HDF5 file with a "flat" layout.
    Each array is stored as a separate HDF ``Dataset`` because their shapes (numbers of frames) may vary.
    ``storage_path`` corresponds to the HDF5 file path;
    ``storage_key`` for each utterance is the key corresponding to the array (i.e. HDF5 "Group" name).
    """

    name = "lilcom_hdf5"

    def __init__(self, storage_path: Path, *args, **kwargs):
        super().__init__()
        import h5py
        self.hdf5_file = h5py.File(storage_path, mode="r")

    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: int = None,
    ) -> np.ndarray:
        arr = lilcom.decompress(self.hdf5_file[key][()].tobytes())
        return arr[left_offset_frames:right_offset_frames]
    
class LilcomHdf5Writer(FeatureStorageWriter):
    """
    Writes lilcom-compressed numpy arrays to a HDF5 file with a "flat" layout.
    Each array is stored as a separate HDF ``Dataset`` because their shapes (numbers of frames) may vary.
    ``storage_path`` corresponds to the HDF5 file path;
    ``storage_key`` for each utterance is the key corresponding to the array (i.e. HDF5 "Group" name).
    """

    name = "lilcom_hdf5"

    def __init__(
        self,
        storage_path: Path,
        dtype=np.float32,
        tick_power: int = -5,
        mode: str = "w",
        *args,
        **kwargs,
    ):
        """
        :param storage_path: Path under which we'll create the HDF5 file.
            We will add a ``.h5`` suffix if it is not already in ``storage_path``.
        :param tick_power: Determines the lilcom compression accuracy;
            the input will be compressed to integer multiples of 2^tick_power.
        :param mode: Modes supported by h5py:
            w        Create file, truncate if exists (default)
            w- or x  Create file, fail if exists
            a        Read/write if exists, create otherwise
        """
        super().__init__()
        import h5py

        p = Path(storage_path)
        self.storage_path_ = p.with_suffix(
            p.suffix + ".h5" if p.suffix != ".h5" else ".h5"
        )
        self.hdf = h5py.File(self.storage_path, mode=mode)
        self.tick_power = tick_power
        self.dtype = dtype

    @property
    def storage_path(self) -> str:
        return str(self.storage_path_)

    def write(self, key: str, value: np.ndarray) -> str:
        value = value.cpu().numpy().astype(self.dtype)
        serialized_feats = lilcom.compress(value, tick_power=self.tick_power)
        self.hdf.create_dataset(key, data=np.void(serialized_feats))
        return key

    def close(self) -> None:
        return self.hdf.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

@dataclass
class FeatureStorageConfig:
    name: str
    dtype: np.dtype
    writer_class: type = LilcomHdf5Writer
    writer_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.writer_kwargs is None:
            self.writer_kwargs = {}
            
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
            open_writers = {ft: stack.enter_context(writers[ft]) for ft in writers.keys()}
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

def create_feature_storage_writers(
    feature_configs: Dict[str, FeatureStorageConfig],
    base_path: str,
    prefix: str = "",
    suffix: str = ""
) -> Dict[str, FeatureStorageWriter]:
    """
    Create writers from feature configurations.
    
    Args:
        feature_configs: Dict mapping feature names to FeatureConfig objects
        base_path: Base directory for storing files
        prefix: Optional prefix for filenames
        suffix: Optional suffix for filenames
    
    Returns:
        Dict mapping feature names to initialized writers
    """
    writers = {}
    for feature_name, config in feature_configs.items():
        filename = f"{prefix}{feature_name}{suffix}"
        storage_path = os.path.join(base_path, filename)
        
        writers[feature_name] = config.writer_class(
            storage_path=storage_path,
            dtype=config.dtype,
            **config.writer_kwargs
        )
    
    return writers

if __name__ == "__main__":
    from librispeech_prepare import prepare_librispeech

    prepare_librispeech(
        data_folder= os.environ["SLURM_TMPDIR"] + "/LibriSpeech",
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
        ),
    }

    # Create writers with one line
    writers = create_feature_storage_writers(
        feature_configs=feature_configs,
        base_path=os.path.join(os.environ['SLURM_TMPDIR'], "save"),
        prefix="train_lilcom_"
    )

    feature_extractor = ExtractFeatures(
        feature_extraction_config=FeatureExtractionConfig(
            utterance_id_key="id",
            filterbank_key="fbanks",
        )
    )
    
    feature_extractor.cache_features(writers, train_data, loader_kwargs={"batch_size": 12})
