from abc import ABCMeta, abstractmethod
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any
import os
import lilcom


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

@dataclass
class FeatureStorageConfig:
    name: str
    dtype: np.dtype
    writer_class: type
    writer_kwargs: Dict[str, Any] = field(default_factory=dict)


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
        os.makedirs(p.parent, exist_ok=True)
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