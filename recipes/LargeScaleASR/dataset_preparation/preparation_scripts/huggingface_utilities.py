""" A few utilities useful to convert a file based dataset into a sharded dataset compatible with Huggingface datasets (.parquet format).

Authors: Titouan Parcollet

"""

import datasets
from datasets import Audio, Features, Value
from datasets.io.parquet import get_writer_batch_size
from datasets.table import embed_table_storage


def shards_with_embedded_external_files(shards):
    """This function is copied from HuggingFace. It makes it feasible
    to embed an external fil into the arrow/parquet table. It returns
    a generator.

    Arguments
    ---------
    shards: datasets.shard()
        list of shards obtained from a datasets with datasets.shard()

    Yields
    ------
    shard: datasets.shard
        A HuggingFace dataset shard.

    """
    for shard in shards:
        format = shard.format
        shard = shard.with_format("arrow")
        shard = shard.map(
            embed_table_storage,
            batched=True,
            batch_size=get_writer_batch_size(shard.features),
            keep_in_memory=True,
        )
        shard = shard.with_format(**format)
        yield shard


def convert_file_size_to_int(size):
    """Copied from HuggingFace. Converts a size expressed as a string with digits an unit (like `"50MB"`) to an integer (in bytes).

    Arguments
    ---------
    size : int or str
        The size to convert. Will be directly returned if an `int`.

    Return
    ------
    size: int
        The equivalent size in bytes expressed as a int of the given size.
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("PIB"):
        return int(size[:-3]) * (2**50)
    if size.upper().endswith("TIB"):
        return int(size[:-3]) * (2**40)
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("PB"):
        int_size = int(size[:-2]) * (10**15)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("TB"):
        int_size = int(size[:-2]) * (10**12)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError(
        f"`size={size}` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'."
    )


def clean_the_parquet(hparams):
    """This function loads The LargeScaleASR Set as an HuggingFace datasets object
    and stores it as parquet (sharded) files. Each subset/split must be done individually. E.g. large then medium then small then val then test.

    The size of the shards can be changed in the yaml.

    """

    header = Features(
        {
            "ID": Value(dtype="string"),
            "duration": Value(dtype="float32"),
            "start": Value(dtype="float32"),
            "wav": Value(dtype="string"),
            "spk_id": Value(dtype="string"),
            "sex": Value(dtype="string"),
            "text": Value(dtype="string"),
        }
    )

    ds = datasets.load_dataset(
        hparams["HF_DATASET_ROOT"],
        name=hparams["PARQUET_SUBSET"],
        features=header,
    ).cast_column("wav", Audio(decode=False))[hparams["PARQUET_SPLIT"]]

    dataset_nbytes = ds._estimate_nbytes()
    max_shard_size = convert_file_size_to_int(hparams["MAX_SHARD_SIZE"])
    num_shards = int(dataset_nbytes / max_shard_size) + 1
    num_shards = max(num_shards, 1)

    shards = (
        ds.shard(num_shards=num_shards, index=i, contiguous=True)
        for i in range(num_shards)
    )
    shards = shards_with_embedded_external_files(shards)

    for i, shard in enumerate(shards):
        out_folder = hparams["PARQUET_OUTPUT_FOLDER"]
        parquet_split = hparams["PARQUET_SPLIT"]
        shard_path_in_repo = (
            f"{out_folder}/{parquet_split}-{i:05d}-of-{num_shards:05d}.parquet"
        )
        shard.to_parquet(shard_path_in_repo)
