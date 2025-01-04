"""
Unified interface for token extraction and pretrained embeddings handling for speech tokenizers.

Authors
---------
* Jarod Duret, 2024
"""

import math
import logging
import pathlib as pl
import kaldiio
import torch
import torchaudio
import numpy as np
from tqdm.auto import tqdm
import speechbrain as sb
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataio import load_pkl, save_pkl


logger = logging.getLogger(__name__)


def get_device(use_cuda):
    logger.info("=" * 30)
    logger.info(f"USE_CUDA SET TO: {use_cuda}")
    logger.info(f"CUDA AVAILABLE?: {torch.cuda.is_available()}")
    logger.info("=" * 30)
    use_cuda = use_cuda and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


class TokensExtractor:
    """
    Extracts tokens from audio data using a tokenizer and saves them to a specified format.

    Arguments
    ---------
    tokenizer : torch.nn.Module
        The tokenizer model to use for token extraction.
    sample_rate : int
        The sample rate of the audio data.
    src_key : str, optional
        The key in the dataset that contains the audio data (default: "wav").
    id_key : str, optional
        The key in the dataset that contains unique identifiers (default: "id").
    start_key: str, optional
        The key in the dataset that contains the audio start timestep (default: None).
    duration_key: str, optional
        The key in the dataset that contains the audio duration (default: None).
    save_format : str, optional
        The format to save the tokens ('numpy', 'pickle', 'soundfile_flac') (default: "numpy").
    use_cuda : bool, optional
        Whether to use CUDA for computation (default: True).
    dataloader_opts : dict, optional
        Options for the data loader (default: None).
    rank : int, optional
        The index of current shard (default: 0).
    num_shards : int, optional
        Total number of shards (default: 1).

    Raises
    ------
    ValueError
        If an unsupported save_format is provided.
    ValueError
        If the tokenizer's sample rate does not match the provided sample_rate.
    """

    def __init__(
        self,
        tokenizer,
        sample_rate,
        src_key="wav",
        id_key="id",
        start_key=None,
        duration_key=None,
        save_format="numpy",
        use_cuda=True,
        dataloader_opts=None,
        rank=0,
        num_shards=1,
    ):
        self.id_key = id_key
        self.src_key = src_key
        self.start_key = start_key
        self.duration_key = duration_key

        self.device = get_device(use_cuda)
        self.tokenizer = tokenizer.to(self.device)
        self.sample_rate = sample_rate
        self.rank = rank
        self.num_shards = num_shards

        if tokenizer.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: {self.sample_rate} != {tokenizer.sample_rate}"
            )

        if save_format not in ["numpy", "pickle", "soundfile_flac"]:
            raise ValueError(f"Unsupported save_format: {save_format}")
        self.save_format = save_format

        if not dataloader_opts:
            dataloader_opts = {}
        self.dataloader_opts = dataloader_opts
        self.pipelines = self._make_pipelines()

    def extract_tokens(self, dataset, num_codebooks, save_path, save_name="tokens"):
        """
        Extracts tokens from the dataset and saves them to the specified format.

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset or dict
            The dataset from which to extract tokens. Can be a DynamicItemDataset or a dictionary.
        num_codebooks: int
            The number of codebooks to retrieve from the tokens.
        save_path: str
            The path where tokens will be saved.
        save_name: str
            The name of the .scp and .ark files.
        """

        total_size = len(dataset)
        start, end = self._compute_shard_range(total_size)

        save_path = pl.Path(save_path).absolute()
        shard_save_name = f"{save_name}_shard_{self.rank}"

        conf = {
            "sample_rate": self.sample_rate,
            "save_folder": save_path,
            "dataset_length": end - start,
            "rank": self.rank,
            "num_shards": self.num_shards,
        }

        # Create output directory
        save_path.mkdir(parents=True, exist_ok=True)

        # Check if the extraction is already done (if so, skip it)
        if _skip(save_path, save_name, conf):
            logger.info("Skipping extraction, completed in previous run.")
            return

        self.wspecifier = (
            f"ark,scp,t:{save_path}/{shard_save_name}.ark,"
            f"{save_path}/{shard_save_name}.scp"
        )
        self.writer = kaldiio.WriteHelper(self.wspecifier, write_function="numpy")

        dataset = self.shard_dataset(dataset, self.rank, self.num_shards, start, end)
        dataset.set_output_keys([self.src_key, self.id_key, "sig"])
        for pipeline in self.pipelines:
            dataset.add_dynamic_item(pipeline)

        dataloader = make_dataloader(dataset, **self.dataloader_opts)
        batch_size = self.dataloader_opts.get("batch_size", 1)
        batch_count = int(math.ceil(len(dataset) / batch_size))
        for batch in tqdm(dataloader, total=batch_count):
            batch = batch.to(self.device)
            x, x_lengths = batch["sig"]
            ids = batch[self.id_key]
            batch_tokens = self.tokenizer.sig_to_tokens(
                x, x_lengths, num_codebooks=num_codebooks
            )
            batch_tokens = sb.utils.data_utils.undo_padding(batch_tokens, x_lengths)
            self.process_batch(batch_tokens, ids)

        logger.info("Extraction completed.")

        save_opt = save_path / f"opt_extract_{self.rank}.pkl"
        save_pkl(conf, save_opt.as_posix())

    def process_batch(self, batch, ids):
        """
        Processes a batch of tokens and writes them to the output files.

        Arguments
        ---------
        batch : list
            A list of tokens for each item in the batch.
        ids : list
            A list of unique identifiers corresponding to each item in the batch.
        """
        for tokens, utt_id in zip(batch, ids):
            tokens = np.array(tokens)
            self.writer(utt_id, tokens)

    def _make_pipelines(self):
        """
        Creates the data processing pipeline for audio data.

        The pipeline reads audio files, resamples them to the desired sample rate, and provides
        the processed signal under the key "sig".

        Returns
        -------
        pipeline : list
            A list containing the audio processing pipeline function.
        """

        if self.start_key and self.duration_key:

            @sb.utils.data_pipeline.takes(
                self.src_key, self.start_key, self.duration_key
            )
            @sb.utils.data_pipeline.provides("sig")
            def audio_pipeline(wav, start, duration):
                info = torchaudio.info(wav)
                sr = info.sample_rate
                start = float(start)
                duration = float(duration)
                expected_frames = int(duration * sr)

                sig, _ = torchaudio.load(
                    wav,
                    num_frames=expected_frames,
                    frame_offset=int(start * sr),
                    backend="soundfile",
                )
                sig = sig.squeeze(0)
                sig = torchaudio.transforms.Resample(sr, self.sample_rate)(sig)

                actual_frames = sig.size(-1)
                if abs(actual_frames - expected_frames) > sr * 0.02:  # 20ms tolerance
                    print(
                        f"Warning: Expected {expected_frames} frames, got {actual_frames}"
                    )

                return sig

        else:

            @sb.utils.data_pipeline.takes(self.src_key)
            @sb.utils.data_pipeline.provides("sig")
            def audio_pipeline(wav):
                info = torchaudio.info(wav)
                sig, _ = torchaudio.load(wav, backend="soundfile")
                sig = sig.squeeze(0)
                sig = torchaudio.transforms.Resample(
                    info.sample_rate,
                    self.sample_rate,
                )(sig)
                return sig

        return [audio_pipeline]

    def shard_dataset(self, dataset, rank, num_shards, start, end):
        """Shard a dataset.

        Arguments
        ---------
        dataset : dict or DynamicItemDataset
            The dataset to shard
        rank : int
            Current shard index
        num_shards : int
            Total number of shards

        Returns
        -------
        DynamicItemDataset
            A dataset containing only the current shard's data
        """
        data_dict = dataset if isinstance(dataset, dict) else dataset.data
        all_keys = list(data_dict.keys())
        shard_indices = list(range(start, end))

        # Create new data dictionary using only the keys we need
        shard_keys = [all_keys[i] for i in shard_indices]
        sharded_data = {k: data_dict[k] for k in shard_keys}

        # Return as DynamicItemDataset
        return DynamicItemDataset(sharded_data)

    def _compute_shard_range(self, total_size):
        """Calculate the range of indices for the current shard.

        Arguments
        ---------
        total_size : int
            Total size of the dataset

        Returns
        -------
        start : int
            Start index for this shard
        end : int
            End index for this shard
        """
        items_per_shard = math.ceil(total_size / self.num_shards)
        start = self.rank * items_per_shard
        end = min(start + items_per_shard, total_size)
        logger.info(
            f"Processing shard {self.rank + 1}/{self.num_shards}: "
            f"items {start}-{end} out of {total_size}"
        )
        return start, end

    def save_pretrained_embeddings(
        self,
        save_path,
        save_name="embeddings",
        vocab_size=None,
        num_codebooks=None,
    ):
        """
        Saves the pretrained embeddings of the tokenizer to a specified directory.

        This method retrieves the pretrained embeddings from the tokenizer,
        converts them to a NumPy array, and saves them as a `.npy` file.

        Parameters
        ----------
        save_path : str or pathlib.Path
            The directory where the pretrained embeddings will be saved.
            If the directory does not exist, it will be created.
        save_name : str, optional
            The base name of the saved embeddings file (default is "embeddings").
            The embeddings will be saved as `<save_name>.npy` in the specified directory.
        """
        save_path = pl.Path(save_path).absolute()
        save_path.mkdir(parents=True, exist_ok=True)

        embeddings = self.tokenizer.get_pretrained_embeddings(vocab_size, num_codebooks)
        embeddings = embeddings.cpu().numpy()
        np.save(save_path / save_name, embeddings)

    def __del__(self):
        """
        Close the writer.
        """
        self.writer.close()


def merge_sharded_tokens(save_path, save_name="tokens", num_shards=1):
    """Merge sharded .scp files into a single file.

    Arguments
    ---------
    save_path : str or Path
        Directory containing the sharded files
    save_name : str
        Base name of the token files
    """
    save_path = pl.Path(save_path)
    merged_scp = save_path / f"{save_name}.scp"

    # Read and merge all scp files
    with open(merged_scp, "w") as outfile:
        for rank in range(num_shards):
            shard_name = f"{save_name}_shard_{rank}.scp"
            shard_path = save_path / shard_name

            if not shard_path.exists():
                logger.warning(f"Shard file {shard_path} not found!")
                continue

            with open(shard_path, "r") as infile:
                for line in infile:
                    outfile.write(line)

    logger.info(f"Merged scp file saved to {merged_scp}")


def _skip(save_path, save_name, conf):
    """
    Detects if the dataset extraction has been already done.
    If the extraction has been done, we can skip it.

    Arguments
    ---------
    save_path : str
        The path to the directory containing extracted tokens.
    save_name : str
        The base name of the saved tokens file.
    conf : dict
        Configuration to match against saved config.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    skip = True

    # Checking ark,scp files
    for ext in [".ark", ".scp"]:
        save_file = save_path / f"{save_name}{ext}"
        if not save_file.exists:
            skip = False

    # Checking saved options
    save_opt = save_path / f"opt_extract_{conf['rank']}.pkl"
    if skip is True:
        if save_opt.exists():
            opts_old = load_pkl(save_opt.as_posix())
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False
    return skip


class TokensLoader:
    """
    A loader class for retrieving tokens corresponding to utterance IDs.

    Arguments
    ---------
    data_path: str
        The path to the data directory containing the token files.
    save_name: str, optional
        The base name of the tokens files (default: "tokens").
    """

    def __init__(
        self,
        data_path,
        save_name="tokens",
    ):
        self.data_path = pl.Path(data_path)
        if not self.data_path.exists():
            raise ValueError(f"Data folder not found: {self.data_path.as_posix()}")
        self.tokens = self._load(data_path, save_name)

    def tokens_by_uttid(self, utt_id, num_codebooks=None):
        """
        Retrieves the tokens corresponding to a given utterance ID.

        Arguments
        ---------
        utt_id : str
            The utterance ID to retrieve tokens for.
        num_codebooks : int or list, optional
            The number of codebooks to retrieve from the tokens. If specified as an int, the tokens
            will be truncated to include only the first `num_codebooks` codebooks. If specified as a list,
            the tokens will include only the codebooks at the specified indices. If not specified, all codebooks are returned.

        Returns
        -------
        result : torch.LongTensor [T, N_Q]
            The tokens associated with the utterance ID, possibly truncated to `num_codebooks` codebooks.

        Raises
        ------
        KeyError
            If the utterance ID is not found in the tokens.
        ValueError
            If `num_codebooks` is invalid or exceeds the number of available codebooks.
        """
        if utt_id not in self.tokens:
            raise KeyError(f"Utterance ID '{utt_id}' not found in tokens.")
        tokens_path = self.tokens[utt_id]
        tokens = kaldiio.load_mat(tokens_path)
        tokens = torch.from_numpy(tokens).long()

        if num_codebooks is not None:
            if isinstance(num_codebooks, int):
                if num_codebooks <= 0:
                    raise ValueError(
                        f"Invalid num_codebooks value: {num_codebooks}. It must be a positive integer."
                    )
                if num_codebooks > tokens.size(-1):
                    raise ValueError(
                        f"Invalid number of codebooks: {num_codebooks}. "
                        f"Available codebooks: {tokens.size(-1)}."
                    )
                tokens = tokens[:, :num_codebooks]
            elif isinstance(num_codebooks, list):
                if not all(
                    isinstance(idx, int) and 0 <= idx < tokens.size(-1)
                    for idx in num_codebooks
                ):
                    raise ValueError(
                        f"Invalid indices in num_codebooks list: {num_codebooks}. "
                        f"All indices must be integers within the range [0, {tokens.size(-1) - 1}]."
                    )
                tokens = tokens[:, num_codebooks]
            else:
                raise ValueError("num_codebooks must be an int or a list.")

        return tokens

    def _load(self, data_path, save_name):
        """
        Loads the mapping from utterance IDs to token file paths.

        Arguments
        ---------
        data_path: str
            The path to the data directory containing the token files.
        save_name: str
            The base name of the tokens files.

        Returns
        -------
        utt2toks: dict
            A dictionary mapping utterance IDs to their corresponding token file paths.
        """
        scp_path = f"{data_path}/{save_name}.scp"
        with open(scp_path, "r") as f:
            utt2toks = {
                line.strip().split(None, 1)[0]: line.strip().split(None, 1)[1]
                for line in f
                if line.strip()
            }
        return utt2toks

    def load_pretrained_embeddings(self, data_path, save_name="embeddings"):
        """
        Loads pretrained embeddings from a specified path.

        Arguments
        ---------
        data_path : str
            The directory where the embeddings are saved.
        save_name : str, optional
            The name of the embeddings file (default: "embeddings").

        Returns
        -------
        embeddings : torch.Tensor
            The loaded embeddings as a PyTorch tensor.

        Raises
        ------
        FileNotFoundError
            If the embeddings file does not exist at the specified path.
        """
        data_path = pl.Path(data_path).absolute()
        if not self.data_path.exists():
            raise ValueError(f"Data folder not found: {data_path.as_posix()}")
        embeddings = np.load(data_path / f"{save_name}.npy")
        embeddings = torch.from_numpy(embeddings)
        return embeddings