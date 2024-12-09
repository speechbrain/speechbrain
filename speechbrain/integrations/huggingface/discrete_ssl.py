"""This lobe enables the integration of pretrained discrete SSL (hubert,wavlm,wav2vec) for extracting semnatic tokens from output of SSL layers.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Author
 * Pooneh Mousavi 2024
 * Jarod Duret 2024
"""

import os
from glob import glob

import joblib
import torch
from huggingface_hub import snapshot_download
from torch import nn

from speechbrain.inference.vocoders import UnitHIFIGAN
from speechbrain.tokenizers.discrete_SSL_tokenizer import DiscreteSSLTokenizer
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class DiscreteSSL(nn.Module):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained Discrete SSL models.

    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed Discrete feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    The following table summarizes the compatible SSL models, their respective HF encoders, k-means training details, supported layers, and pretrained vocoder:

    | SSL Model  | HF Encoder                             | K-Means Dataset | K-Means Size | SSL Layers           | Vocoder Model                            |
    |------------|----------------------------------------|-----------------|--------------|----------------------|------------------------------------------|
    | WavLM      | microsoft/wavlm-large                  | LibriSpeech960  | 1000         | 1, 3, 7, 12, 18, 23  | speechbrain/hifigan-wavlm-k1000-LibriTTS |
    | HuBERT     | facebook/hubert-large-ll60k            | LibriSpeech960  | 1000         | 1, 3, 7, 12, 18, 23  | WIP                                      |
    | Wav2Vec2   | facebook/wav2vec2-large-960h-lv60-self | LibriSpeech960  | 1000         | 1, 3, 7, 12, 18, 23  | WIP                                      |


    Arguments
    ---------
    save_path : str
        Path (dir) of the downloaded model.
    ssl_model : str
        SSL model to extract semantic tokens from its layers' output. Note that output_all_hiddens should be set to True to enable multi-layer discretization.
    kmeans_dataset : str
        Name of the dataset that Kmeans model on HF repo is trained with.
    vocoder_repo_id: str
        Huggingface repository that contains the pre-trained HiFi-GAN model.
    num_clusters : int or List[int] (default: 1000)
        Determine the number of clusters of the targeted kmeans models to be downloaded. It could be varying for each layer.
    layers_num : List[int] (Optional)
        Detremine layers to be download from HF repo. If it is not provided, all layers with num_clusters(int) is loaded from HF repo. If num_clusters is a list, the layers_num should be provided to determine the cluster number for each layer.
    device : str (default 'cpu')
        The device to use for computation ('cpu' or 'cuda').
    sample_rate : int (default: 16000)
        Sample rate of the input audio.
    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.huggingface_transformers.wavlm import (WavLM)
    >>> inputs = torch.rand([3, 2000])
    >>> model_hub = "microsoft/wavlm-large"
    >>> save_path = "savedir"
    >>> ssl_layer_num = [7,23]
    >>> deduplicate =[False, True]
    >>> bpe_tokenizers=[None, None]
    >>> vocoder_repo_id = "speechbrain/hifigan-wavlm-k1000-LibriTTS"
    >>> kmeans_dataset = "LibriSpeech"
    >>> num_clusters = 1000
    >>> ssl_model = WavLM(model_hub, save_path,output_all_hiddens=True)
    >>> model = DiscreteSSL(save_path, ssl_model, vocoder_repo_id=vocoder_repo_id, kmeans_dataset=kmeans_dataset,num_clusters=num_clusters)
    >>> tokens, _, _ = model.encode(inputs,SSL_layers=ssl_layer_num, deduplicates=deduplicate, bpe_tokenizers=bpe_tokenizers)
    >>> print(tokens.shape)
    torch.Size([3, 6, 2])
    >>> sig = model.decode(tokens, ssl_layer_num)
    >>> print(sig.shape)
    torch.Size([3, 1, 1920])
    """

    def __init__(
        self,
        save_path,
        ssl_model,
        kmeans_dataset,
        vocoder_repo_id="speechbrain/hifigan-wavlm-k1000-LibriTTS",
        num_clusters=1000,
        layers_num=None,
        device="cpu",
        sample_rate=16000,
    ):

        super().__init__()
        self.device = device
        self.ssl_model = ssl_model
        model_name = ssl_model.__class__.__name__.lower()
        self.check_if_input_is_compatible(layers_num, num_clusters)

        self.kmeans_models, self.ssl_layer_ids, self.num_clusters = (
            self.load_kmeans(
                vocoder_repo_id,
                kmeans_dataset,
                model_name,
                self.num_clusters,
                save_path,
                layers_num,
            )
        )

        self.vocabularies = []
        for model in self.kmeans_models:
            self.vocabularies.append(model.cluster_centers_)

        self.tokenizer = DiscreteSSLTokenizer(self.num_clusters)
        self.codec_vocoder = UnitHIFIGAN.from_hparams(
            source=vocoder_repo_id,
            savedir=save_path,
        )
        self.codec_vocoder.tokenize = False
        self.sample_rate = sample_rate

    def check_if_input_is_compatible(self, layers_num, num_clusters):
        """check if layer_number and num_clusters is consistent with each other.

        Arguments
        ---------
        layers_num: List[int] (Optional)
            If num_clusters is a list, the layers_num should be provided to determine the cluster number for each layer.
        num_clusters: int or List[int]
            determine the number of clusters of the targeted kmeans models to be downloaded. It could be varying for each layer.
        """

        if layers_num:
            if isinstance(num_clusters, int):
                num_clusters = [num_clusters for i in layers_num]
            assert len(num_clusters) == len(
                layers_num
            ), "length of num_clusters and layers_num should be the same!!!"
        if layers_num is None:
            assert isinstance(
                num_clusters, int
            ), "num_clusters is expected to be int since the layers_num is not provided."
        self.num_clusters = num_clusters

    def load_kmeans(
        self,
        repo_id,
        kmeans_dataset,
        encoder_name,
        num_clusters,
        cache_dir,
        layers_num=None,
    ):
        """Load a Pretrained kmeans model from HF.

        Arguments
        ---------
        repo_id : str
           The hugingface repo id that contains the model.
        kmeans_dataset : str
            Name of the dataset that Kmeans model are trained with in HF repo that need to be downloaded.
        encoder_name : str
            Name of the encoder for locating files.
        num_clusters : int or List[int]
            determine the number of clusters of the targeted kmeans models to be downloaded. It could be varying for each layer.
        cache_dir : str
            Path (dir) of the downloaded model.
        layers_num : List[int] (Optional)
            If num_clusters is a list, the layers_num should be provided to determine the cluster number for each layer.

        Returns
        -------
        kmeans_model : MiniBatchKMeans
            pretrained Kmeans  model loaded from the HF.
        layer_ids : List[int]
            supported layer nums for kmeans (extracted from the name of kmeans model.)
        """

        kmeans_models = []
        layer_ids = []
        file_patterns = []
        if layers_num:
            for i, layer in enumerate(layers_num):
                file_patterns.append(
                    f"kmeans/{kmeans_dataset}_{encoder_name}_k{num_clusters[i]}_L{layer}.pt"
                )
        else:
            file_patterns.append(
                f"kmeans/{kmeans_dataset}_{encoder_name}_k{num_clusters}*.pt"
            )
        kmeans_dir = snapshot_download(
            repo_id=repo_id, allow_patterns=file_patterns, cache_dir=cache_dir
        )
        files = []
        for ext in file_patterns:
            for file in glob(os.path.join(kmeans_dir, ext)):
                if file not in files:
                    files.append(file)
                    layer_ids.append(
                        int(
                            file.split("/")[-1].split("_")[-1].split(".")[0][1:]
                        )
                    )
                    kmeans_models.append(joblib.load(file))

        assert (
            len(layer_ids) > 0
        ), f"There is no trained k-means model available for {repo_id}"

        if isinstance(num_clusters, int):
            num_clusters = [num_clusters for i in layer_ids]
        layer_ids, kmeans_models, num_clusters = zip(
            *sorted(zip(layer_ids, kmeans_models, num_clusters))
        )

        return kmeans_models, layer_ids, num_clusters

    def forward(
        self,
        wav,
        wav_lens=None,
        SSL_layers=None,
        deduplicates=None,
        bpe_tokenizers=None,
    ):
        """Takes an input waveform and return its corresponding tokens and reconstructed signal.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_lens : tensor
            The relative length of the wav given in SpeechBrain format.
        SSL_layers: List[int]:
            determine which layers of SSL should be used to extract information.
        deduplicates: List[boolean]:
            determine to apply deduplication(remove duplicate subsequent tokens) on the tokens extracted for the corresponding layer.
        bpe_tokenizers: List[int]:
            determine to apply subwording on the tokens extracted for the corresponding layer if the sentencePiece tokenizer is trained for that layer.

        Returns
        -------
        tokens : torch.Tensor
            A (Batch x Seq x num_SSL_layers) tensor of audio tokens
        waveforms: torch.tensor
            Batch of mel-waveforms [batch, time]
        """

        tokens = self.encode(
            wav, wav_lens, SSL_layers, deduplicates, bpe_tokenizers
        )[0]
        sig = self.decode(tokens, SSL_layers=SSL_layers)
        return tokens, sig

    def encode(
        self,
        wav,
        wav_lens=None,
        SSL_layers=None,
        deduplicates=None,
        bpe_tokenizers=None,
    ):
        """Takes an input waveform and return its corresponding encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_lens : tensor
            The relative length of the wav given in SpeechBrain format.
        SSL_layers: List[int]:
            determine which layers of SSL should be used to extract information.
        deduplicates: List[boolean]:
            determine to apply deduplication(remove duplicate subsequent tokens) on the tokens extracted for the corresponding layer.
        bpe_tokenizers: List[int]:
            determine to apply subwording on the tokens extracted for the corresponding layer if the sentencePiece tokenizer is trained for that layer.

        Returns
        -------
        tokens : torch.Tensor
            A (Batch x Seq x num_SSL_layers) tensor of audio tokens
        emb : torch.Tensor
            A (Batch x Seq x num_SSL_layers x embedding_dim ) cluster_centers embeddings for each tokens
        processed_tokens : torch.Tensor
            A (Batch x Seq x num_SSL_layers) tensor of audio tokens after applying deduplication and subwording if necessary.
        """

        if SSL_layers is None:
            SSL_layers = self.ssl_layer_ids
        if deduplicates is None:
            deduplicates = [False] * len(SSL_layers)
        if bpe_tokenizers is None:
            bpe_tokenizers = [None] * len(SSL_layers)

        assert (
            len(deduplicates) == len(SSL_layers) == len(bpe_tokenizers)
        ), "length of SSL_layers,deduplicates,bpe_tokenizers should be the same!!!"

        embeddings = []
        token_ids = []

        for layer in SSL_layers:
            if layer not in self.ssl_layer_ids:
                raise ValueError(
                    f"Layer {layer} is not among trained layers for k-means. Supported layers are: {self.ssl_layer_ids}."
                )

        with torch.no_grad():
            feats = self.ssl_model.extract_features(wav, wav_lens)
            for layer_num, model, vocabulary in zip(
                self.ssl_layer_ids, self.kmeans_models, self.vocabularies
            ):
                if layer_num not in SSL_layers:
                    continue
                tokens = model.predict(
                    feats[layer_num].flatten(end_dim=-2).cpu()
                )
                embs = vocabulary[tokens]
                embeddings.append(
                    torch.tensor(
                        embs.reshape(wav.shape[0], -1, embs.shape[-1]),
                        dtype=torch.float,
                        device=wav.device,
                    )
                )
                token_ids.append(
                    torch.tensor(
                        tokens.reshape(wav.shape[0], -1),
                        dtype=torch.long,
                        device=wav.device,
                    )
                )

        org_tokens = torch.stack(token_ids, 2)
        org_embedding = torch.stack(embeddings, 2)

        processed_tokens = self.tokenizer.encode(
            org_tokens, SSL_layers, deduplicates, bpe_tokenizers
        )
        return org_tokens, org_embedding, processed_tokens

    def decode(self, tokens, SSL_layers=None):
        """Takes an input waveform and return its corresponding waveform.
        Original source:
        https://github.com/speechbrain/benchmarks/blob/c87beb61d4747909a133d3e1b3a3df7c8eda1f08/
        benchmarks/DASB/Libri2Mix/separation/conformer/train_discrete_ssl.py#L44

        Arguments
        ---------
        tokens : torch.Tensor
            A (Batch, codes, layers) tensor of discrete units
        SSL_layers: List[int]:
            determine which layers of SSL should be used by the vocoder.

        Returns
        -------
        waveforms: torch.tensor
            Batch of mel-waveforms [batch, time]
        """

        assert all(
            cluster == self.num_clusters[0] for cluster in self.num_clusters
        ), "All values in num_clusters must be equal."
        num_clusters = self.num_clusters[0]

        offsets = torch.arange(
            0,
            len(self.ssl_layer_ids) * num_clusters,
            num_clusters,
            device=self.device,
        )

        layers = self.ssl_layer_ids
        if SSL_layers is not None:
            layers = SSL_layers

        offset_idxes = [self.ssl_layer_ids.index(x) for x in layers]
        offsets = offsets[offset_idxes]
        tokens = tokens + offsets + 1

        if len(layers) < len(self.ssl_layer_ids):
            full_tokens = torch.zeros(
                *tokens.shape[:2],
                len(self.ssl_layer_ids),
                dtype=tokens.dtype,
                device=self.device,
            )
            for i, idx in enumerate(offset_idxes):
                full_tokens[..., idx] = tokens[..., i]
            tokens = full_tokens

        return self.codec_vocoder(tokens)
