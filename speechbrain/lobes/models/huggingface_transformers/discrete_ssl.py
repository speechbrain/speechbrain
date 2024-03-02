"""This lobe enables the integration of pretrained discrete SSL( hubert,wavlm,wav2vec) for extracting semnatic tokens from output of SSL layers.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Author
 * Pooneh Mousavi 2024
"""
import logging
import torch
import joblib
from huggingface_hub import snapshot_download
from pathlib import Path
import os
from torch import nn
from speechbrain.tokenizers.discrete_SSL_tokenizer import DiscreteSSLTokenizer

logger = logging.getLogger(__name__)


class DiscreteSSL(nn.Module):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained Discrete SSL models.

    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed Discrete feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/hubert-base-ls960"
    save_path : str
        Path (dir) of the downloaded model.
    ssl_model : str
        SSL model to extract semantic tokens from its layers' output. Note that output_all_hiddens should be set to True to enable multi-layer discretenation.
    kmeans_repo_id : str
        Huggingface repository that contains the pre-trained k-means models.
    kmeans_dataset : str
        Name of the dataset that Kmeans model on HF repo is trained with.
    num_clusters:  (int) (default: 128)
        determine the number of clusters of the targeted kmeans models to be downloaded.



    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.huggingface_transformers.hubert import (HuBERT)
    >>> inputs = torch.rand([3, 2000])
    >>> model_hub = "facebook/hubert-large-ll60k"
    >>> save_path = "savedir"
    >>> ssl_layer_num = [7,23]
    >>> deduplicate =[False, True]
    >>> bpe_tokenizers=[None, None]
    >>> kmeans_repo_id = "speechbrain/SSL_Quantization"
    >>> kmeans_dataset = "LibriSpeech-100-360-500"
    >>> num_clusters = 1000
    >>> ssl_model = HuBERT(model_hub, save_path,output_all_hiddens=True)
    >>> model = DiscreteSSL(save_path, ssl_model, kmeans_repo_id=kmeans_repo_id, kmeans_dataset=kmeans_dataset,num_clusters=num_clusters)
    >>> tokens, embs ,pr_tokens= model(inputs,SSL_layers=ssl_layer_num, deduplicates=deduplicate, bpe_tokenizers=bpe_tokenizers)
    >>> print(tokens.shape)
    torch.Size([3, 6, 2])
    >>> rint(embs.shape)
    torch.Size([3, 6, 2, 1024])
    >>> print(pr_tokens.shape)
    torch.Size([3, 6, 2])
    """

    def __init__(
        self,
        save_path,
        ssl_model,
        kmeans_dataset,
        kmeans_repo_id="speechbrain/SSL_Quantization",
        num_clusters=128,
    ):

        super().__init__()

        self.ssl_model = ssl_model
        model_name = ssl_model.__class__.__name__.lower()
        self.kmeans_models, self.ssl_layer_ids = self.load_kmeans(
            kmeans_repo_id,
            kmeans_dataset,
            model_name,
            num_clusters,
            save_path,
        )

        self.vocabularies = []
        for model in self.kmeans_models:
            self.vocabularies.append(model.cluster_centers_)

        self.num_clusters = num_clusters

        self.tokenizer = DiscreteSSLTokenizer(self.num_clusters)

    def load_kmeans(
        self, repo_id, kmeans_dataset, encoder_name, num_clusters, cache_dir
    ):
        """Load a Pretrained kmeans model from HF.

        Arguments
        ---------
        repo_id : str
           The hugingface repo id that contains the model.
        kmeans_dataset : str
            Name of the dataset that Kmeans model are trained with in HF repo that need to be downloaded.
        cache_dir: str
            Path (dir) of the downloaded model.
        num_clusters:  (int)
            determine the number of clusters of the targeted kmeans models to be downloaded.
        Returns:
        ---------
        kmeans_model : MiniBatchKMeans:
            pretrained Kmeans  model loaded from the HF.
        layer_ids : List[int] :
            supported layer nums for kmeans (extracted from the name of kmeans model.)
        """

        kmeans_models = []
        layer_ids = []
        file_pattern = f"{kmeans_dataset}/{encoder_name}/*_k{num_clusters}*.pt"
        kmeans_dir = snapshot_download(
            repo_id=repo_id, allow_patterns=file_pattern, cache_dir=cache_dir
        )
        files = Path(
            os.path.join(kmeans_dir, kmeans_dataset, encoder_name)
        ).glob("*.pt")
        for file in files:
            layer_ids.append(
                int(file.name.split("/")[-1].split("_")[-1].split(".")[0][1:])
            )
            kmeans_models.append(joblib.load(file))
        assert (
            len(layer_ids) > 0
        ), f"There is no trained k-means model avaiable for {repo_id}/{file_pattern}"
        layer_ids, kmeans_models = zip(*sorted(zip(layer_ids, kmeans_models)))

        return kmeans_models, layer_ids

    def forward(
        self,
        wav,
        wav_lens=None,
        SSL_layers=[7],
        deduplicates=[False],
        bpe_tokenizers=[None],
    ):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_len : tensor
            The relative length of the wav given in SpeechBrain format.
        SSL_layers: List[int] (default: [7]):
            determine which layers of SSL should be used to extract information.
        deduplicates: List[boolean] (default: [False]):
            determine to apply deduplication(remove duplicate subsequent tokens) on the tokens extracted for the corresponding layer.
        bpe_tokenizers: List[int] (default: [None]):
            determine to apply subwording on the tokens extracted for the corresponding layer if the sentencePiece tokenizer is trained for that layer.
        Returns:
        ---------
        tokens : torch.Tensor
            A (Batch x Seq x num_SSL_layers) tensor of audio tokens
        emb : torch.Tensor
            A (Batch x Seq x num_SSL_layers x embedding_dim ) cluster_centers embeddings for each tokens
        processed_tokens : torch.Tensor
            A (Batch x Seq x num_SSL_layers) tensor of audio tokens after applying deduplication and subwording if necessary.
        """

        assert (
            len(deduplicates) == len(SSL_layers) == len(bpe_tokenizers)
        ), f"length of SSL_layers,deduplicates,bpe_tokenizers should be the same!!!"

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
