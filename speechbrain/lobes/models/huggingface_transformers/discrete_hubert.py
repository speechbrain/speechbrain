"""This lobe enables the integration of pretrained discrete Hubert.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Reference: https://arxiv.org/abs/2110.13900
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Author
 * Pooneh Mousavi 2023

"""
import logging
import torch
import joblib
from huggingface_hub import snapshot_download
from pathlib import Path
import os
import numpy as np
from speechbrain.lobes.models.huggingface_transformers.hubert import HuBERT

logger = logging.getLogger(__name__)


class DiscreteHuBERT(HuBERT):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained Discrete HuBERT models.

    Source paper HuBERT: https://arxiv.org/abs/2106.07447
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed Discrete feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    For now, HuggingFace's HuBERT and WavLM model can be loaded using the exact code for Wav2Vec2 model.
    For this reason, HuBERT and WavLM can be fine inheriting the Wav2Vec2 class.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/hubert-base-ls960"
    save_path : str
        Path (dir) of the downloaded model.
    kmeans_repo_id : str
        Huggingface repository if that contains the pretrained kmean model
    kmeans_dataset : str
        Name of the dataset that Kmeans model on HF repo is trained with.
    kmeans_cache_dir: str
        Path (dir) of the downloaded kmeans model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the HuBERT model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the featue_extractor module of the model is Frozen. If False
        all the HuBERT model will be trained including featue_extractor module.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface Hubert Model() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    output_all_hiddens : bool (default: True)
        If True, the forward function outputs the hidden states from all transformer layers.
        For example facebook/hubert-base-ls960 has 12 transformer layers and the output is of shape (13, B, T, C),
        where a projection of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer.
    num_clusters:  (int) (default: 128)
        determine the number of clusters of the targeted kmeans models to be downloaded.
 


    Example
    -------
    >>> import torch
    >>> inputs = torch.rand([10, 2000])
    >>> model_hub = "facebook/hubert-base-ls960"
    >>> save_path = "savedir"
    >>> ssl_layer_num = [5,7]
    >>> kmeans_repo_id = "speechbrain/SSL_Quantization"
    >>> kmeans_dataset = "LibriSpeech-100-360-500"
    >>> kmeans_cache_dir="savedir"
    >>> num_clusters = 128

    >>> model = DiscreteHuBERT(model_hub, save_path,freeze = True,kmeans_repo_id=kmeans_repo_id, kmeans_dataset=kmeans_dataset, kmeans_cache_dir=kmeans_cache_dir,num_clusters=num_clusters)
    >>> embs, tokens = model(inputs,ssl_layer_num=ssl_layer_num,deduplicte=True)
    >>> print(embs.shape)
    torch.Size([10, 4, 2])
    >>> print(tokens.shape)
    torch.Size([10, 4, 2, 768])
    """

    def __init__(
        self,
        source,
        save_path,
        kmeans_dataset,
        kmeans_cache_dir,
        kmeans_repo_id="speechbrain/SSL_Quantization",
        output_norm=False,
        freeze=False,
        freeze_feature_extractor=False,
        apply_spec_augment=False,
        output_all_hiddens=True,
        num_clusters = 128,
    ):
        super().__init__(
            source=source,
            save_path=save_path,
            output_norm=output_norm,
            freeze=freeze,
            freeze_feature_extractor=freeze_feature_extractor,
            apply_spec_augment=apply_spec_augment,
            output_all_hiddens=output_all_hiddens,
        )

        self.kmeans_models , self.ssl_layer_ids = self.load_kmeans(
            kmeans_repo_id, kmeans_dataset, num_clusters, kmeans_cache_dir
        )

        self.vocabularies = []
        for model in self.kmeans_models:
            self.vocabularies.append(model.cluster_centers_)

        self.num_clusters = num_clusters

    def load_kmeans(self, repo_id, kmeans_dataset,num_clusters , cache_dir):
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

        kmeans_models =[]
        layer_ids=[]
        file_pattern = f"{kmeans_dataset}/hubert/*_k{num_clusters}*.pt"
        kmeans_dir = snapshot_download(repo_id=repo_id, allow_patterns=file_pattern,cache_dir=cache_dir)
        files= Path( os.path.join(kmeans_dir,kmeans_dataset,'hubert')).glob('*.pt')
        for file in files:
            layer_ids.append(int(file.name.split('/')[-1].split('_')[-1].split(".")[0][1:]))
            kmeans_models.append(joblib.load(file))
        layer_ids, kmeans_models = zip(*sorted(zip(layer_ids, kmeans_models)))
        return kmeans_models,layer_ids

    def forward(self, wav, wav_lens=None, ssl_layer_num=[7], deduplicte=False):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_len : tensor
            The relative length of the wav given in SpeechBrain format.
        ssl_layer_num: List[int] (default: [7]): 
            determine which layers of SSL should be used to extract information.
        deduplicte: False (default: False)
            whether to remove duplicate tokens or not.
        Returns:
        ---------
        tokens : torch.Tensor
            A (Batch x Seq) tensor of audio tokens
        emb : torch.Tensor
            A (Batch x Seq x embedding_dim ) cluster_centers embeddings for each tokens
        """

        # If we freeze, we simply remove all grads from the graph.
        embeddings=[]
        token_ids=[]
        bs= wav.shape[0]
         
        # check the availability of the input SSL layers.
        layers=[]
        for layer in ssl_layer_num:
            if layer not in self.ssl_layer_ids:
                logger.warn(f"Layer {layer} is not among trained layers for kmenas: {self.ssl_layer_ids}. We will igoner this layer when computing the tokens.")
            else:
                layers.append(layer)
        if len(layers) == 0:
            raise ValueError(f"None of the passed layer numbers are among the suppoorted ones: {self.ssl_layer_ids} ")
   

        
        with torch.no_grad():
            feats = self.extract_features(wav, wav_lens)
            for layer_num, model, vocabulary in zip(self.ssl_layer_ids,self.kmeans_models,self.vocabularies):
                if layer_num not in ssl_layer_num:
                    continue
                tokens = model.predict(feats[layer_num].flatten(end_dim=-2).cpu())
                tokens = tokens.reshape(bs,-1)

                if deduplicte:    
                    # assign unique token-ids for each quantizer (first later start from 0-num-cluster, second layer start from 1*num-cluster, .. 2*num-cluster,.... )
                    tokens = tokens.reshape(bs,-1)
                    unique_token_ids = [row[np.diff(row, prepend=np.nan).astype(bool)] for row in tokens]
                    layer_token_ids = [torch.tensor(row+ self.num_clusters*layer_num, dtype=torch.long,device=wav.device) for row in unique_token_ids]
                    token_ids.extend(layer_token_ids)
                    embs = [torch.tensor(vocabulary[row], dtype=torch.float,device=wav.device) for row in unique_token_ids]
                    embeddings.extend(embs) 
                else:
                    layer_token_ids = [torch.tensor(row+ self.num_clusters*layer_num, dtype=torch.long,device=wav.device) for row in tokens]
                    token_ids.extend(layer_token_ids)
                    embs = [torch.tensor(vocabulary[row], dtype=torch.float,device=wav.device) for row in tokens]
                    embeddings.extend(embs) 

        return (
            torch.stack(torch.split( torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True), bs),dim=2),
            torch.stack(torch.split( torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True), bs),dim=2)
        )
