"""This lobe enables the integration of pretrained discrete wav2vec2 model.

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
from huggingface_hub import hf_hub_download
import joblib

from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2

logger = logging.getLogger(__name__)


class DiscreteWav2Vec2(Wav2Vec2):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained Discrete Wav2Vec2 models.

     Source paper wav2vec2.0: https://arxiv.org/abs/2006.11477
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed Discrete feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    kmeans_repo_id : str
        Huggingface repository if that contains the pretrained kmean model
    kmeans_filename : str
        Name of the file in HF repo that need to be downloaded.
    kmeans_cache_dir: str
        Path (dir) of the downloaded kmeans model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the Wav2Vec2 model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the featue_extractor module of the model is Frozen. If False
        all the Wav2Vec2 model will be trained including featue_extractor module.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface Wav2Vec2 Model() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    output_all_hiddens : bool (default: True)
        If True, the forward function outputs the hidden states from all transformer layers.
        For example facebook/wav2vec2-large-lv60 has 12 transformer layers and the output is of shape (13, B, T, C),
        where a projection of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer.
    ssl_layer_num : (int) (default: -1)
        determine the output of which layer of the SSL model should be used for clustering.


    Example
    -------
    >>> import torch
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-large-lv60"
    >>> save_path = "savedir"
    >>> ssl_layer_num = -1
    >>> kmeans_repo_id = "speechbrain/SSL_Quantization"
    >>> kmeans_filename = "LibriSpeech_wav2vec_k128_L7.pt"
    >>> kmeans_cache_dir="savedir"
    >>> model = DiscreteWav2Vec2(model_hub, save_path,freeze = True,ssl_layer_num=ssl_layer_num,kmeans_repo_id=kmeans_repo_id, kmeans_filename=kmeans_filename, kmeans_cache_dir=kmeans_cache_dir)
    >>> embs, tokens = model(inputs)
    >>> embs.shape
    torch.Size([10, 1, 1024])
    >>> tokens.shape
    torch.Size([10, 1])
    """

    def __init__(
        self,
        source,
        save_path,
        kmeans_filename,
        kmeans_cache_dir,
        kmeans_repo_id="speechbrain/SSL_Quantization",
        output_norm=False,
        freeze=False,
        freeze_feature_extractor=False,
        apply_spec_augment=False,
        output_all_hiddens=True,
        ssl_layer_num=-1,
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

        self.kmeans = self.load_kmeans(
            kmeans_repo_id, kmeans_filename, kmeans_cache_dir
        )
        self.vocabulary = self.kmeans.cluster_centers_
        self.ssl_layer_num = ssl_layer_num

    def load_kmeans(self, repo_id, filename, cache_dir):
        """Load a Pretrained kmeans model from HF.

        Arguments
        ---------
        repo_id : str
           The hugingface repo id that contains the model.
        filename : str
            The name of the checkpoints in the repo that need to be downloaded.
        cache_dir: str
            Path (dir) of the downloaded model.
        Returns:
        ---------
        kmeans_model : MiniBatchKMeans:
            pretrained Kmeans  model loaded from the HF.
        """
        kmeans_model = joblib.load(
            hf_hub_download(
                repo_id=repo_id, filename=filename, cache_dir=cache_dir
            )
        )
        return kmeans_model

    def forward(self, wav, wav_lens=None):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_len : tensor
            The relative length of the wav given in SpeechBrain format.
        Returns:
        ---------
        tokens : torch.Tensor
            A (Batch x Seq) tensor of audio tokens
        emb : torch.Tensor
            A (Batch x Seq x embedding_dim ) cluster_centers embeddings for each tokens
        """

        # If we freeze, we simply remove all grads from the graph.
        with torch.set_grad_enabled(not self.freeze):
            feats = self.extract_features(wav, wav_lens)[self.ssl_layer_num]
        tokens = self.kmeans.predict(feats.flatten(end_dim=-2).cpu())
        embs = self.vocabulary[tokens]
        return (
            torch.tensor(
                embs.reshape(wav.shape[0], -1, embs.shape[-1]),
                dtype=torch.float,
                device=wav.device,
            ),
            torch.tensor(
                tokens.reshape(wav.shape[0], -1),
                dtype=torch.long,
                device=wav.device,
            ),
        )
