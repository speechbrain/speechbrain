"""UTMOS.
Authors
 * Jarod Duret 2024
 * Artem Ploujnikov 2024 (cosmetic changes only)
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from speechbrain.utils.fetching import fetch
from speechbrain.utils.metric_stats import MetricStats

__all__ = ["UTMOSModel", "UTMOSMetric"]


SAMPLE_RATE = 16000
DEFAULT_ENCODER_HUB = "chaanks/wav2vec2-small"
DEFAULT_MODEL_URL = "https://huggingface.co/chaanks/UTMOS/resolve/main"
DEFAULT_MODEL_NAME = "utmos.ckpt"
DEFAULT_SAVE_DIR = "./pretrained_models"
DEFAULT_JUDGE_ID = 288
DEFAULT_DOMAIN_ID = 0


class UTMOSModel(nn.Module):
    """The UTMOS model wrapper

    Arguments
    ---------
    source : str
        The WavLM source
    save_path : str | path-like
        The path where the model will be saved
    features_dim : int, optional
        The features dimension
    num_domains : int, optional
        The number of domains
    domain_dim : int, optional
        The dimension of each domain
    num_judges : int, optional
        The number of "judges"
    judge_dim : int, optional
        The dimension of each judge
    decoder_hidden_size : int, optional
        The size of the decoder hidden state
    multiplier : float, optional
        The number that the raw model output is multiplied by
        to compute the score
    offset : float, optional
        The number that (raw output * multiplier) will be added
        to in order to get the score
    """

    def __init__(
        self,
        source,
        save_path,
        features_dim=768,
        num_domains=3,
        domain_dim=128,
        num_judges=3000,
        judge_dim=128,
        decoder_hidden_size=512,
        multiplier=2.0,
        offset=3.0,
    ):
        super().__init__()

        self.ssl_encoder = Wav2Vec2(
            source,
            save_path,
            freeze=True,
            output_norm=False,
            freeze_feature_extractor=True,
            output_all_hiddens=False,
        )

        self.domain_embedding = nn.Embedding(num_domains, domain_dim)
        self.judge_embedding = nn.Embedding(num_judges, judge_dim)

        self.decoder = nn.LSTM(
            input_size=features_dim + domain_dim + judge_dim,
            hidden_size=decoder_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(decoder_hidden_size * 2, 2048),
            torch.nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1),
        )
        self.multiplier = multiplier
        self.offset = offset

    def forward(self, wav, domain_id=None, judge_id=None):
        """Computes the forward pass

        Arguments
        ---------
        wav : torch.Tensor
            The raw waveforms
        domain_id : torch.Tensor
            The domain identifiers
        judge_id : torch.Tensor
            The judge identifier

        Returns
        -------
        result : torch.Tensor
            The predicted rating(s)
        """

        if domain_id is None:
            domain_id = torch.zeros(
                len(wav), dtype=torch.int, device=wav.device
            )
        if judge_id is None:
            judge_id = (
                torch.ones(len(wav), dtype=torch.int, device=wav.device)
                * DEFAULT_JUDGE_ID
            )

        ssl_features = self.ssl_encoder(wav)
        domain_emb = self.domain_embedding(domain_id)
        judge_emb = self.judge_embedding(judge_id)

        domain_emb = domain_emb.unsqueeze(1).expand(
            -1, ssl_features.size(1), -1
        )
        judge_emb = judge_emb.unsqueeze(1).expand(-1, ssl_features.size(1), -1)
        concatenated_feature = torch.cat(
            [ssl_features, domain_emb, judge_emb], dim=2
        )

        decoder_output, _ = self.decoder(concatenated_feature)
        pred = self.classifier(decoder_output)

        return pred.mean(dim=1).squeeze(1) * self.multiplier + self.offset


class UTMOSMetric(MetricStats):
    """A metric implementing UTMOS

    Arguments
    ---------
    sample_rate : int
        The audio sample rate
    save_path : str | path-like, optional
        The path where the model will be saved
    encoder_hub : str`, optional
        The HuggingFace hube name for the encoder
    model_name : str, optional
        The name of the model
    model_url : str, optional
        The download URL for the model
    features_dim : int, optional
        The features dimension
    num_domains : int, optional
        The number of domains
    domain_dim : int, optional
        The dimension of each domain
    num_judges : int, optional
        The number of "judges"
    judge_dim : int, optional
        The dimension of each judge
    decoder_hidden_size : int, optional
        The size of the decoder hidden state
    domain_id : int, optional
        The domain identifier
    judge_id : int, optional
        The judge identifier
    """

    def __init__(
        self,
        sample_rate,
        save_path=None,
        encoder_hub=None,
        model_name=None,
        model_url=None,
        features_dim=768,
        num_domains=3,
        domain_dim=128,
        num_judges=3000,
        judge_dim=128,
        decoder_hidden_size=512,
        domain_id=None,
        judge_id=None,
    ):
        self.sample_rate = sample_rate
        self.clear()

        if encoder_hub is None:
            encoder_hub = DEFAULT_ENCODER_HUB
        if model_name is None:
            model_name = DEFAULT_MODEL_NAME
        if save_path is None:
            save_path = DEFAULT_SAVE_DIR
        if domain_id is None:
            domain_id = DEFAULT_DOMAIN_ID
        if judge_id is None:
            judge_id = DEFAULT_JUDGE_ID

        encoder_path = Path(save_path) / "encoder"
        self.model = UTMOSModel(
            source=encoder_hub,
            save_path=encoder_path.as_posix(),
            features_dim=features_dim,
            num_domains=num_domains,
            domain_dim=domain_dim,
            num_judges=num_judges,
            judge_dim=judge_dim,
            decoder_hidden_size=decoder_hidden_size,
        )

        # Download utmos model checkpoint
        fetch(model_name, model_url, save_path)
        model_path = Path(save_path) / model_name
        assert model_path.exists()

        # Load weights
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.domain_id = domain_id
        self.judge_id = judge_id

    def append(
        self, ids, hyp_audio, lens=None, domain_ids=None, judge_ids=None
    ):
        """Computes the UTMOS metric for the provided audio

        Arguments
        ---------
        ids : list
            The list of item IDs
        hyp_audio : torch.Tensor
            The audio prediction to be evaluated (e.g. TTS output)
        lens : torch.Tensor, optional
            Relative lengths
        domain_ids : torch.Tensor, optional
            The domain IDs. The default will be used if not provided
        judge_ids : torch.Tensor
            The judge IDs. The default will be used if not provided
        """
        assert hyp_audio.ndim == 2

        # Resample
        hyp_audio = torchaudio.functional.resample(
            hyp_audio, self.sample_rate, SAMPLE_RATE
        )

        self.model.device = hyp_audio.device
        self.model.to(hyp_audio.device)

        if domain_ids is None:
            domain_ids = torch.zeros(
                len(hyp_audio), dtype=torch.int, device=hyp_audio.device
            )
        if judge_ids is None:
            judge_ids = (
                torch.ones(
                    len(hyp_audio), dtype=torch.int, device=hyp_audio.device
                )
                * self.judge_id
            )

        output = self.model(hyp_audio, domain_ids, judge_ids)
        self.scores += output.cpu().tolist()

        self.ids += ids
