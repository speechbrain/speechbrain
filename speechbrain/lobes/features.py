"""
Authors: Mirco Ravanelli 2020, Peter Plantinga 2020
"""
import torch
from speechbrain.utils.data_utils import load_extended_yaml


class Features(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Arguments
    ---------
    feature_type : str
        One of 'spectrogram', 'fbank', or 'mfcc', the type of feature
        to generate.
    deltas : bool
        Whether or not to append derivatives and second derivatives
        to the features.
    context : bool
        Whether or not to append forward and backward contexts to
        the features.
    requires_grad : bool
        Whether to allow parameters (i.e. fbank centers and
        spreads) to update during training.
    **overrides
        A set of overrides to use when reading the default
        parameters from `features.yaml`

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = Features(feature_type='fbank')
    >>> feats = feature_maker(inputs, init_params=True)
    >>> feats.shape
    torch.Size([10, 101, 759])

    Hyperparams
    -----------
        .. include:: features.yaml
    """

    def __init__(
        self,
        feature_type="spectrogram",
        deltas=True,
        context=True,
        requires_grad=False,
        **overrides,
    ):
        super().__init__()
        self.feature_type = feature_type
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad
        path = "speechbrain/lobes/features.yaml"
        self.params = load_extended_yaml(open(path), overrides)
        self.compute_STFT = self.params["compute_STFT"]
        self.compute_spectrogram = self.params["compute_spectrogram"]
        self.compute_fbanks = self.params["compute_fbanks"]
        self.compute_mfccs = self.params["compute_mfccs"]
        self.compute_deltas = self.params["compute_deltas"]
        self.context_window = self.params["context_window"]

    def forward(self, wav, init_params=False):
        """Returns a set of features generated from the input waveforms.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        STFT = self.compute_STFT(wav)
        features = self.compute_spectrogram(STFT)

        if self.feature_type in ["fbank", "mfcc"]:
            features = self.compute_fbanks(features, init_params=init_params)
        if self.feature_type == "mfcc":
            features = self.compute_mfccs(features, init_params=init_params)

        if self.deltas:
            delta1 = self.compute_deltas(features, init_params=init_params)
            delta2 = self.compute_deltas(delta1)
            features = torch.cat([features, delta1, delta2], dim=2)

        if self.context:
            features = self.context_window(features)

        return features
