import torch
from speechbrain.utils.data_utils import load_extended_yaml


class Features(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Args:
        feature_type: One of 'spectrogram', 'fbank', or 'mfcc', the type of
            feature to generate.
        deltas: Whether or not to append derivatives and second derivatives
            to the features.
        context: Whether or not to append forward and backward contexts to
            the features.
        requires_grad: Whether to allow parameters (i.e. fbank centers and
            spreads) to update during training.
        overrides: A set of overrides to use when reading the default
            parameters from `features.yaml`

    Shapes:
        - wav: [batch, time_steps] or [batch, channels, time_steps]
        - output: see corresponding documentation in relevant feature
            function, in `speechbrain/processing/features.py`

    Example:
        >>> import torch
        >>> inputs = torch.randn([10, 16000])
        >>> feature_maker = Features(feature_type='fbank')
        >>> feats = feature_maker(inputs)
        >>> feats.shape
        torch.Size([10, 759, 101])

    .. literalinclude:: /../../speechbrain/lobes/features.yaml
        :language: yaml

    Authors:
        Mirco Ravanelli and Peter Plantinga 2020
    """
    def __init__(
        self,
        feature_type='spectrogram',
        deltas=True,
        context=True,
        normalize=True,
        requires_grad=False,
        **overrides
    ):
        super().__init__()
        self.feature_type = feature_type
        self.deltas = deltas
        self.context = context
        self.normalize = normalize
        self.requires_grad = requires_grad
        path = 'speechbrain/lobes/features.yaml'
        self.params = load_extended_yaml(open(path), overrides)

    def forward(self, wav, wav_len=None):
        STFT = self.params['compute_STFT'](wav)
        features = self.params['compute_spectrogram'](STFT)

        if self.feature_type in ['fbank', 'mfcc']:
            features = self.params['compute_fbanks'](features)
        if self.feature_type == 'mfcc':
            features = self.params['compute_mfccs'](features)

        if self.deltas:
            delta1 = self.params['compute_deltas'](features)
            delta2 = self.params['compute_deltas'](delta1)
            features = torch.cat([features, delta1, delta2], dim=-2)

        if self.context:
            features = self.params['context_window'](features)

        if self.normalize:
            features = self.params['normalize'](features, wav_len)

        return features
