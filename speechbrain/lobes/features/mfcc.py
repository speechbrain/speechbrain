"""
Authors: Mirco Ravanelli 2020, Peter Plantinga 2020
"""
import os
import torch
from speechbrain.yaml import load_extended_yaml
from speechbrain.processing.features import spectral_magnitude


class MFCC(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Arguments
    ---------
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
        parameters from `mfcc.yaml`

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = MFCC()
    >>> feats = feature_maker(inputs, init_params=True)
    >>> feats.shape
    torch.Size([10, 101, 660])

    Hyperparams
    -----------
        .. include:: mfcc.yaml
    """

    def __init__(
        self, deltas=True, context=True, requires_grad=False, **overrides,
    ):
        super().__init__()
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "mfcc.yaml")
        self.params = load_extended_yaml(open(path), overrides)

    def forward(self, wav, init_params=False):
        """Returns a set of mfccs generated from the input waveforms.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        STFT = self.params.compute_STFT(wav)
        mag = spectral_magnitude(STFT)
        fbanks = self.params.compute_fbanks(mag, init_params)
        mfccs = self.params.compute_dct(fbanks, init_params)

        if self.deltas:
            delta1 = self.params.compute_deltas(mfccs, init_params)
            delta2 = self.params.compute_deltas(delta1)
            mfccs = torch.cat([mfccs, delta1, delta2], dim=2)

        if self.context:
            mfccs = self.params.context_window(mfccs)

        return mfccs
