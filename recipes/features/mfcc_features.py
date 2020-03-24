import torch
from speechbrain.utils.data_utils import load_extended_yaml
from speechbrain.processing.features import (
    STFT,
    spectrogram,
    FBANKs,
    MFCCs,
    deltas,
    context_window,
)


class mfcc_features(torch.nn.Module):
    def __init__(self, sample_rate, overrides={}):
        super().__init__()
        # path = 'recipes/features/features.yaml'
        # self.params = load_extended_yaml(open(path), overrides)
        params = overrides.get('STFT', {})
        self.compute_STFT = STFT(sample_rate, **params)
        params = overrides.get('spectrogram', {})
        self.compute_spectrogram = spectrogram(**params)
        params = overrides.get('FBANKs', {})
        self.compute_fbanks = FBANKs(**params)
        params = overrides.get('MFCCs', {})
        self.compute_mfccs = MFCCs(**params)
        params = overrides.get('deltas', {})
        self.compute_deltas = deltas(**params)
        params = overrides.get('context_window', {})
        self.context_window = context_window(**params)

    def forward(self, wav):

        # mfcc computation pipeline
        STFT = self.compute_STFT(wav)
        spectr = self.compute_spectrogram(STFT)
        FBANKs = self.compute_fbanks(spectr)
        MFCCs = self.compute_mfccs(FBANKs)

        # computing derivatives
        delta1 = self.compute_deltas(MFCCs)
        delta2 = self.compute_deltas(delta1)

        # concatenate mfcc+delta1+delta2
        mfcc_with_deltas = torch.cat([MFCCs, delta1, delta2], dim=-2)

        # applying the context window
        return self.context_window(mfcc_with_deltas)
