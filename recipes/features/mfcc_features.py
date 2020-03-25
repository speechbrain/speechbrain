import torch
from speechbrain.utils.data_utils import load_extended_yaml


class mfcc_features(torch.nn.Module):
    def __init__(self, **overrides):
        super().__init__()
        path = 'recipes/features/features.yaml'
        self.params = load_extended_yaml(open(path), overrides)

    def forward(self, wav):

        # mfcc computation pipeline
        STFT = self.params['compute_STFT'](wav)
        spectr = self.params['compute_spectrogram'](STFT)
        FBANKs = self.params['compute_fbanks'](spectr)
        MFCCs = self.params['compute_mfccs'](FBANKs)

        # computing derivatives
        delta1 = self.params['compute_deltas'](MFCCs)
        delta2 = self.params['compute_deltas'](delta1)

        # concatenate mfcc+delta1+delta2
        mfcc_with_deltas = torch.cat([MFCCs, delta1, delta2], dim=-2)

        # applying the context window
        return self.params['context_window'](mfcc_with_deltas)
