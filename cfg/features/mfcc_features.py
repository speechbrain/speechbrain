import torch
from speechbrain.core import load_params

class mfcc_features(torch.nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        g = kwargs['global_config']
        self.f, self.params = load_params(__file__, 'features.yaml', g)

    def forward(self, wav):

        # mfcc computation pipeline
        STFT = self.f.compute_STFT(wav)
        spectr = self.f.compute_spectrogram([STFT])
        FBANKs = self.f.compute_fbanks([spectr])
        MFCCs = self.f.compute_mfccs([FBANKs])

        # computing derivatives
        delta1 = self.f.compute_deltas([MFCCs])
        delta2 = self.f.compute_deltas([delta1])

        # concatenate mfcc+delta1+delta2
        mfcc_with_deltas = torch.cat([MFCCs, delta1, delta2], dim=-2)

        # applying the context window
        return self.f.context_window([mfcc_with_deltas])
