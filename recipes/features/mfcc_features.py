import torch
from speechbrain.core import load_params


class mfcc_features(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        g = kwargs['global_config']
        self.sb, self.params = load_params('features.yaml', g)

    def forward(self, wav):

        # mfcc computation pipeline
        STFT = self.sb.compute_STFT(wav)
        spectr = self.sb.compute_spectrogram(STFT)
        FBANKs = self.sb.compute_fbanks(spectr)
        MFCCs = self.sb.compute_mfccs(FBANKs)

        # computing derivatives
        delta1 = self.sb.compute_deltas(MFCCs)
        delta2 = self.sb.compute_deltas(delta1)

        # concatenate mfcc+delta1+delta2
        mfcc_with_deltas = torch.cat([MFCCs, delta1, delta2], dim=-2)

        # applying the context window
        return self.sb.context_window(mfcc_with_deltas)
