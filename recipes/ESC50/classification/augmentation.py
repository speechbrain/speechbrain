import torch
import torch.nn as nn
from speechbrain.lobes import augment
import pdb


class TFAugmentation(nn.Module):
    def __init__(
        self,
        time_warp=True,
        time_warp_window=5,
        time_warp_mode="bicubic",
        freq_mask=True,
        freq_mask_width=(0, 20),
        n_freq_mask=2,
        time_mask=True,
        time_mask_width=(0, 100),
        n_time_mask=2,
        replace_with_zero=True,
        time_roll=True,
        time_roll_limit=(0, 200),
        freq_shift=True,
        freq_shift_limit=(-20, 20),
    ):
        super().__init__()
        self.specaugment = augment.SpecAugment(
            time_warp=time_warp,
            time_warp_window=time_warp_window,
            time_warp_mode=time_warp_mode,
            freq_mask=freq_mask,
            freq_mask_width=freq_mask_width,
            n_freq_mask=n_freq_mask,
            time_mask=time_mask,
            time_mask_width=time_mask_width,
            n_time_mask=n_time_mask,
            replace_with_zero=replace_with_zero
        )
        self.time_roll = time_roll
        self.time_roll_limit = time_roll_limit
        self.freq_shift = freq_shift
        self.freq_shift_limit = freq_shift_limit

    def forward(self, x, lens):
        '''
            x: [batch, time, freq]
        '''
        with torch.no_grad():
            if x.dim() == 4 and x.shape[1] == 1:
                x = x.squeeze(1)
            x = self.specaugment(x.clone())
            # circular shift in time
            if self.time_roll:
                roll_offset = torch.randint(self.time_roll_limit[0], self.time_roll_limit[1]+1, (1,)).item()
                x = torch.roll(x, roll_offset, 1)
            # truncated shift in frequency
            if self.freq_shift:
                shift_offset = torch.randint(self.freq_shift_limit[0], self.freq_shift_limit[1]+1, (1,)).item()
                x = torch.roll(x, shift_offset, 2)
                if shift_offset > 0:
                    #  x[:, :, shift_offset:] = x[:, :, :-shift_offset]
                    x[:, :, :shift_offset] = 0
                elif shift_offset < 0:
                    #  x[:, :, :shift_offset] = x[:, :, -shift_offset:]
                    x[:, :, shift_offset:] = 0
        return x


