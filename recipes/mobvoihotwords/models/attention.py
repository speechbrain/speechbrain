

import torch
import torch.nn as nn
from speechbrain.lobes.models import CRDNN


class Attention(torch.nn.Sequential):
    """Basic attetnion layers.

    "Attention-based End-to-End Models for Small-Footprint Keyword Spotting"
    "Attentionbased models for text-dependent speaker verification"

    Arguments
    ---------
    input_size : int
        Size of the expected input in the 3rd dimension.
    rnn_size : int
        Number of neurons to use in rnn (for each direction -> and <-).
    projection : int
        Number of neurons in projection layer.
    layers : int
        Number of RNN layers to use.
    """

    def __init__(self, input_size=161,
                 output_size=3,
                 att_feature=128):
        super().__init__()

        self.fc1 = nn.Linear(in_features=input_size, out_features=att_feature)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(in_features=att_feature, out_features=1, bias=False)

        self.fc2 = nn.Linear(in_features=input_size, out_features=output_size, bias=False)


        # self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor):
        """model forward

        Args:
            x (tensor): input tenosr, [N,T,F]

        Returns:
            [type]: [description]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)       # [N,T,F] to [N,1, T, F]
        N, C, T, F = x.size()

        output = self.fc1(x)
        # print('output:{}'.format(output.shape))

        e = self.v(self.tanh(output))
        # print('e:{}'.format(e.shape))
        alpha_t = torch.nn.functional.softmax(e, dim=2)     # time dim attention
        # print('alpha_t:{}'.format(alpha_t.shape))

        output = torch.sum(x * alpha_t, dim=2)              # time domain summation
        # print('output:{}'.format(output.shape))


        output = self.fc2(output)

        # # return decoder_out.squeeze(1)
        output = torch.nn.functional.log_softmax(output, dim=-1)
        return output  # [N, 1, 3]


if __name__ == "__main__":
    N, C, T, F = 10, 1, 151, 128
    data = torch.rand((N, T, F))
    print(data.shape)
    model = Attention(input_size=F)
    output = model(data)
    print(output.shape)
    from torchsummary import summary
    summary(model, (C, T, F))
