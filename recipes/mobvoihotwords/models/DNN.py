

import torch
import torch.nn as nn
from speechbrain.lobes.models import CRDNN


class DNN(torch.nn.Sequential):
    """Basic RNN model with projection layers between RNN layers.

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
                 rnn_type='LSTM',
                 contex=0,
                 bidir=False,
                 rnn_size=128,
                 projection=64,
                 layers=2,
                 rnn_neurons=64,
                 cnn_channels=[32, 64],
                 dnn_neurons=128):
        super().__init__()
        # self.layers = torch.nn.ModuleList()

        # self.layers.append(CRDNN())

        # self.encoder = CRDNN(input_size=input_size,
        #                      cnn_channels=cnn_channels,
        #                      rnn_neurons=rnn_neurons,
        #                      dnn_neurons=dnn_neurons)
        # self.encoder = CRDNN()

        self.fc1 = nn.Linear(in_features=dnn_neurons, out_features=3)
        self.fc2 = nn.Linear(in_features=151 * 3, out_features=output_size)

        # self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor):
        """model forward

        Args:
            x (tensor): input tenosr, [N,T,F]

        Returns:
            [type]: [description]
        """
        pass
        N, T, F = x.size()
        if x.dim() == 4:
            x = x.squeeze()       # [N,1,T,F] to [N, T, F]

        # N, T, F = x.size()

        # output = self.encoder(x)

        output = self.fc1(x)

        output = output.reshape(N, 1, -1)

        # # output = self.dropout(output)

        output = self.fc2(output)

        # # return decoder_out.squeeze(1)
        output = torch.nn.functional.log_softmax(output, dim=-1)
        return output  # [N, 1, 3]


if __name__ == "__main__":
    N, C, T, F = 10, 1, 151, 128
    data = torch.rand((N, T,F))
    print(data.shape)
    model = DNN(input_size=F)
    output = model(data)
    print(output.shape)
    # # input_size = 257
    # # contex = 3
    # # model = CustomModel(input_size, contex=contex)
    # # # input_data = torch.rand(100, 20, input_size)
    # from torchsummary import summary
    # summary(model, (C, T, F))
