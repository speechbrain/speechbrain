"""
This file contains a very simple PyTorch module to use for enhancement.

To replace this model, change the `!new:` tag in the hyperparameter file
to refer to a built-in SpeechBrain model or another file containing
a custom PyTorch module.

Authors
 * Peter Plantinga 2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomModel(torch.nn.Module):
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

    def __init__(self, input_size, output_size, contex=0, bidir=False, rnn_size=128, projection=64, layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        contex_size = input_size * (2 * contex + 1)

        # Alternate RNN and projection layers.
        for i in range(layers):
            self.layers.append(
                torch.nn.GRU(
                    batch_first=True,
                    input_size=contex_size if i == 0 else projection,
                    hidden_size=rnn_size,
                    bidirectional=bidir,
                )
            )

            if bidir:
                linear_in_feat = rnn_size * 2
            else:
                linear_in_feat = rnn_size

            # Projection layer reduces size, except last layer, which
            # goes back to input size to create the mask
            linear_size = input_size if i == layers - 1 else projection
            self.layers.append(
                torch.nn.Linear(
                    in_features=linear_in_feat, out_features=linear_size,
                )
            )

        self.fc = nn.Linear(in_features=151*40, out_features=output_size)

        # self.layers.append(torch.nn.Linear(in_features=151*, out_features=output_size))

        # self.fc = torch.nn.Linear(in_features=3624, out_features=output_size)

    def forward(self, x):
        # x = x.transpose(0, 1)
        # x = x.squeeze(1)
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]

        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc(x)

        x = torch.nn.functional.log_softmax(x, dim=-1)
        x = torch.unsqueeze(x, 1)

        return x

if __name__ == "__main__":
    N, C, T, F = 100, 1, 151, 40
    contex = 0
    model = CustomModel(F, contex=contex)
    input_data = torch.rand(N, C, T, F)
    output = model(input_data)
    print(output.shape)
    from torchsummary import summary
    summary(model, (C, T, F))
