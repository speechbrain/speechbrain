"""
single channel speech enhancement for wind noise reduction.

refer to
    "A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement" .

Authors
 * Wang Wei 2021
"""
import torch
import torch.nn as nn

class CNN_Block(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size=[3, 3],
        stride=(1,2),
        padding=(1,0)) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()

        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ELU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class RNN_Block(torch.nn.Module):
    def __init__(self,
        input_size=1792,
        hidden_size=1792,
        num_layers=2,
        rnn_type='LSTM',
        dropout=0.2) -> None:
        super().__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        if self.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(self.input_size,
                                             self.hidden_size, self.num_layers,
                                             batch_first=True, dropout=self.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, nonlinearity=nonlinearity, dropout=self.dropout)

        # self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size=1):
        if self.rnn_type == 'GRU':
            return torch.zeros(self.num_layers * self.directions_count, batch_size, self.hidden_dim).to(self.device)
        elif self.rnn_type == 'LSTM':
            return (
                    torch.zeros(self.num_layers * self.directions_count, batch_size, self.hidden_dim).to(self.device),
                    torch.zeros(self.num_layers * self.directions_count, batch_size, self.hidden_dim).to(self.device))
        else:
            raise Exception('Unknown rnn_type. Valid options: "gru", "lstm"')

    def forward(self, x):
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)

        return x

class DeCNN_Block(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size=[3, 3],
        stride=(1,2),
        padding=(1,0),
        output_padding=0) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()

        self.layers.append(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ELU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class Encoder(torch.nn.Module):
    def __init__(self, in_channels=1, channels=16, layers=5, scale=2) -> None:
        super().__init__()

        self.cnn_b1 = CNN_Block(1, channels)
        self.cnn_b2 = CNN_Block(channels, channels*2)
        self.cnn_b3 = CNN_Block(channels*2, channels*4)
        # self.cnn_b4 = CNN_Block(channels*4, channels*8)
        # self.cnn_b5 = CNN_Block(channels*8, channels*16)

    def forward(self, x):
        o1 = self.cnn_b1(x)
        o2 = self.cnn_b2(o1)
        o3 = self.cnn_b3(o2)
        # o4 = self.cnn_b4(o3)
        # o5 = self.cnn_b5(o4)

        # return o1, o2, o3, o4, o5
        return o1, o2, o3


class Decoder(torch.nn.Module):
    def __init__(self, in_channels=512, layers=5, scale=2) -> None:
        super().__init__()

        self.decnn_b5 = DeCNN_Block(512, 128)
        self.decnn_b4 = DeCNN_Block(256, 64)
        self.decnn_b3 = DeCNN_Block(128, 32)
        self.decnn_b2 = DeCNN_Block(64, 16, output_padding=(0,1))
        self.decnn_b1 = DeCNN_Block(32, 1)

    def forward(self, x, decoder_o5, decoder_o4, decoder_o3, decoder_o2, decoder_o1):
        o5 = self.decnn_b5(torch.cat((x, decoder_o5), 1))
        o4 = self.decnn_b4(torch.cat((o5, decoder_o4), 1))
        o3 = self.decnn_b3(torch.cat((o4, decoder_o3), 1))
        o2 = self.decnn_b2(torch.cat((o3, decoder_o2), 1))
        o = self.decnn_b1(torch.cat((o2, decoder_o1), 1))

        return o



class crn_att(torch.nn.Module):
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

    def __init__(self, input_size=161, output_size=3, rnn_type='LSTM', contex=0, bidir=False, rnn_size=128, projection=64, layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        if input_size == 257:
            rnn_size = 1792
        elif input_size == 161:
            rnn_size = 1024
        elif input_size == 40:
            rnn_size = 256

        self.encoder = Encoder()
        self.rnn = RNN_Block(input_size=rnn_size, hidden_size=rnn_size, rnn_type=rnn_type)
        self.decoder = Decoder()

        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(in_features=128, out_features=1, bias=False)

        self.fc2 = nn.Linear(in_features=256, out_features=output_size, bias=False)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor):
        """model forward

        Args:
            x (tensor): input tenosr, [N,T,F]

        Returns:
            [type]: [description]
        """
        # N, T, F = x.size()
        if x.dim() == 3:
            x = x.unsqueeze(1)       # [N,T,F] to [N, 1, T, F]

        N, C, T, F = x.size()

        # o1, o2, o3, o4, o5 = self.encoder(x)
        o1, o2, o3 = self.encoder(x)

        embeded_ch = o3.size(1)

        rnn_in = o3.transpose(1, 2)
        rnn_in = rnn_in.reshape(N, T, -1)
        # print(rnn_in.shape)
        rnn_in = self.dropout(rnn_in)
        rnn_out = self.rnn(rnn_in)
        rnn_out = rnn_out.unsqueeze(1)

        output = self.fc1(rnn_out)
        # print('output:{}'.format(output.shape))

        e = self.v(self.tanh(output))
        # print('e:{}'.format(e.shape))
        alpha_t = torch.nn.functional.softmax(e, dim=2)       # attention
        # print('alpha_t:{}'.format(alpha_t.shape))

        output = torch.sum(rnn_out * alpha_t, dim=2)
        # print('output:{}'.format(output.shape))

        # output = self.dropout(output)

        output = self.fc2(output)

        # decoder_out = self.decoder(decoder_in, o5, o4, o3, o2, o1)

        # return decoder_out.squeeze(1)
        output = torch.nn.functional.log_softmax(output, dim=-1)
        return output # [N, 1, 3]

if __name__ == "__main__":
    N, C, T, F = 10, 1, 151, 40
    data = torch.rand((N, T,F))
    print(data.shape)
    model = crn_att(input_size=F, rnn_type='GRU')
    output = model(data)
    print(output.shape)
    # input_size = 257
    # contex = 3
    # model = CustomModel(input_size, contex=contex)
    # # input_data = torch.rand(100, 20, input_size)
    from torchsummary import summary
    summary(model, (C, T, F))
