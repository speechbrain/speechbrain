"""
single channel speech enhancement for wind noise reduction.

refer to
    "Learning Complex Spectral Mapping With Gated Convolutional Recurrent Networks for Monaural Speech Enhancement" .

Authors
 * Wang Wei 2021
"""
import torch
import torch.nn as nn

class ConvGLU_Block(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size=[1, 3],
        stride=(1,2),
        padding=0) -> None:
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.activate = torch.nn.Sigmoid()

    def forward(self,x):
        o1 = self.conv_1(x)
        o2 = self.conv_2(x)
        o2 = self.activate(o2)

        o = o1 * o2

        return o

class DeConvGLU_Block(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size=[3, 3],
        stride=(1,2),
        padding=(1,0),
        output_padding=0) -> None:
        super().__init__()

        self.deconv_1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding)

        self.deconv_2 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding)
        self.activate = torch.nn.Sigmoid()

    def forward(self,x):
        o1 = self.deconv_1(x)
        o2 = self.deconv_2(x)
        o2 = self.activate(o2)

        o = o1 * o2

        return o

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
    def __init__(self, in_channels=2, channels=16, layers=5, scale=2) -> None:
        super().__init__()

        self.cnn_b1 = ConvGLU_Block(in_channels, channels)
        self.cnn_b2 = ConvGLU_Block(channels, channels*2)
        self.cnn_b3 = ConvGLU_Block(channels*2, channels*4)
        self.cnn_b4 = ConvGLU_Block(channels*4, channels*8)
        self.cnn_b5 = ConvGLU_Block(channels*8, channels*16)

    def forward(self, x):
        o1 = self.cnn_b1(x)
        o2 = self.cnn_b2(o1)
        o3 = self.cnn_b3(o2)
        o4 = self.cnn_b4(o3)
        o5 = self.cnn_b5(o4)

        return o1, o2, o3, o4, o5


class Decoder(torch.nn.Module):
    def __init__(self, in_channels=512, layers=5, scale=2) -> None:
        super().__init__()

        self.decnn_b5 = DeConvGLU_Block(512, 128)
        self.decnn_b4 = DeConvGLU_Block(256, 64)
        self.decnn_b3 = DeConvGLU_Block(128, 32)
        self.decnn_b2 = DeConvGLU_Block(64, 16, output_padding=(0,1))
        self.decnn_b1 = DeConvGLU_Block(32, 1)

        self.linear = torch.nn.Linear(161, 161)

    def forward(self, x, decoder_o5, decoder_o4, decoder_o3, decoder_o2, decoder_o1):
        o5 = self.decnn_b5(torch.cat((x, decoder_o5), 1))
        o4 = self.decnn_b4(torch.cat((o5, decoder_o4), 1))
        o3 = self.decnn_b3(torch.cat((o4, decoder_o3), 1))
        o2 = self.decnn_b2(torch.cat((o3, decoder_o2), 1))
        o = self.decnn_b1(torch.cat((o2, decoder_o1), 1))
        o = self.linear(o)

        return o



class ccrn(torch.nn.Module):
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

    def __init__(self, input_size=161, contex=0, bidir=False, rnn_size=128, projection=64, layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        if input_size == 257:
            rnn_size = 1792
        elif input_size == 161:
            rnn_size = 1024

        self.encoder = Encoder()
        self.rnn = RNN_Block(input_size=rnn_size, hidden_size=rnn_size)
        self.decoder_real = Decoder()
        self.decoder_imag = Decoder()

    def forward(self, x: torch.Tensor):
        """model forward

        Args:
            x (tensor): input tenosr, [N,T,F,2]

        Returns:
            [type]: [description]
        """

        x = x.transpose(2,3).transpose(1,2)  # [N,T,F,2] to [N,2,T,F]

        N, C, T, F = x.size()
        # x = x.unsqueeze(1)

        o1, o2, o3, o4, o5 = self.encoder(x)

        embeded_ch = o5.size(1)

        rnn_in = o5.transpose(1, 2)
        rnn_in = rnn_in.reshape(N, T, -1)
        rnn_out = self.rnn(rnn_in)
        rnn_out = rnn_out.unsqueeze(1)

        decoder_in = rnn_out.reshape(N, embeded_ch, T, -1)

        decoder_out_real = self.decoder_real(decoder_in, o5, o4, o3, o2, o1)
        decoder_out_real = decoder_out_real.squeeze().unsqueeze(-1)
        decoder_out_imag = self.decoder_imag(decoder_in, o5, o4, o3, o2, o1)
        decoder_out_imag = decoder_out_imag.squeeze().unsqueeze(-1)

        out = torch.cat((decoder_out_real, decoder_out_imag), -1)

        return out

def test_istft():
    from speechbrain.processing.features import STFT
    from speechbrain.processing.features import ISTFT

    fs = 16000
    inp = torch.randn([10, 16000])
    # inp = torch.stack(3 * [inp], -1)

    compute_stft = STFT(sample_rate=fs)
    compute_istft = ISTFT(sample_rate=fs)
    stft_out = compute_stft(inp)
    print(stft_out.dtype)
    print(stft_out.shape)
    out = compute_istft(stft_out, sig_length=16000)
    print(out.shape)

    assert torch.sum(torch.abs(inp - out) < 1e-6) >= inp.numel() - 5

    assert torch.jit.trace(compute_stft, inp)
    assert torch.jit.trace(compute_istft, compute_stft(inp))

if __name__ == "__main__":
    # test_istft()
    N, C, T, F = 10, 100, 161, 2
    data = torch.rand((N, C, T,F))
    print(data.shape)
    model = ccrn(input_size=161)
    output = model(data)
    print(output.shape)

    # test_istft()

    # x = output.permute(0, 2, 1, 3)

    # # isft ask complex input
    # x = torch.complex(x[..., 0], x[..., 1])

    # istft = torch.istft(
    #     input=x,
    #     n_fft=320,
    #     hop_length=160,
    #     win_length=320,
    #     center=True,
    #     onesided=True
    # )
    # print(istft.shape)

    # input_size = 161
    from torchsummary import summary
    summary(model, (C, T, F))
