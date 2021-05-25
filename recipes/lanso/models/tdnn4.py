import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(torch.nn.Module):
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

class TDNN_Block(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=True,
                    dropout_p=0.0
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN_Block, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)

        return x

class TDNN(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=True,
                    dropout_p=0.0
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

        self.tdnn_dim = 64

        self.tdnn1 = TDNN_Block(input_dim=40, output_dim=self.tdnn_dim, context_size=5, dilation=1)
        self.tdnn2 = TDNN_Block(input_dim=self.tdnn_dim, output_dim=self.tdnn_dim, context_size=5, dilation=2)
        self.tdnn3 = TDNN_Block(input_dim=self.tdnn_dim, output_dim=self.tdnn_dim, context_size=5, dilation=4)
        self.tdnn4 = TDNN_Block(input_dim=self.tdnn_dim, output_dim=self.tdnn_dim, context_size=5, dilation=8)
        self.tdnn5 = TDNN_Block(input_dim=self.tdnn_dim, output_dim=self.tdnn_dim, context_size=3, dilation=1)
        self.tdnn6 = TDNN_Block(input_dim=self.tdnn_dim, output_dim=self.tdnn_dim, context_size=3, dilation=2)
        self.tdnn7 = TDNN_Block(input_dim=self.tdnn_dim, output_dim=self.tdnn_dim, context_size=3, dilation=4)
        self.tdnn8 = TDNN_Block(input_dim=self.tdnn_dim, output_dim=self.tdnn_dim, context_size=3, dilation=8)

        self.attention = Attention(input_size=self.tdnn_dim, output_size=4)

        # self.fc1 = nn.Linear(in_features=self.tdnn_dim, out_features=16)
        # self.fc2 = nn.Linear(in_features=976, out_features=4)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        if x.ndim == 4:
            x = torch.squeeze(x, dim=1)
        # x = torch.squeeze(x, dim=1)
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = self.tdnn6(x)
        x = self.tdnn7(x)
        x = self.tdnn8(x)

        x = self.dropout(x)

        # x = self.fc1(x)

        # x = x.reshape(x.shape[0],1, -1)

        # x = self.fc2(x)

        x = self.attention(x)

        x = torch.nn.functional.log_softmax(x, dim=-1)

        return x

if __name__ == "__main__":

    model = TDNN()
    batch = 64
    input_data = torch.randn((batch, 151, 40))
    output = model(input_data)
    print(output.size()) # (N, T-14, 1500)

    from torchsummary import summary
    summary(model, (1, 151, 40))

