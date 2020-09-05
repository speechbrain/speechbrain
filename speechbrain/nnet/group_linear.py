

import torch.nn.functional as F
import torch
import torch.nn as nn
import math

class GroupLinear(nn.Module):

    def __init__(self, din, dout, num_blocks, bias=True, a = None):
        super(GroupLinear, self).__init__()
        self.nb = num_blocks

        din = din // num_blocks
        dout = dout // num_blocks

        self.dout = dout

        if a is None:
            a = 1. / math.sqrt(dout)

        #gain = 1.0 / math.sqrt(2)
        #a = gain * math.sqrt(6.0 / (din + dout))

        self.weight = nn.Parameter(torch.FloatTensor(num_blocks,din,dout).uniform_(-a,a))

        self.bias = bias

        if bias is True:
            self.bias = nn.Parameter(torch.FloatTensor(num_blocks,dout).uniform_(-a,a))
            #self.bias = nn.Parameter(torch.zeros(dout*num_blocks))
        else:
            self.bias = None

    def forward(self,x):

	#input: ts x bs x blocks*nhid
	#ts*bs , blocks, nhid
	#blocks, ts*bs, nhid
        ts,bs,m = x.shape

        x = x.reshape((ts*bs, self.nb, m//self.nb))
        x = x.permute(1,0,2)
        x = torch.bmm(x,self.weight)
        x = x.permute(1,0,2)

        if not self.bias is None:
            x = x + self.bias

        x = x.reshape((ts, bs, self.dout*self.nb))

        #if not self.bias is None:
        #    x += self.bias

        return x

class GroupMLP(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, din, dout, num_blocks, dropout=0.1):
        super(GroupMLP, self).__init__()

        self.w_1 = nn.Parameter(0.01 * torch.randn(num_blocks,din,dout))
        self.w_2 = nn.Parameter(0.01 * torch.randn(num_blocks,dout,din))

        self.layer_norm = nn.LayerNorm(din)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):

        residual = x*1.0
        x = x.permute(1,0,2)
        x = torch.bmm(F.relu(torch.bmm(x,self.w_1)), self.w_2)
        x = x.permute(1,0,2)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x

if __name__ == "__main__":

    GLN = GroupLinear(512, 1024, 2, bias=True)

    #bs, blocks, nhid
    x = torch.randn(64,12,2*512)

    print(GLN(x).shape)

    #for p in GLN.parameters():
    #    print(p.shape)

