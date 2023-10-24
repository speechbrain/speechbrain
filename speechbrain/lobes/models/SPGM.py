import torch
import torch.nn as nn
import torch.nn.functional as F


class SPGMBlock(nn.Module):
    """A block that takes the last element of each chunk for all segments, 
    averages across the segments, 
    then uses the average to perform featurewise linear modulation.
    
    ASSUMES THIS IS USED AS INTER BLOCK IN DualPathModel.
    Expects input BK, S, N
    """

    def __init__(
        self,
        n_embd,
        pool,
        att_h=None,#Only relevant when pool='att'
        att_dropout=0, #Only relevant when pool='att'
        
    ):
        super(SPGMBlock, self).__init__()
        
        self.pool = Pooling(d_input=n_embd,
                            pool = pool,
                            att_h = att_h,
                            att_dropout=att_dropout,
                           )
                                
        self.s_lin = nn.Linear(n_embd,n_embd)
        self.g_lin = nn.Linear(n_embd,n_embd)

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [BK, S, N],
            where, BK = Batchsize and timepoints,
                   S = the number of chunks
                   N = number of filters

        """
        BK, S, N = x.size()

        # within chunk summarization
        glob_info = self.pool(x)
        
        # Average across chunks
        glob_info = glob_info.mean(0)
        s = self.s_lin(glob_info).unsqueeze(0).unsqueeze(0)
        g = self.g_lin(glob_info).unsqueeze(0).unsqueeze(0)

        return (torch.sigmoid(s)*x) + (g*x)
            
#%% Pooling
class Pooling(nn.Module):
    '''
    Pooling: Main Pooling module. It can load either attention-pooling, average
    pooling, maxpooling
    '''      
    def __init__(self,
                 d_input,
                 pool='att',
                 att_h=None,
                 att_dropout=0,
                 ):
        super().__init__()
        
        if pool=='att':
            if att_h is None:
                self.model = PoolAtt(d_input)
            else:
                self.model = PoolAttFF(d_input, h=att_h, dropout=att_dropout)                    
        elif pool=='max':
            self.model = PoolMax()  
        elif pool=='avg':
            self.model = PoolAvg()
        else:
            raise NotImplementedError('Pool option not available')                     

    def forward(self, x):
        '''
        x: [BK, S, N]
        reutrns: [S,N]
        '''
        return self.model(x)

class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, d_input, h, dropout=0.1):
        '''
        d_input: this is the number of channels
        output_size: this is output size of the K dimension. This should be 1 for SPMM modulation block
        h: this is the hidden dimension
        dropout; dropout
        '''
        super().__init__()
        
        self.linear1 = nn.Linear(d_input, h)
        self.linear2 = nn.Linear(h, 1)
        
        # self.linear3 = nn.Linear(d_input, output_size)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        '''
        x: [BK, S, N]
        out: [S,N]
        
        example
        >>>> x = torch.randn(250,10,64)
        >>>> pool = PoolAttFF(64,1, 256)
        >>>> out = pool(x)
        >>>> out.shape
        torch.Size([10, 64])
        '''
        BK, S, N = x.size() #accepts this from the rest of the layer
        x = x.permute(1,0,2) # permutes to make it work with existing code
        #[S, BK, N]
        
        att = self.linear2(self.dropout(self.activation(self.linear1(x)))) #Two linear layers for the hidden dim to compute activation
        #[S, BK, 1]
        att = att.transpose(2,1)
        #[S,1,BK]

        # mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        # att[~mask.unsqueeze(1)] = float("-Inf")        

        att = F.softmax(att, dim=2) #softmax over the activations
        #[S,1,BK]
        
        x = torch.bmm(att, x) #matrix multiplication for masking
        #[S,1,N]

        x = x.squeeze(1)
        #[S,N]
        
        return x
    
class PoolAtt(torch.nn.Module):
    '''
    PoolAtt: Attention-Pooling module.
    '''          
    def __init__(self, d_input):
        super().__init__()
        
        self.linear1 = nn.Linear(d_input, 1)

    def forward(self, x):
        '''
        example
        >>>> x = torch.randn(250,10,64)
        >>>> pool = PoolAtt(64)
        >>>> out = pool(x)
        >>>> out.shape
        torch.Size([10, 64])
        '''
        BK, S, N = x.size() #accepts this from the rest of the layer
        x = x.permute(1,0,2) # permutes to make it work with existing code
        #[S, BK, N]        
        att = self.linear1(x)
        #[S, BK, 1]
        att = att.transpose(2,1)
        #[S, 1, BK]
        # mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        # att[~mask.unsqueeze(1)] = float("-Inf")          
        att = F.softmax(att, dim=2)
        #[S, 1, BK]
        x = torch.bmm(att, x) 
        #[S, 1, N]        
        x = x.squeeze(1)
        #[S, N]
            
        return x

class PoolAvg(torch.nn.Module):
    '''
    PoolAvg: Average pooling that consideres masked time-steps.
    '''          
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        '''
        example
        >>>> x = torch.randn(250,10,64)
        >>>> pool = PoolAvg()
        >>>> out = pool(x)
        >>>> out.shape
        torch.Size([10, 64])
        '''
        BK, S, N = x.size() #accepts this from the rest of the layer
        x = x.permute(1,0,2) # permutes to make it work with existing code
        
        # mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        # mask = ~mask.unsqueeze(2).to(x.device)
        # x.masked_fill_(mask, 0)

        x = torch.div(x.sum(1), x.shape[1])   
        # [S, N]
                
        return x

class PoolMax(torch.nn.Module):
    '''
    PoolMax: Max-pooling that consideres masked time-steps.
    '''        
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        '''
        example
        >>>> x = torch.randn(250,10,64)
        >>>> pool = PoolMax()
        >>>> out = pool(x)
        >>>> out.shape
        torch.Size([10, 64])
        '''
        BK, S, N = x.size() #accepts this from the rest of the layer
        x = x.permute(1,0,2) # permutes to make it work with existing code
                
        # mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        # mask = ~mask.unsqueeze(2).to(x.device)
        # x.masked_fill_(mask, float("-Inf"))

        x = x.max(1)[0]
            
        return x  

if __name__ == '__main__':
    from speechbrain.lobes.models.dual_path import Dual_Computation_Block, SBTransformerBlock

    intra_block = SBTransformerBlock(1, 64, 8)
    inter_block = SPGMBlock(64, 'att',512,0.2)

    dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
    x = torch.randn(10, 64, 100, 10)
    x = dual_comp_block(x)
    print(x.shape)
    # torch.Size([10, 64, 100, 10]) 
