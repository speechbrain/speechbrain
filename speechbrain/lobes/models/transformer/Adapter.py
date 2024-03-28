import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.CNN import Conv2d, Conv1d

# Adapted from https://github.com/Adapter-Hub/adapter-transformers/blob/169ca63856ce00f5c6d9a2da5f00ee228790a41f/src/transformers/adapters/modeling.py#L50

class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """
    def __init__(self, hidden_act):
        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":
            def swish(x):
                return x * torch.sigmoid(x)
            self.f = swish
        elif hidden_act.lower() == "gelu":
            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            self.f = gelu_new
        elif hidden_act.lower() == "gelu_orig":
            self.f = nn.functional.gelu
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu

        super().__init__()

    def forward(self, x):
        return self.f(x)



class Houlsby_Adapter(nn.Module):
    """
    Implementation of Houlsby 2019 Adapter (single adapter block)
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            residual=True,
            non_linearity="relu",
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            ):
        super().__init__()
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual = residual

        # List with all modules
        seq_list = []

        # If layer norm on the input, we add it here first in the seq list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # If down_sampling_size not provided, we take half of the original input size
        self.down_sample = down_sampling_size
        if down_sampling_size is None:
            self.down_sample = self.input_size // 2
        #  ensure that the down sampling size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # Down-projection layer
        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # Non linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        seq_list.append(self.non_linearity)
        self.adapter_down = nn.Sequential(*seq_list)

        # Up-projection layer
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

    def forward(self,x):
        down =  self.adapter_down(x)

        output = self.adapter_up(down)

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)
        if self.residual:
            output = x + output
        return output

    @staticmethod
    def init_bert_weights(module):
        """
        Initialise the weights
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class Emb_Adapter(Houlsby_Adapter):
    """
    Implementation of embedding adapter (adapter + norm at the input level)
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            non_linearity="relu",
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            ):
       super().__init__(input_size, down_sampling_size, non_linearity, add_layer_norm_before, add_layer_norm_after)
       self.norm = sb.nnet.normalization.LayerNorm(input_size, eps=1e-6)

       def forward(self,x):
           x = self.norm(x)
           return super().forward(x)
#---------------------TEST------------------------

class ML_Adapter(nn.Module):
    """
    Implementation of Multi-layer Adapter (single adapter block)
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            non_linearity="relu",
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            ):
        super().__init__()
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after

        # List with all modules
        seq_list = []

        # If layer norm on the input, we add it here first in the seq list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # If down_sampling_size not provided, we take half of the original input size
        self.down_sample = down_sampling_size
        if down_sampling_size is None:
            self.down_sample = self.input_size // 2

        #  ensure that the down sampling size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # Down-projection layer
        seq_list.append(nn.Linear(self.input_size, 1536))

        # Non linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        seq_list.append(self.non_linearity)

        # Add a second layer here
        seq_list.append(nn.Linear(1536, self.down_sample))

        seq_list.append(self.non_linearity)
        self.adapter_down = nn.Sequential(*seq_list)

        # Up-projection layer
        seq_list_up = []
        seq_list_up.append(nn.Linear(self.down_sample, 1536))
        seq_list_up.append(self.non_linearity)
        seq_list_up.append(nn.Linear(1536, self.input_size))
        self.adapter_up = nn.Sequential(*seq_list_up)

        # If we want to have a layer norm on output
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

    def forward(self,x):
        down =  self.adapter_down(x)

        output = self.adapter_up(down)

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        return output


class Variational_Adapter(nn.Module):
    """
    Implementation of variational Adapter (single adapter block)
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            non_linearity="relu",
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            ):
        super().__init__()
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after

        # List with all modules
        #seq_list = []

        # If layer norm on the input, we add it here first in the seq list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            #seq_list.append(self.adapter_norm_before)

        # If down_sampling_size not provided, we take half of the original input size
        self.down_sample = down_sampling_size
        if down_sampling_size is None:
            self.down_sample = self.input_size // 2

        #  ensure that the down sampling size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # Down-projection layer
        self.linear1 =  nn.Linear(self.input_size,self.down_sample)
        self.linear2 =  nn.Linear(self.input_size,self.down_sample)

        # Non linearity
        #self.non_linearity = Activation_Function_Class(non_linearity.lower())
        #seq_list.append(self.non_linearity)

        # Add a second layer here
        #seq_list.append(nn.Linear(1536, self.down_sample))

        #seq_list.append(self.non_linearity)
        #self.adapter_down = nn.Sequential(*seq_list)

        #self.N = torch.distributions.Normal(0, 1)
        #self.N.loc = self.N.loc.cuda()
        #self.N.scale = self.N.scale.cuda()
        self.kl = 0


        # Up-projection layer
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)
        # If we want to have a layer norm on output
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

    def forward(self,x,train):
        #dev = 'cuda:0' # by default

        #for i in self.linear1.parameters():
        #    dev = i.device
        #    break
        #self.N.loc = self.N.loc.to(dev)
        #self.N.scale = self.N.scale.to(dev)
        if self.add_layer_norm_before:
            x = self.adapter_norm_before(x)

        mu = self.linear1(x)
        sigma = torch.exp(self.linear2(x)* 0.5)
        if train:
            eps = sigma.data.new(sigma.size()).normal_()
            z = eps.mul(sigma).add_(mu)#mu + sigma*self.N.sample(mu.shape)
        else:
            z = mu
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()

        output = self.adapter_up(z)

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        return output



class xvector_Adapter(nn.Module):
    """
    Implementation of X-vector adapters
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            non_linearity="relu",
            xvector_size=192,
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            ):
        super().__init__()
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.down_sample = down_sampling_size
        self.xvect_size = xvector_size
        # List with all modules
        seq_list = []

        # Handle x-vector with different size
        if down_sampling_size != self.xvect_size:
            self.xvector_proj = nn.Linear(self.xvect_size, self.down_sample)

        # If layer norm on the input, we add it here first in the seq list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # If down_sampling_size not provided, we take half of the original input size
        if down_sampling_size is None:
            self.down_sample = self.input_size // 2
        #  ensure that the down sampling size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # Down-projection layer
        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # Non linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        seq_list.append(self.non_linearity)
        self.adapter_down = nn.Sequential(*seq_list)

        # Up-projection layer
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

    def forward(self, x, emb):
        #import pdb; pdb.set_trace()

        down_ =  self.adapter_down(x)

        #import pdb; pdb.set_trace()
        # Handle x-vector
        if self.down_sample != self.xvect_size:
            emb = self.non_linearity(self.xvector_proj(emb))
        #import pdb; pdb.set_trace()
        down_ = down_ + emb.unsqueeze(1)
        output = self.adapter_up(down_)

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        return output


class xvector_Adapter_scaling_shifting(nn.Module):
    """
    Implementation of X-vector adapters
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            non_linearity="relu",
            xvector_size=192,
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            ):
        super().__init__()
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.down_sample = down_sampling_size
        self.xvect_size = xvector_size
        # List with all modules
        seq_list_scale = []
        seq_list_shift = []

        # Handle x-vector with different size
        if down_sampling_size != self.xvect_size:
            self.xvector_proj = nn.Linear(self.xvect_size, self.down_sample)

        # If layer norm on the input, we add it here first in the seq list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            self.adapter_norm_before2 = nn.LayerNorm(self.input_size)
            seq_list_scale.append(self.adapter_norm_before)
            seq_list_shift.append(self.adapter_norm_before2)

        # If down_sampling_size not provided, we take half of the original input size
        if down_sampling_size is None:
            self.down_sample = self.input_size // 2
        #  ensure that the down sampling size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # Down-projection layer
        seq_list_scale.append(nn.Linear(self.input_size, self.down_sample))
        seq_list_shift.append(nn.Linear(self.input_size, self.down_sample))

        # Non linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        seq_list_scale.append(self.non_linearity)
        seq_list_shift.append(self.non_linearity)
        self.adapter_down_scale = nn.Sequential(*seq_list_scale)
        self.adapter_down_shift = nn.Sequential(*seq_list_shift)

        # Up-projection layer
        self.adapter_up_scale = nn.Linear(self.down_sample, self.input_size)
        self.adapter_up_shift = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

    def forward(self, x, emb):
        #import pdb; pdb.set_trace()

        down_scale =  self.adapter_down_scale(x)
        down_shift =  self.adapter_down_shift(x)

        #import pdb; pdb.set_trace()
        # Handle x-vector
        if self.down_sample != self.xvect_size:
            emb = self.non_linearity(self.xvector_proj(emb))
        #import pdb; pdb.set_trace()
        down_scale = down_scale + emb.unsqueeze(1)
        down_shift = down_shift + emb.unsqueeze(1)
        up_scale = self.adapter_up_scale(down_scale)
        up_shift = self.adapter_up_shift(down_shift)

        # Sum and element wise product
        output = x * up_scale # Residual of x here
        output = output + up_shift

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        return output

class xvector_Adapter_scaling_shifting_stacked(nn.Module):
    """
    Implementation of X-vector adapters
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            non_linearity="relu",
            xvector_size=192,
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            ):
        super().__init__()
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.down_sample = down_sampling_size
        self.xvect_size = xvector_size
        # List with all modules
        seq_list = []
        seq_list_scale = []
        seq_list_shift = []

        # Handle x-vector with different size
        if down_sampling_size != self.xvect_size:
            self.xvector_proj = nn.Linear(self.xvect_size, self.down_sample)

        # If layer norm on the input, we add it here first in the seq list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            self.adapter_norm_before2 = nn.LayerNorm(self.input_size)
            self.adapter_norm_before3 = nn.LayerNorm(self.input_size)
            seq_list_scale.append(self.adapter_norm_before)
            seq_list_shift.append(self.adapter_norm_before2)
            seq_list.append(self.adapter_norm_before3)

        # If down_sampling_size not provided, we take half of the original input size
        if down_sampling_size is None:
            self.down_sample = self.input_size // 2
        #  ensure that the down sampling size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # Down-projection layer
        seq_list.append(nn.Linear(self.input_size, self.down_sample))
        seq_list_scale.append(nn.Linear(self.input_size, self.down_sample))
        seq_list_shift.append(nn.Linear(self.input_size, self.down_sample))

        # Non linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        seq_list_scale.append(self.non_linearity)
        seq_list_shift.append(self.non_linearity)
        seq_list.append(self.non_linearity)
        self.adapter_down = nn.Sequential(*seq_list)
        self.adapter_down_scale = nn.Sequential(*seq_list_scale)
        self.adapter_down_shift = nn.Sequential(*seq_list_shift)

        # Up-projection layer
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)
        self.adapter_up_scale = nn.Linear(self.down_sample, self.input_size)
        self.adapter_up_shift = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

    def forward(self, x, emb):
        #import pdb; pdb.set_trace()
        # First adapter
        down_ = self.adapter_down(x)
        up = self.adapter_up(down_)

        # Scaling + shifting adapter
        down_scale =  self.adapter_down_scale(up)
        down_shift =  self.adapter_down_shift(up)

        #import pdb; pdb.set_trace()
        # Handle x-vector
        if self.down_sample != self.xvect_size:
            emb = self.non_linearity(self.xvector_proj(emb))
        #import pdb; pdb.set_trace()
        down_scale = down_scale + emb.unsqueeze(1)
        down_shift = down_shift + emb.unsqueeze(1)
        up_scale = self.adapter_up_scale(down_scale)
        up_shift = self.adapter_up_shift(down_shift)

        # Sum and element wise product
        output = x * up_scale
        output = output + up_shift

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        return output
class xvector_Adapter_scaling_shifting_stacked_bigger(nn.Module):
    """
    Implementation of X-vector adapters
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            non_linearity="relu",
            xvector_size=192,
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            ):
        super().__init__()
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.down_sample = down_sampling_size
        self.xvect_size = xvector_size
        # List with all modules
        seq_list = []
        seq_list_scale = []
        seq_list_shift = []
        xvector_proj = []
        self.non_linearity = Activation_Function_Class(non_linearity.lower())

        # Handle x-vector with different size
        if down_sampling_size != self.xvect_size:
            xvector_proj.append(nn.Linear(self.xvect_size, self.down_sample))
            xvector_proj.append(self.non_linearity)
            xvector_proj.append(nn.Linear(self.down_sample, self.down_sample))
            xvector_proj.append(self.non_linearity)
        self.xvec_proj = nn.Sequential(*xvector_proj)

        # If layer norm on the input, we add it here first in the seq list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            self.adapter_norm_before2 = nn.LayerNorm(self.input_size)
            self.adapter_norm_before3 = nn.LayerNorm(self.input_size)
            seq_list_scale.append(self.adapter_norm_before)
            seq_list_shift.append(self.adapter_norm_before2)
            seq_list.append(self.adapter_norm_before3)

        # If down_sampling_size not provided, we take half of the original input size
        if down_sampling_size is None:
            self.down_sample = self.input_size // 2
        #  ensure that the down sampling size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # Down-projection + middle transformation layer
        seq_list.append(nn.Linear(self.input_size, self.down_sample))
        seq_list_scale.append(nn.Linear(self.input_size, self.down_sample))
        seq_list_shift.append(nn.Linear(self.input_size, self.down_sample))

        # Non linearity
        seq_list_scale.append(self.non_linearity)
        seq_list_shift.append(self.non_linearity)
        seq_list.append(self.non_linearity)

        seq_list.append(nn.Linear(self.down_sample, self.down_sample))
        seq_list_scale.append(nn.Linear(self.down_sample, self.down_sample))
        seq_list_shift.append(nn.Linear(self.down_sample, self.down_sample))

        seq_list_scale.append(self.non_linearity)
        seq_list_shift.append(self.non_linearity)
        seq_list.append(self.non_linearity)

        self.adapter_down = nn.Sequential(*seq_list)
        self.adapter_down_scale = nn.Sequential(*seq_list_scale)
        self.adapter_down_shift = nn.Sequential(*seq_list_shift)


        # Up-projection layer
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)
        self.adapter_up_scale = nn.Linear(self.down_sample, self.input_size)
        self.adapter_up_shift = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

    def forward(self, x, emb):
        #import pdb; pdb.set_trace()
        # First adapter
        down_ = self.adapter_down(x)
        up = self.adapter_up(down_)

        # Scaling + shifting adapter
        down_scale =  self.adapter_down_scale(up)
        down_shift =  self.adapter_down_shift(up)

        #import pdb; pdb.set_trace()
        # Handle x-vector
        if self.down_sample != self.xvect_size:
            #emb = self.non_linearity(self.xvector_proj(emb))
            emb = self.xvec_proj(emb)
        #import pdb; pdb.set_trace()
        down_scale = down_scale + emb.unsqueeze(1)
        down_shift = down_shift + emb.unsqueeze(1)
        up_scale = self.adapter_up_scale(down_scale)
        up_shift = self.adapter_up_shift(down_shift)

        # Sum and element wise product
        output = x * up_scale
        output = output + up_shift

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        return output
class xvector_Adapter_newShiftScale(nn.Module):
    """
    Implementation of X-vector adapters
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            non_linearity="relu",
            xvector_size=192,
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            ):
        super().__init__()
        self.input_size = input_size
        #self.add_layer_norm_before = add_layer_norm_before
        #self.add_layer_norm_after = add_layer_norm_after
        #self.down_sample = down_sampling_size
        self.xvect_size = xvector_size
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        # List with all modules
        #seq_list = []

        # Handle x-vector with different size
        if self.input_size != self.xvect_size:
            self.gamma = nn.Linear(self.xvect_size, self.input_size)
            self.beta = nn.Linear(self.xvect_size, self.input_size)

        # If layer norm on the input, we add it here first in the seq list

    def forward(self, x, emb):
        #import pdb; pdb.set_trace()

        # Handle x-vector
        if self.input_size != self.xvect_size:
            gamma_ = self.non_linearity(self.gamma(emb))
            beta_ = self.non_linearity(self.beta(emb))
        output =   gamma_.unsqueeze(1) * x + beta_.unsqueeze(1)
        return output
class xvector_Adapter_newversion(nn.Module):
    """
    Implementation of X-vector adapters
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            non_linearity="relu",
            xvector_size=192,
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            ):
        super().__init__()
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.down_sample = down_sampling_size
        self.xvect_size = xvector_size
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        # List with all modules
        seq_list = []

        # Handle x-vector with different size
        if self.input_size != self.xvect_size:
            self.gamma = nn.Linear(self.xvect_size, self.input_size)
            self.beta = nn.Linear(self.xvect_size, self.input_size)

        # If layer norm on the input, we add it here first in the seq list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # If down_sampling_size not provided, we take half of the original input size
        self.down_sample = down_sampling_size
        if down_sampling_size is None:
            self.down_sample = self.input_size // 2
        #  ensure that the down sampling size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # Down-projection layer
        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # Non linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        seq_list.append(self.non_linearity)
        self.adapter_down = nn.Sequential(*seq_list)

        # Up-projection layer
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)


    def forward(self, x, emb):
        #import pdb; pdb.set_trace()

        if self.input_size != self.xvect_size:
            gamma_ = self.non_linearity(self.gamma(emb))
            beta_ = self.non_linearity(self.beta(emb))
        # Handle x-vector
        x =   gamma_.unsqueeze(1) * x + beta_.unsqueeze(1)

        down =  self.adapter_down(x)
        output = self.adapter_up(down)

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        return output

class Adapter_scaling_shifting(nn.Module):
    """
    Implementation o scaling and shifting adapters
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            non_linearity="relu",
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            ):
        super().__init__()
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.down_sample = down_sampling_size
        # List with all modules
        seq_list_scale = []
        seq_list_shift = []

        # If layer norm on the input, we add it here first in the seq list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            self.adapter_norm_before2 = nn.LayerNorm(self.input_size)
            seq_list_scale.append(self.adapter_norm_before)
            seq_list_shift.append(self.adapter_norm_before2)

        # If down_sampling_size not provided, we take half of the original input size
        if down_sampling_size is None:
            self.down_sample = self.input_size // 2
        #  ensure that the down sampling size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # Down-projection layer
        seq_list_scale.append(nn.Linear(self.input_size, self.down_sample))
        seq_list_shift.append(nn.Linear(self.input_size, self.down_sample))

        # Non linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        seq_list_scale.append(self.non_linearity)
        seq_list_shift.append(self.non_linearity)
        self.adapter_down_scale = nn.Sequential(*seq_list_scale)
        self.adapter_down_shift = nn.Sequential(*seq_list_shift)

        # Up-projection layer
        self.adapter_up_scale = nn.Linear(self.down_sample, self.input_size)
        self.adapter_up_shift = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

    def forward(self, x):

        down_scale =  self.adapter_down_scale(x)
        down_shift =  self.adapter_down_shift(x)

        up_scale = self.adapter_up_scale(down_scale)
        up_shift = self.adapter_up_shift(down_shift)

        # Sum and element wise product
        output = x * up_scale # Residual of x here
        output = output + up_shift

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        return output
class Houlsby_Adapter_xvector(nn.Module):
    """
    Implementation of Houlsby 2019 Adapter (single adapter block)
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            residual=True,
            non_linearity="relu",
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            xvector_size=192,
            ):
        super().__init__()
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual = residual
        self.xvect_size = xvector_size
        # List with all modules
        seq_list = []

        # If layer norm on the input, we add it here first in the seq list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # If down_sampling_size not provided, we take half of the original input size
        self.down_sample = down_sampling_size
        if down_sampling_size is None:
            self.down_sample = self.input_size // 2
        #  ensure that the down sampling size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # Handle x-vector with different size
        if down_sampling_size != self.xvect_size:
            self.f = nn.Linear(xvector_size, self.down_sample)
            self.g = nn.Linear(xvector_size, self.down_sample)

        # Down-projection layer
        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # Non linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        seq_list.append(self.non_linearity)
        self.adapter_down = nn.Sequential(*seq_list)

        # Up-projection layer
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)
        self.m = nn.Tanh()

    def forward(self,x, emb):
        down =  self.adapter_down(x)
        # Handle x-vector
        if self.down_sample != self.xvect_size:
            scale = self.m(self.f(emb))
            shift =self.m(self.g(emb))
        down = down * scale.unsqueeze(1) + shift.unsqueeze(1)

        output = self.adapter_up(down)

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)
        if self.residual:
            output = x + output
        return output

class AdaptersFusion(nn.Module):
    """
    Implementation of Adapterfusion block
    """

    def __init__(
            self,
            dense_size,
            attention_probs_dropout_probs=0.0,
            ):
        super(AdaptersFusion, self).__init__()

        self.dense_size = dense_size
        self.dropout = nn.Dropout(attention_probs_dropout_probs)

        self.query = nn.Linear(self.dense_size, self.dense_size)
        self.query.apply(Houlsby_Adapter.init_bert_weights)

        self.key = nn.Linear(self.dense_size, self.dense_size)
        self.key.apply(Houlsby_Adapter.init_bert_weights)

        self.value = nn.Linear(dense_size,dense_size,bias=False)
        self.value.apply(Houlsby_Adapter.init_bert_weights)
        self.value.weight.data = (torch.zeros(self.dense_size, self.dense_size) + 0.000001).fill_diagonal_(1.0)

        self.T = 1.0
        self.reduction = self.T / 1000.0

    def forward(self, query, key, value, residual):

        query_layer = self.query(query)
        key_layer = self.key(key)

        # key/value have dim => batch, length, number-of-adapters, features
        value_layer = self.value(value)

        #Take the dot product between query and key to get the raw attention scores
        attention_scores = torch.squeeze(torch.matmul(query_layer.unsqueeze(2),key_layer.transpose(-2,-1)), dim=2)
        attention_scores = self.dropout(attention_scores)

        # Normalize the attention score to probabilities with a softmax
        attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T)
        self.T = max(self.T - self.reduction, 1.0)

        context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_layer), dim=2)
        context_layer += residual

        return context_layer


class Conv_Adapter(nn.Module):
    """
    Implementation ofthe Conv Adapter (single adapter block)
    """
    def __init__(self,
            input_size,
            down_sampling_size=None,
            residual=True,
            non_linearity="relu",
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            type_conv="down",
            ):
        super().__init__()
        self.type_conv = type_conv
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual = residual

        # List with all modules
        seq_list = []

        # If layer norm on the input, we add it here first in the seq list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # If down_sampling_size not provided, we take half of the original input size
        self.down_sample = down_sampling_size
        if down_sampling_size is None:
            self.down_sample = self.input_size // 2
        #  ensure that the down sampling size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # Down-projection layer
        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # Non linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        seq_list.append(self.non_linearity)
        self.adapter_down = nn.Sequential(*seq_list)

        # Up-projection layer
        #self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)
        # Conv
        self.conv = Conv1d(
                input_shape=(None, None, self.input_size),
                out_channels=self.input_size,
                kernel_size=3,
                stride=1,
                padding="same",
                groups=self.input_size,
                conv_init="normal",
                skip_transpose=False)

    def forward(self,x):
        if self.type_conv == "down":
            down = self.conv(x)
        else:
            down =  self.adapter_down(x)

        if self.type_conv == "middle":
            down = self.conv(down)

        if self.type_conv == "up":
            output = self.conv(down)
        else:
            output = self.adapter_up(down)

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)
        if self.residual:
            output = x + output
        return output

    @staticmethod
    def init_bert_weights(module):
        """
        Initialise the weights
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

