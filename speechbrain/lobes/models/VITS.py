import torch.nn as nn

class WN():
    def __init__(
        self,
        in_features,
        hidden_features,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f'{kernel_size}'
        assert hidden_channels % 2 == 0, f'{hidden_channels}'

        layers = torch.nn.ModuleList()
        res_skip_layers = torch.nn.ModuleList()
        
        for idx in range(num_layers):
            dilation = dilation_rate ** idx
            padding = int((kernel_size * dilation - dilation) / 2)
            layer = torch.nn.Conv1d(
                hidden_channels, 
                2*hidden_channels, 
                kernel_size=kernel_size,
                dilation=dilation, 
                padding=padding
            )
            layer = torch.nn.utils.weight_norm(layer, name='weight')
            layers.append(in_layer)
            
            res_skip_features = 2 * hidden_features
            if idx == num_layers - 1:
                res_skip_features = hidden_features
            
            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_features, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            res_skip_layers.append(res_skip_layer)
        
        self.layers = layers
        self.res_skip_layers = res_skip_layers
        self.dropout = nn.Dropout(dropout)
        self.hidden_features = hidden_features
        self.num_layers = num_layers
    
    def forward(self, x, x_mask):
        output = torch.zeros_like(x)
        num_features_tensor = torch.IntTensor([self.hidden_features])
        for idx in range(self.num_layers):
            x_in = self.layers[idx](x)
            x_in = self.dropout(x_in)
            acts = self.fused_add_tanh_sigmoid_multiply(x_in, num_features_tensor)
            res_skip_acts = self.res_skip_layers[idx](acts)
            if idx != self.n_layers - 1:
                x = torch.sum(x, res_skip_acts[:,:self.hidden_features,:]) * x_mask
                output = torch.sum(output, res_skip_acts[:,self.hidden_features:,:])
            else:
                output = torch.sum(output, res_skip_acts)
        output = output * x_mask
        return output


    @torch.jit.script
    def fused_add_tanh_sigmoid_multiply(x, n_features):
        n_features = n_features[0]
        t_act = torch.tanh(x[:, :n_features, :])
        s_act = torch.sigmoid(x[:, n_features:, :])
        acts = t_act * s_act
        return acts

    def remove_weight_norm(self):
        for layer in self.layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)
            
        
class PosteriorEncoder(nn.Module):
    def __init__(self, 
                in_features, 
                hidden_features, 
                out_features, 
                kernel_size, 
                dilation_rate,
                num_layers,
                dropout,
                ):
        super(PosteriorEncoder, self).__init__()
        
        pre_encoder = nn.Conv1d(in_features, hidden_features)
        pre_encoder = torch.nn.utils.weight_norm(pre_encoder)
        
        post_encoder = nn.Conv1d(hidden_features, in_features)
        post_encoder.weight.data.zero_()
        post_encoder.bias.data.zero_()
        
        self.post_encoder = post_encoder
        self.pre_encoder = pre_encoder
        self.wavenet = WN(in_features, hidden_features, kernel_size, dilation_rate, num_layers, dropout)
        self.out_features = out_features
        
    def forward(self, x):
        x_mask = None
        x = self.pre_encoder(x) * x_mask
        x = self.wavenet(x, x_mask)
        x = self.post_encoder(x) * x_mask
        mu, log_s = torch.split(x, self.out_features, dim=1)
        z = (mu + torch.randn_like(mu) * torch.exp(log_s)) * x_mask
        return z, mu, log_s, x_mask
    
class PriorEncoder(nn.Module):
    def __init__(self):
        super(PriorEncoder, self).__init__()
        pass

    def forward(self):
       return

class StochasticDurationPredictor(nn.Module):
    def __init__(self):
        super(StochasticDurationPredictor, self).__init__()
        pass

    def forward(self):
       return

   
class VITS(nn.Module):
    def __init__(self):
        super(VITS, self).__init__()
        self.prior_encoder = PriorEncoder()
        self.posterior_encoder = PosteriorEncoder()
        

    def forward(self, inputs):
       return