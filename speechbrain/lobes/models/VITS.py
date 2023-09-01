"""VITS implementaion in the SpeechBrain.
Authors
* Sathvik Udupa 2023
"""

import torch.nn as nn
import math
import torch
from speechbrain.lobes.models.transformer import Transformer

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
        
        self.wavenet = WN(
            in_features, 
            hidden_features, 
            kernel_size, 
            dilation_rate, 
            num_layers, 
            dropout
        )
        self.out_features = out_features
        self.post_encoder = post_encoder
        self.pre_encoder = pre_encoder
         
    def forward(self, x):
        x_mask = None
        x = self.pre_encoder(x) * x_mask
        x = self.wavenet(x, x_mask)
        x = self.post_encoder(x) * x_mask
        mu, log_s = torch.split(x, self.out_features, dim=1)
        z = (mu + torch.randn_like(mu) * torch.exp(log_s)) * x_mask
        return z, mu, log_s, x_mask
    
class PriorEncoder(nn.Module):
    def __init__(
        self,
        num_tokens,
        in_features,
        out_features,
        hidden_features,
        ffn_features,
        num_heads,
        num_layers,
        attention_type,
        dropout
        
    ):
        super(PriorEncoder, self).__init__()
        self.encoder = Transformer.TransformerEncoder(
            num_layers=num_layers,
            nhead=num_heads,
            d_ffn=ffn_features,
            d_model=hidden_features,
            dropout=dropout,
            attention_type=attention_type,
        )
        self.token_embedding = nn.Embedding(num_tokens, hidden_features)
        nn.init.normal_(self.token_embedding.weight, 0.0, hidden_features**-0.5)
        self.post_encoder = nn.Conv1d(hidden_features, out_features*2, 1)
        self.hidden_features = hidden_features
        self.out_features = out_features
        
    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.hidden_features)
        x_mask = None
        x = self.encoder(x, x_mask)
        x_d = self.proj(x)
        mu, log_s = torch.split(x_d, self.out_features, dim=1)
        return x, mu, log_s, x_mask

class StochasticDurationPredictor(nn.Module):
    def __init__(self):
        super(StochasticDurationPredictor, self).__init__()
        pass

    def forward(self):
       return

class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        kernel_size,
        dropout,
        
    ):
        super(DurationPredictor, self).__init__()
        self.conv1 = nn.Conv1d(
            in_features, 
            hidden_features, 
            kernel_size, 
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            hidden_features, 
            hidden_features, 
            kernel_size, 
            padding=kernel_size // 2
        )
        self.conv3 = nn.Conv1d(
            hidden_features, 
            1, 
            1, 
        )
        self.norm1 = LayerNorm(hidden_features)
        self.norm2 = LayerNorm(hidden_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, x_mask):
        x = self.drop(self.norm1(self.relu(self.conv1(x))))
        x = self.drop(self.norm2(self.relu(self.conv2(x))))
        x = self.conv3(x)
        return x
    
class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout,
        mean_only,
    ):
        super().__init__()
        self.features = in_features // 2
        self.mean_only = mean_only
        
        self.pre_decoder = nn.Conv1d(in_features // 2, hidden_features, 1)
        self.decoder = WN(
            in_features=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.post_decoder = nn.Conv1d(hidden_channels, (in_features // 2) * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask):
        x0, x1 = torch.split(x, [self.features] * 2, 1)
        h = self.pre_decoder(x0) * x_mask
        h = self.decoder(h, x_mask)
        x0_d = self.post_decoder(h) * x_mask
        if not self.mean_only:
            mu, log_s = torch.split(x0_d, [self.features] * 2, 1)
        else:
            mu = x0_d
            log_s = torch.zeros_like(mu)

        x1 = mu + x1 * torch.exp(log_s) * x_mask
        x = torch.cat([x0, x1], 1)
        logdet = torch.sum(log_s, [1, 2])
        return x, logdet
    
    def reverse(self, x, x_mask):
        x0, x1 = torch.split(x, [self.features] * 2, 1)
        h = self.pre_decoder(x0) * x_mask
        h = self.decoder(h, x_mask)
        x0_d = self.post_decoder(h) * x_mask
        if not self.mean_only:
            mu, log_s = torch.split(x0_d, [self.features] * 2, 1)
        else:
            mu = x0_d
            log_s = torch.zeros_like(mu)
            
        x1 = (x1 - m) * torch.exp(-log_s) * x_mask
        x = torch.cat([x0, x1], 1)
        return x   

        
class ResidualCouplingBlock(nn.Module()):
    def __init__(
        self,
        in_features,
        hidden_features,
        kernel_size,
        dilation_rate,
        num_layers,
        num_flows,
        mean_only,
        dropout,
    ):
        super().__init__()
        flows = nn.ModuleList()
        for i in range(num_flows):
            flows.append(ResidualCouplingLayer(
                in_features=in_features, 
                hidden_features=hidden_features, 
                kernel_size=kernel_size, 
                dilation_rate=dilation_rate, 
                num_layers=num_layers,
                dropout=dropout,
                mean_only=mean_only,
            ))
        self.flows = flows
        
    def forward(self, x, x_mask):
        for flow in self.flows:
            x, _ = flow(x, x_mask, reverse=False)
            x = torch.flip(x, [1])
        return x

    def reverse(self, x, x_mask):
        for flow in reversed(self.flows):
            x = torch.flip(x, [1])
            x = flow(x, x_mask, reverse=True)
        return x
    
class VITS(nn.Module):
    def __init__(
        self,
        
    ):
        super(VITS, self).__init__()
        self.prior_encoder = PriorEncoder()
        self.posterior_encoder = PosteriorEncoder()
        
        if sdp:
            self.duration_predictor = StochasticDurationPredictor()
        else:
            self.duration_predictor = DurationPredictor(
                in_features=in_features,
                hidden_features=hidden_features,
                kernel_size=duration_predictor_kernel_size,
                dropout=duration_predictor_dropout,
            )
        
        self.flow_decoder = ResidualCouplingBlock()

    def mas(self, mu_p, log_s_p, z_p, mu_p, x_mask, y_mask):
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * log_s_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - log_s_p, [1], keepdim=True) 
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (mu_p * s_p_sq_r)) 
            neg_cent4 = torch.sum(-0.5 * (mu_p ** 2) * s_p_sq_r, [1], keepdim=True) 
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            attn = None
        return attn
    
    def forward(self, inputs):
        (x, x_lengths, y, y_lengths)= inputs
        
        x_mask = None
        y_mask = None
        x, mu_p, log_s_p, x_mask = self.text_encoder(x, x_lengths)
        z, mu_q, log_s_q, y_mask = self.posterior_encoder(y, y_lengths)
        z_p = self.flow(z, y_mask)
        _ = self.mas(mu_p, log_s_p, z_p, mu_p, x_mask, y_mask)                   
        return

    def infer(self, inputs):
        return