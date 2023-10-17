"""VITS implementaion in the SpeechBrain.
Authors
* Sathvik Udupa 2023
"""

import torch.nn as nn
import math
import torch
import numpy as np
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    get_key_padding_mask
)
from speechbrain.lobes.models.FastSpeech2 import PositionalEmbedding

def mask_from_lengths(lengths):    
    max_length = torch.max(lengths)
    mask = torch.arange(max_length).to(lengths).view(1, -1)
    mask = mask < lengths.unsqueeze(1)
    return mask.unsqueeze(1)
    
class WN(nn.Module):
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
        assert hidden_features % 2 == 0, f'{hidden_features}'

        layers = torch.nn.ModuleList()
        res_skip_layers = torch.nn.ModuleList()
        
        for idx in range(num_layers):
            dilation = dilation_rate ** idx
            padding = int((kernel_size * dilation - dilation) / 2)
            layer = torch.nn.Conv1d(
                hidden_features, 
                2*hidden_features, 
                kernel_size=kernel_size,
                dilation=dilation, 
                padding=padding
            )
            # layer = torch.nn.utils.parametrizations.weight_norm(layer, name='weight') #for touch2.0
            layer = torch.nn.utils.weight_norm(layer, name='weight')
            layers.append(layer)
            
            res_skip_features = 2 * hidden_features
            if idx == num_layers - 1:
                res_skip_features = hidden_features
            
            res_skip_layer = torch.nn.Conv1d(hidden_features, res_skip_features, 1)
            # res_skip_layer = torch.nn.utils.parametrizations.weight_norm(res_skip_layer, name='weight')
            es_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
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
            x_in = self.layers[idx](x) * x_mask
            x_in = self.dropout(x_in)
            acts = self.fused_add_tanh_sigmoid_multiply(x_in, num_features_tensor)
            res_skip_acts = self.res_skip_layers[idx](acts)
            if idx != self.num_layers - 1:
                x = torch.add(x, res_skip_acts[:,:self.hidden_features,:]) * x_mask
                output = torch.add(output, res_skip_acts[:,self.hidden_features:,:])
            else:
                output = torch.add(output, res_skip_acts)
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
        
        pre_encoder = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        # pre_encoder = torch.nn.utils.parametrizations.weight_norm(pre_encoder)
        pre_encoder = torch.nn.utils.weight_norm(pre_encoder)
        post_encoder = nn.Conv1d(hidden_features, out_features*2, kernel_size=1)
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
         
    def forward(self, x, x_lengths):
        x_mask = mask_from_lengths(x_lengths)
        x = self.pre_encoder(x) * x_mask
        x = self.wavenet(x, x_mask)
        x = self.post_encoder(x) * x_mask
        mu, log_s = torch.split(x, self.out_features, dim=1)
        z = (mu + torch.randn_like(mu) * torch.exp(log_s)) 
        return z, mu, log_s, x_mask
    
class PriorEncoder(nn.Module):
    def __init__(
        self,
        num_tokens,
        padding_idx,
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
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            nhead=num_heads,
            d_ffn=ffn_features,
            d_model=hidden_features,
            dropout=dropout,
            attention_type=attention_type,
        )
        self.num_heads = num_heads
        self.token_embedding = nn.Embedding(num_tokens, hidden_features)
        nn.init.normal_(self.token_embedding.weight, 0.0, hidden_features**-0.5)
        self.post_encoder = nn.Conv1d(hidden_features, out_features*2, 1)
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.padding_idx = padding_idx
        self.sinusoidal_positional_embed_encoder = PositionalEmbedding(hidden_features)
        
    def forward(self, x, x_lengths):
        
        srcmask = get_key_padding_mask(x, pad_idx=self.padding_idx)
        
        
        x = self.token_embedding(x) * math.sqrt(self.hidden_features)
        srcmask_inverted = (~srcmask).unsqueeze(-1)
        pos_embed = self.sinusoidal_positional_embed_encoder(
            x.shape[1], srcmask, x.dtype
        )
        x = torch.add(x, pos_embed) * srcmask_inverted
        
        attn_mask = (
            srcmask.unsqueeze(-1)
            .repeat(self.num_heads, 1, x.shape[1])
            .permute(0, 2, 1)
            .bool()
        )
        
        x, _ = self.encoder(
            x, src_mask=attn_mask, src_key_padding_mask=srcmask
        )
        x = x * srcmask_inverted
        
        x_d = self.post_encoder(x.permute(0, 2, 1))
        mu, log_s = torch.split(x_d, self.out_features, dim=1)
        return x, mu, log_s, srcmask_inverted

class StochasticDurationPredictor(nn.Module):
    def __init__(self):
        super(StochasticDurationPredictor, self).__init__()
        raise NotImplementedError()

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
        self.norm1 = nn.LayerNorm(hidden_features)
        self.norm2 = nn.LayerNorm(hidden_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, x_mask):
        x = self.dropout(self.norm1(self.relu(self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)))) * x_mask
        x = self.dropout(self.norm2(self.relu(self.conv2(x.permute(0, 2, 1)).permute(0, 2, 1)))) * x_mask
        x = self.conv3(x.permute(0, 2, 1)).permute(0, 2, 1) * x_mask
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
            in_features=hidden_features,
            hidden_features=hidden_features,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.post_decoder = nn.Conv1d(hidden_features, (in_features // 2) * (2 - mean_only), 1)
        self.post_decoder.weight.data.zero_()
        self.post_decoder.bias.data.zero_()

    def forward(self, x, x_mask):
        x0, x1 = torch.split(x, [self.features] * 2, 1)
        h = self.pre_decoder(x0) 
        h = self.decoder(h, x_mask)
        x0_d = self.post_decoder(h) 
        if not self.mean_only:
            mu, log_s = torch.split(x0_d, [self.features] * 2, 1)
        else:
            mu = x0_d
            log_s = torch.zeros_like(mu)

        x1 = mu + x1 * torch.exp(log_s) 
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

        
class ResidualCouplingBlock(nn.Module):
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
            x, _ = flow(x, x_mask)
            x = torch.flip(x, [1])
        return x

    def reverse(self, x, x_mask):
        for flow in reversed(self.flows):
            x = torch.flip(x, [1])
            x = flow.reverse(x, x_mask)
        return x
    
class VITS(nn.Module):
    def __init__(
        self,
        num_tokens,
        padding_idx,
        prior_encoder_in_features,
        prior_encoder_out_features,
        prior_encoder_hidden_features,
        prior_encoder_ffn_features,
        prior_encoder_num_heads,
        prior_encoder_num_layers,
        prior_encoder_attn_type,
        prior_encoder_dropout,
        posterior_encoder_in_features, 
        posterior_encoder_hidden_features, 
        posterior_encoder_out_features, 
        WN_posterior_kernel_size, 
        WN_posterior_dilation_rate,
        WN_posterior_num_layers,
        posterior_encoder_dropout,
        duration_predictor_hidden_features,
        duration_predictor_kernel_size,
        duration_predictor_dropout,
        flow_hidden_features,
        WN_flow_kernel_size,
        WN_flow_dilation_rate,
        WN_flow_num_layers,
        flow_mean_only,
        num_flows,
        flow_dropout,
        sdp,
    ):
        super().__init__() 
        self.prior_encoder = PriorEncoder(
            num_tokens,
            padding_idx=padding_idx,
            in_features=prior_encoder_in_features,
            out_features=prior_encoder_out_features,
            hidden_features=prior_encoder_hidden_features,
            ffn_features=prior_encoder_ffn_features,
            num_heads=prior_encoder_num_heads,
            num_layers=prior_encoder_num_layers,
            attention_type=prior_encoder_attn_type,
            dropout=prior_encoder_dropout,
        )
        
        self.posterior_encoder = PosteriorEncoder(
            in_features=posterior_encoder_in_features, 
            hidden_features=posterior_encoder_hidden_features, 
            out_features=posterior_encoder_out_features, 
            kernel_size=WN_posterior_kernel_size, 
            dilation_rate=WN_posterior_dilation_rate,
            num_layers=WN_posterior_num_layers,
            dropout=posterior_encoder_dropout,
        )
        
        if sdp:
            self.duration_predictor = StochasticDurationPredictor()
            
        else:
            self.duration_predictor = DurationPredictor(
                in_features=prior_encoder_out_features,
                hidden_features=duration_predictor_hidden_features,
                kernel_size=duration_predictor_kernel_size,
                dropout=duration_predictor_dropout,
            )

        self.flow_decoder = ResidualCouplingBlock(
            in_features=posterior_encoder_out_features,
            hidden_features=flow_hidden_features,
            kernel_size=WN_flow_kernel_size,
            dilation_rate=WN_flow_dilation_rate,
            num_layers=WN_flow_num_layers,
            num_flows=num_flows,
            mean_only=flow_mean_only,
            dropout=flow_dropout
        )
        
        
        
    def mas(self, mu_p, log_s_p, z_p, x_mask, y_mask):
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * log_s_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - log_s_p, [1], keepdim=True) 
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (mu_p * s_p_sq_r)) 
            neg_cent4 = torch.sum(-0.5 * (mu_p ** 2) * s_p_sq_r, [1], keepdim=True) 
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            attn_mask = (torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)).squeeze().permute(0, 2, 1)
            neg_cent = neg_cent * attn_mask
            path = get_mas_path(
                    attn=neg_cent, 
                    mask=attn_mask,
                    device=neg_cent.device,
                    dtype=neg_cent.dtype,
                )
        return neg_cent, path
    
    def forward(self, inputs):
        (tokens, token_lengths, mels, mel_lengths) = inputs
        tokens, mu_p, log_s_p, token_mask = self.prior_encoder(tokens, token_lengths)
        z, mu_q, log_s_q, target_mask = self.posterior_encoder(mels, mel_lengths)
        z_p = self.flow_decoder(z, target_mask)
        attn, path = self.mas(mu_p, log_s_p, z_p, token_mask, target_mask)        
        predicted_durations = self.duration_predictor(tokens, token_mask)
        return predicted_durations, path, target_mask, z_p, log_s_p, mu_p, log_s_q

    def infer(self, inputs):
        return

def get_mas_path(attn, mask, device, dtype, max_neg_val=-np.inf):
    attn = attn.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(bool)
    bs, tx, ty = attn.shape
    direction = np.zeros((bs, tx, ty), dtype=np.int64)
    v = np.zeros((bs, tx), dtype=np.float32)
    x_range = np.arange(tx, dtype=np.float32).reshape(1, -1)
    for j in range(ty):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + attn[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    max_path = np.zeros(attn.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(bs)
    for j in range(ty-1, -1, -1):
        max_path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    max_path = max_path * mask.astype(np.float32)
    max_path = torch.from_numpy(max_path).to(device=device, dtype=dtype).squeeze()
    return max_path

class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    result: tuple
        a tuple of tensors to be used as inputs/targets
        (
            text_padded,
            dur_padded,
            input_lengths,
            mel_padded,
            output_lengths,
            len_x,
            labels,
            wavs
        )
    """

    # TODO: Make this more intuitive, use the pipeline
    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        Arguments
        ---------
        batch: list
            [text_normalized, mel_normalized]
        """
        # TODO: Remove for loops
        raw_batch = list(batch)
        for i in range(
            len(batch)
        ):  # the pipline return a dictionary wiht one elemnent
            batch[i] = batch[i]["mel_text_pair"]

        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]
        max_audio_len = max([len(x[1]) for x in batch])
        max_mel_len = max([x[2].shape[-1] for x in batch])
        
        text_padded = torch.LongTensor(len(batch), max_input_len)
        wavs_padded = torch.LongTensor(len(batch), max_audio_len)
        mel_padded = torch.LongTensor(len(batch), 80, max_mel_len)
        
        text_padded.zero_()
        wavs_padded.zero_()
        mel_padded.zero_()
        
        input_lengths = []
        mel_lengths = []
        wav_lengths = []
        raw_text = []
        wav_fnames = []
        
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text
            audio = batch[ids_sorted_decreasing[i]][1]
            wavs_padded[i, : audio.size(0)] = audio
            melspec = batch[ids_sorted_decreasing[i]][2]
            mel_padded[i, : , : melspec.size(1)] = melspec
            
            raw_text.append(batch[ids_sorted_decreasing[i]][3])
            wav_fnames.append(batch[ids_sorted_decreasing[i]][4])
            input_lengths.append(text.size(0))
            mel_lengths.append(melspec.size(1))
            wav_lengths.append(audio.size(0))
        
        input_lengths = torch.from_numpy(np.array(input_lengths))
        mel_lengths = torch.from_numpy(np.array(mel_lengths))
        wav_lengths = torch.from_numpy(np.array(wav_lengths))
        
        return (
            text_padded,
            input_lengths,
            mel_padded,
            mel_lengths,
            wavs_padded,
            wav_lengths,
            raw_text,
            wav_fnames
        )

class VITSLoss(nn.Module):
    def __init__(
        self,
        log_scale_durations,
        duration_loss_weight,
        kl_loss_weight,
        duration_loss_fn
    ):
        super().__init__()
        self.duration_loss_weight = duration_loss_weight
        self.kl_loss_weight = kl_loss_weight
        
        if duration_loss_fn == "L1":
            self.duration_loss = nn.L1Loss()
        elif duration_loss_fn == "MSE":
            self.duration_loss = nn.MSELoss()
        else:
            raise NotADirectoryError(f"'L1' and 'MSE' supported for Duration Loss")

    @staticmethod
    def calc_kl_loss(z_p, log_s_p, mu_p, log_s_q, target_mask):
        z_p = z_p.float()
        log_s_q = log_s_q.float()
        mu_p = mu_p.float()
        log_s_p = log_s_p.float()
        target_mask = target_mask.float()

        kl = log_s_p - log_s_q - 0.5
        kl += 0.5 * ((z_p - mu_p) ** 2) * torch.exp(-2.0 * log_s_p)
        kl = torch.sum(kl * target_mask)
        loss = kl / torch.sum(target_mask)
        return loss
    
    def forward(
        self,
        outputs,
        inputs,
    ):
        (   
            duration_predict, 
            duration_target,
            target_mask,
            z_p, 
            log_s_p,
            mu_p, 
            log_s_q, 
            
        ) = outputs
        
        # () = inputs
        
        losses = {}
        duration_target = duration_target.permute(0, 2, 1).sum(-1)
        duration_predict = duration_predict.squeeze()
        
        duration_loss = self.duration_loss(
            duration_predict, 
            duration_target
        ) 
        losses["duration_loss"] = duration_loss * self.duration_loss_weight
        
        kl_loss = self.calc_kl_loss(
            z_p=z_p,
            log_s_p=log_s_p,
            mu_p=mu_p,
            log_s_q=log_s_q,
            target_mask=target_mask,
        )
        losses["kl_loss"] = kl_loss * self.kl_loss_weight
        
        losses["total_loss"] = sum(losses.values())
        return losses
        

