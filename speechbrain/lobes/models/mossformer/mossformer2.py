'''Library to support Mossformer2

Authors
* Shengkui Zhao 2024
* Jia Qi Yip 2024
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import PyTorchModelHubMixin

from .utils.one_path_flash_fsmn import Encoder, Decoder, Dual_Path_Model, SBFLASHBlock_DualA

def getCheckpoints(config_name):
    
    from huggingface_hub import hf_hub_download

    for file in ['encoder','decoder','masknet']:
        if not os.path.exists(f'./model_weights/{config_name}/{file}.ckpt'):
            print(f'downloading {file}.cpkt')
            hf_hub_download(repo_id=f'alibabasglab/{config_name}', filename=f'{file}.ckpt', local_dir=f'./model_weights/{config_name}')
            print(f'{file}.cpkt downloaded')
        else:
            print(f'{file}.cpkt already downloaded')

class Mossformer2Wrapper(nn.Module, PyTorchModelHubMixin):
    """The wrapper for the Mossformer2 model which combines the Encoder, Masknet and the Encoder
    https://arxiv.org/pdf/2312.11825v1.pdf 

    Example
    -----
    >>> model = Mossformer2Wrapper(config)
    >>> inp = torch.rand(1, 160)
    >>> result = model.forward(inp)
    >>> result.shape
    torch.Size([1, 160, 2])
    """

    def __init__(
        self,
        config: dict
    ):

        super(Mossformer2Wrapper, self).__init__()
        
        self.config_name = config["config_name"]
        print(f'{self.config_name} config loaded')

        self.encoder = Encoder(
            kernel_size=config['encoder_kernel_size'],
            out_channels=config['encoder_out_nchannels'],
            in_channels=config['encoder_in_nchannels'],
        )

        intra_model = SBFLASHBlock_DualA(
            num_layers=config['intra_numlayers'],
            d_model=config['encoder_out_nchannels'],
            nhead=config['intra_nhead'],
            d_ffn=config['intra_dffn'],
            dropout=config['intra_dropout'],
            use_positional_encoding=config['intra_use_positional'],
            norm_before=config['intra_norm_before'],
        )

        self.masknet = Dual_Path_Model(
            in_channels=config['encoder_out_nchannels'],
            out_channels=config['encoder_out_nchannels'],
            intra_model=intra_model,
            num_layers=config['masknet_numlayers'],
            norm=config['masknet_norm'],
            K=config['masknet_chunksize'],
            num_spks=config['masknet_numspks'],
            skip_around_intra=config['masknet_extraskipconnection'],
            linear_layer_after_inter_intra=config['masknet_useextralinearlayer'],
        )
        self.decoder = Decoder(
            in_channels=config['encoder_out_nchannels'],
            out_channels=config['encoder_in_nchannels'],
            kernel_size=config['encoder_kernel_size'],
            stride=config['encoder_kernel_size'] // 2,
            bias=False,
        )
        self.num_spks = config['masknet_numspks']
        self.sample_rate = config['sample_rate']

        # Set device to gpu if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        print(f'model initialised on {self.device}')
    
    @property
    def device(self):
        return next(self.parameters()).device

    def loadPretrained(self):
        if not os.path.isdir(f'./model_weights/{self.config_name}'):
            print("no checkpoints have been cached, getting them now...")
            getCheckpoints(self.config_name)

        #load the model checkpoints
        self.encoder.load_state_dict(torch.load(f'model_weights/{self.config_name}/encoder.ckpt', map_location=torch.device(self.device)))
        self.decoder.load_state_dict(torch.load(f'model_weights/{self.config_name}/decoder.ckpt', map_location=torch.device(self.device)))
        self.masknet.load_state_dict(torch.load(f'model_weights/{self.config_name}/masknet.ckpt', map_location=torch.device(self.device)))
    
    def inference(self, mix_file, output_dir):
        '''
        This is a helper function for inference on a single mixture file
        '''
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        test_mix, sample_rate = torchaudio.load(mix_file)
        
        if sample_rate != self.sample_rate:
            raise Exception(f'Sampling rate must be {self.sample_rate}')
        
        with torch.no_grad():
            est_source = self.forward(test_mix.to(self.device))

        #Normalization to prevent clipping during conversion to .wav file        
        est_source_norm = []
        for ns in range(self.num_spks):
            signal = est_source[0, :, ns]
            signal = signal / signal.abs().max()
            est_source_norm.append(signal.unsqueeze(1).unsqueeze(0))
        est_source = torch.cat(est_source_norm, 2)

        for ns in range(self.num_spks):
            torchaudio.save(
                f'{output_dir}/index{ns+1}.wav', est_source[..., ns].detach().cpu(), sample_rate
            )
        return "done"

    def forward(self, mix):
        """ Processes the input tensor x and returns an output tensor."""
        mix_w = self.encoder(mix)
        if self.config_name == "mossformer2-whamr-2spk":
            est_mask = self.masknet(mix_w)
            sep_h = est_mask
        else:
            est_mask = self.masknet(mix_w)
            mix_w = torch.stack([mix_w] * self.num_spks)
            sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source