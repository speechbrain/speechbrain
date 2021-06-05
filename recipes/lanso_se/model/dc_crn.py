import torch
import torch.nn as nn
import os
import sys
# from .show import show_params, show_model
import torch.nn.functional as F
from .conv_stft import ConvSTFT, ConviSTFT 

from .complexnn import ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm



class DCCRN(nn.Module):

    def __init__(
                    self, 
                    rnn_layers=2,
                    rnn_units=128,
                    win_len=400,
                    win_inc=100, 
                    fft_len=512,
                    win_type='hanning',
                    masking_mode='E',
                    use_clstm=False,
                    use_cbn = False,
                    kernel_size=5,
                    kernel_num=[16,32,64,128,256,256]
                ):
        ''' 
            
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag

        '''

        super(DCCRN, self).__init__()

        # for fft 
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type 

        input_dim = win_len
        output_dim = win_len
        
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        #self.kernel_num = [2, 8, 16, 32, 128, 128, 128]
        #self.kernel_num = [2, 16, 32, 64, 128, 256, 256]
        self.kernel_num = [2]+kernel_num 
        self.masking_mode = masking_mode
        self.use_clstm = use_clstm
        
        #bidirectional=True
        bidirectional=False
        fac = 2 if bidirectional else 1 


        fix=True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num)-1):
            self.encoder.append(
                nn.Sequential(
                    #nn.ConstantPad2d([0, 0, 0, 0], 0),
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx+1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx+1]) if not use_cbn else ComplexBatchNorm(self.kernel_num[idx+1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len//(2**(len(self.kernel_num))) 

        if self.use_clstm: 
            rnns = []
            for idx in range(rnn_layers):
                rnns.append(
                        NavieComplexLSTM(
                        input_size= hidden_dim*self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim= hidden_dim*self.kernel_num[-1] if idx == rnn_layers-1 else None,
                        )
                    )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                    input_size= hidden_dim*self.kernel_num[-1],
                    hidden_size=self.rnn_units,
                    num_layers=2,
                    dropout=0.0,
                    bidirectional=bidirectional,
                    batch_first=False
            )
            self.tranform = nn.Linear(self.rnn_units * fac, hidden_dim*self.kernel_num[-1])

        for idx in range(len(self.kernel_num)-1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                        self.kernel_num[idx]*2,
                        self.kernel_num[idx-1],
                        kernel_size =(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2,0),
                        output_padding=(1,0)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx-1]) if not use_cbn else ComplexBatchNorm(self.kernel_num[idx-1]),
                    #nn.ELU()
                    nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                        self.kernel_num[idx]*2,
                        self.kernel_num[idx-1],
                        kernel_size =(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2,0),
                        output_padding=(1,0)
                    ),
                    )
                )
        
        show_model(self)
        show_params(self)
        self.flatten_parameters() 

    def flatten_parameters(self): 
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, inputs, lens=None):
        specs = self.stft(inputs)
        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]
        spec_mags = torch.sqrt(real**2+imag**2+1e-8)
        spec_mags = spec_mags
        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real,imag],1)
        cspecs = cspecs[:,:,1:]
        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        ''' 

        out = cspecs
        encoder_out = []
        
        for idx, layer in enumerate(self.encoder):
            out = layer(out)
        #    print('encoder', out.size())
            encoder_out.append(out)
        
        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)
        if self.use_clstm:
            r_rnn_in = out[:,:,:channels//2]
            i_rnn_in = out[:,:,channels//2:]
            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels//2*dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels//2*dims])
        
            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels//2, dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels//2, dims]) 
            out = torch.cat([r_rnn_in, i_rnn_in],2)
        
        else:
            # to [L, B, C, D]
            out = torch.reshape(out, [lengths, batch_size, channels*dims])
            out, _ = self.enhance(out)
            out = self.tranform(out)
            out = torch.reshape(out, [lengths, batch_size, channels, dims])
       
        out = out.permute(1, 2, 3, 0)
        
        for idx in range(len(self.decoder)):
            out = complex_cat([out,encoder_out[-1 - idx]],1)
            out = self.decoder[idx](out)
            out = out[...,1:]
        #    print('decoder', out.size())
        mask_real = out[:,0]
        mask_imag = out[:,1] 
        mask_real = F.pad(mask_real, [0,0,1,0])
        mask_imag = F.pad(mask_imag, [0,0,1,0])
        
        if self.masking_mode == 'E' :
            mask_mags = (mask_real**2+mask_imag**2)**0.5
            real_phase = mask_real/(mask_mags+1e-8)
            imag_phase = mask_imag/(mask_mags+1e-8)
            mask_phase = torch.atan2(
                            imag_phase,
                            real_phase
                        ) 

            #mask_mags = torch.clamp_(mask_mags,0,100) 
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags*spec_mags 
            est_phase = spec_phase + mask_phase
            real = est_mags*torch.cos(est_phase)
            imag = est_mags*torch.sin(est_phase) 
        elif self.masking_mode == 'C':
            real,imag = real*mask_real-imag*mask_imag, real*mask_imag+imag*mask_real
        elif self.masking_mode == 'R':
            real, imag = real*mask_real, imag*mask_imag 
        
        out_spec = torch.cat([real, imag], 1) 
        out_wav = self.istft(out_spec)
         
        out_wav = torch.squeeze(out_wav, 1)
        #out_wav = torch.tanh(out_wav)
        out_wav = torch.clamp_(out_wav,-1,1)
        return out_spec,  out_wav

    def get_params(self, weight_decay=0.0):
            # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params

    def loss(self, inputs, labels, loss_mode='SI-SNR'):
       
        if loss_mode == 'MSE':
            b, d, t = inputs.shape 
            labels[:,0,:]=0
            labels[:,d//2,:]=0
            return F.mse_loss(inputs, labels, reduction='mean')*d

        elif loss_mode == 'SI-SNR':
            #return -torch.mean(si_snr(inputs, labels))
            return -(si_snr(inputs, labels))
        elif loss_mode == 'MAE':
            gth_spec, gth_phase = self.stft(labels) 
            b,d,t = inputs.shape 
            return torch.mean(torch.abs(inputs-gth_spec))*d

def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True) 
    data = data - mean
    return data
def l2_norm(s1, s2):
    #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    #norm = torch.norm(s1*s2, 1, keepdim=True)
    
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

def test_complex():
    torch.manual_seed(20)
    inputs = torch.randn(10,2,256,10)
    conv = ComplexConv2d(2,32,(3,1),(2,1),(1,0))
    tconv = ComplexConvTranspose2d(32,2,(3,1),(2,1),(1,0),(1,0))
    out = conv(inputs)
    print(out.shape)
    out = tconv(out)
    print(out.shape)

if __name__ == '__main__':
    torch.manual_seed(10)
    torch.autograd.set_detect_anomaly(True)
    inputs = torch.randn([10,16000*4]).clamp_(-1,1)
    labels = torch.randn([10,16000*4]).clamp_(-1,1)
    
    '''
    # DCCRN-E
    net = DCCRN(rnn_units=256,masking_mode='E')
    outputs = net(inputs)[1]
    loss = net.loss(outputs, labels, loss_mode='SI-SNR')
    print(loss)
    
    # DCCRN-R
    net = DCCRN(rnn_units=256,masking_mode='R')
    outputs = net(inputs)[1]
    loss = net.loss(outputs, labels, loss_mode='SI-SNR')
    print(loss)
    
    # DCCRN-C
    net = DCCRN(rnn_units=256,masking_mode='C')
    outputs = net(inputs)[1]
    loss = net.loss(outputs, labels, loss_mode='SI-SNR')
    print(loss)
    
    '''
    # DCCRN-CL
    net = DCCRN(rnn_units=256,masking_mode='E',use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256])
    outputs = net(inputs)[1]
    loss = net.loss(outputs, labels, loss_mode='SI-SNR')
    print(loss)
    print(outputs.shape)
