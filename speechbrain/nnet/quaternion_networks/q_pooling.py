import torch

import speechbrain as sb


class QPooling2d (sb.nnet.pooling.Pooling2d):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.pool_type == 'max':
            self.pool_layer.return_indices = True
    
    def forward(self, x):
        x_r, x_i, x_j, x_k = torch.chunk(x, 4, dim=-1)
        
        if self.pool_type == 'avg':
            x_r = super().forward(x_r)
            x_i = super().forward(x_i)
            x_j = super().forward(x_j)
            x_k = super().forward(x_k)
        
        elif self.pool_type == 'max':
            m = x_r**2 + x_i**2 + x_j**2 + x_k**2
            
            m = (
                m.unsqueeze(-1)
                .unsqueeze(-1)
                .transpose(-2, self.pool_axis[0])
                .transpose(-1, self.pool_axis[1])
                .squeeze(self.pool_axis[1])
                .squeeze(self.pool_axis[0])
            )
            _, idx = self.pool_layer(m)
            idx = (
            idx.unsqueeze(self.pool_axis[0])
                .unsqueeze(self.pool_axis[1])
                .transpose(-2, self.pool_axis[0])
                .transpose(-1, self.pool_axis[1])
                .squeeze(-1)
                .squeeze(-1)
            )
            idx_flat = idx.flatten()
            x_r = x_r.flatten()[idx_flat].reshape(idx.shape)
            x_i = x_i.flatten()[idx_flat].reshape(idx.shape)
            x_j = x_j.flatten()[idx_flat].reshape(idx.shape)
            x_k = x_k.flatten()[idx_flat].reshape(idx.shape)
            
        return torch.concat((x_r, x_i, x_j, x_k), dim=-1)
