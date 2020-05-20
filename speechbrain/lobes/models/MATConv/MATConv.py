"""A multi-time scale atrous pooling (MATConv) layers to perform pooling on the input feature.

   Authors: Jianyuan Zhong 2020
"""
import os
import torch  # noqa: F401
import torch.nn as nn # noqa: F401
import torch.nn.functional as F # noqa: F401
from speechbrain.nnet.sequential import Sequential
from speechbrain.lobes.models.CRDNN import NeuralBlock


class MATConvModule(Sequential):
    """This is the module for astrous (dilated convolution) on time or time-frequency domain

    Arguments
    ---------
    overrides : mapping
        Additional parameters overriding the MATConv block parameters.
    
    MATConv Block Parameters
    ------------------------
        ..include:: MATConv_block1.yaml
    
    Example
    -------
    >>> import torch
    >>> model = MATConvModule()
    >>> input = torch.rand([10, 120, 60])
    >>> output = model(input, init_params=True)
    >>> output.shape
    torch.Size([10, 120, 128])
    """
    def __init__(self, overrides={}):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        matConv = NeuralBlock(
            block_index=1,
            param_file=os.path.join(current_dir, "MATConv_block1.yaml"),
            overrides=overrides
        )

        super().__init__(matConv)
        self._init_weight()

    # TODO: find ways to initialize weight
    def _init_weight(self):
        pass



class MATConvPool(nn.Module):
    """This model is the MATConv Pooling.
    It performs multi-resolutions Pooling on the input vectors 
    via first convolution layers with differient dilation rates and a
    global average pooling, Then project the outputs of above 
    layers to a single vector.

    Arguments
    ---------
    delations : list
        Delation delation rates to be used in MATConv modules
    
    Examples
    --------
    >>> import torch
    >>> model = MATConvPool()
    >>> input = torch.rand([10, 120, 60])
    >>> output = model(input, init_params=True)
    >>> output.shape
    torch.Size([10, 120, 128])
    """
    def __init__(
        self,
        dilations=[1, 6, 12, 18]
    ):
        super().__init__()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.matConv1 = MATConvModule()
        self.matConv2 = MATConvModule(
            overrides={'conv1': {'kernel_size': (3,), 'padding': (dilations[1],), 'dilation': (dilations[1],)}}
        )
        self.matConv3 = MATConvModule(
            overrides={'conv1': {'kernel_size': (3,), 'padding': (dilations[2],), 'dilation': (dilations[2],)}}
        )
        self.matConv4 = MATConvModule(
            overrides={'conv1': {'kernel_size': (3,), 'padding': (dilations[3],), 'dilation': (dilations[3],)}}
        )

        self.global_avg_pool = NeuralBlock(
            block_index=1,
            param_file=os.path.join(current_dir, 'global_avg_pool.yaml')
        )

        self.conv = NeuralBlock(
            block_index=1,
            param_file=os.path.join(current_dir, 'cnn_block.yaml')
        )

        self._init_weight()

    def forward(self, x, init_params=False):
        x1 = self.matConv1(x, init_params)
        x2 = self.matConv2(x, init_params)
        x3 = self.matConv3(x, init_params)
        x4 = self.matConv4(x, init_params)
        x5 = self.global_avg_pool(x, init_params)
        x5 = F.interpolate(x5, x4.size()[2:], mode='linear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=-1)
        x = self.conv(x, init_params)

        return x

     # TODO: find ways to initialize weight
    def _init_weight(self):
        pass
