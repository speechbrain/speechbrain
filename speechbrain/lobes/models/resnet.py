import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels=[513, 1024, 512, 256],
        out_channels=[1024, 512, 256, 128],
        downsample_scales=[1, 1, 1, 1],
        kernel_size=3,
        z_channels=128,
        dilation=True,
        leaky_relu=True,
        dropout=0.0,
        stack_kernel_size=3,
        stack_layers=2,
        nin_layers=0,
        stacks=[3, 3, 3, 3],
        use_weight_norm=True,
        use_causal_conv=False,
    ):
        super(ResNet, self).__init__()

        # check hyper parameters is valid
        assert not use_causal_conv, "Not supported yet."

        # add initial layer
        layers = []

        for (in_channel, out_channel, ds_scale, stack) in zip(
            in_channels, out_channels, downsample_scales, stacks
        ):

            if ds_scale == 1:
                _kernel_size = kernel_size
                _padding = (kernel_size - 1) // 2
                _stride = 1
            else:
                _kernel_size = ds_scale * 2
                _padding = ds_scale // 2 + ds_scale % 2
                _stride = ds_scale

            layers += [
                nn.Conv1d(
                    in_channel,
                    out_channel,
                    _kernel_size,
                    stride=_stride,
                    padding=_padding,
                )
            ]

            # add residual stack
            for j in range(stack):
                layers += [
                    Conv1d_Layernorm_LRelu_Residual(
                        kernel_size=stack_kernel_size,
                        channels=out_channel,
                        layers=stack_layers,
                        nin_layers=nin_layers,
                        dilation=2 ** j if dilation else 1,
                        leaky_relu=leaky_relu,
                        dropout=dropout,
                        use_causal_conv=use_causal_conv,
                    )
                ]

            layers += [
                nn.LeakyReLU(negative_slope=0.2) if leaky_relu else nn.ReLU(),
            ]

        # add final layer
        layers += [nn.Conv1d(out_channels[-1], z_channels, 1)]

        self.net = nn.Sequential(*layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, input):
        """Calculate forward propagation.
        Args:
            input (Tensor): Input tensor (B, in_channels, T).
        Returns:
            Tensor: Output tensor (B, out_channels, T).
        """
        return self.net(input)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                # m.weight.data.normal_(0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                # logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)


class ResSkipNet(nn.Module):
    def __init__(
        self,
        in_channels=[128, 256, 512, 1024],
        out_channels=[256, 512, 1024, 513],
        cond_channels=128,
        skip_channels=80,
        final_channels=80,
        kernel_size=5,
        dilation=True,
        stack_kernel_size=3,
        stacks=[3, 3, 3, 3],
        use_weight_norm=True,
        use_causal_conv=False,
        use_affine=True,
    ):
        super(ResSkipNet, self).__init__()

        # check hyper parameters is valid
        assert not use_causal_conv, "Not supported yet."

        # add initial layer
        layers = nn.ModuleList()

        for (in_channel, out_channel, stack) in zip(
            in_channels, out_channels, stacks
        ):
            layers += [
                nn.Conv1d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                )
            ]
            # add residual stack
            for j in range(stack):
                layers += [
                    Conv1d_Layernorm_GLU_ResSkip(
                        kernel_size=stack_kernel_size,
                        in_channels=out_channel,
                        cond_channels=cond_channels,
                        skip_channels=skip_channels,
                        dilation=2 ** j if dilation else 1,
                        dropout=0.0,
                        use_causal_conv=use_causal_conv,
                        use_affine=use_affine,
                    )
                ]

        # add final layer
        final_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, final_channels, 1),
        )

        self.layers = layers
        self.final_layer = final_layer

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, input, C=None):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Input tensor (B, cond_channels, ...).
        Returns:
            Tensor: Output tensor (B, out_channels, T).
        """
        # return self.decode(x)
        x = input
        x_out = 0.0
        if C.ndim == 3:
            c = C[:, :, :1]
        elif C.ndim == 2:
            c = C.unsqueeze(-1)
        for layer in self.layers:
            if isinstance(layer, Conv1d_Layernorm_GLU_ResSkip):
                x, x_skip = layer(x, c=c)
                x_out += x_skip
            else:
                x = layer(x)
        x = x_out * math.sqrt(1.0 / len(self.layers))
        x = self.final_layer(x)
        return x

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                # m.weight.data.normal_(0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                # logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)


class Conv1d_Layernorm_LRelu_Residual(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        channels=128,
        layers=2,
        nin_layers=0,
        dilation=1,
        leaky_relu=True,
        dropout=0.0,
        use_causal_conv=False,
    ):
        super(Conv1d_Layernorm_LRelu_Residual, self).__init__()

        self.use_causal_conv = use_causal_conv

        if not self.use_causal_conv:
            assert (
                kernel_size - 1
            ) % 2 == 0, "Not support even number kernel size."
            padding1 = (kernel_size - 1) // 2 * dilation
            padding2 = (kernel_size - 1) // 2
            self.total_padding = None
        else:
            padding1 = (kernel_size - 1) * dilation
            padding2 = kernel_size - 1
            self.total_padding = padding1 + padding2 * (layers - 1)

        stack = [
            nn.LeakyReLU(negative_slope=0.2) if leaky_relu else nn.ReLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                padding=padding1,
                bias=True,
            ),
            nn.GroupNorm(
                num_groups=1, num_channels=channels, eps=1e-5, affine=True
            ),
        ]
        for i in range(layers - 1):
            stack += [
                nn.LeakyReLU(negative_slope=0.2) if leaky_relu else nn.ReLU(),
                nn.Conv1d(
                    channels, channels, kernel_size, padding=padding2, bias=True
                ),
                nn.GroupNorm(
                    num_groups=1, num_channels=channels, eps=1e-5, affine=True
                ),
            ]
        for i in range(nin_layers):
            stack += [
                nn.LeakyReLU(negative_slope=0.2) if leaky_relu else nn.ReLU(),
                nn.Conv1d(channels, channels, 1, bias=True),
                nn.GroupNorm(
                    num_groups=1, num_channels=channels, eps=1e-5, affine=True
                ),
            ]

        if dropout > 0.0:
            stack += [nn.Dropout(p=dropout)]

        self.stack = nn.Sequential(*stack)

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, channels, T).
        Returns:
            Tensor: Output tensor (B, channels, T).
        """
        return (self.stack(c)[..., : c.size(-1)] + c) * math.sqrt(0.5)


class Conv1d_Layernorm_GLU_ResSkip(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        in_channels=128,
        cond_channels=128,
        skip_channels=80,
        dilation=0,
        dropout=0.0,
        use_causal_conv=False,
        use_affine=False,
    ):
        super(Conv1d_Layernorm_GLU_ResSkip, self).__init__()

        self.use_causal_conv = use_causal_conv
        self.use_affine = use_affine
        self.dropout = dropout

        if not self.use_causal_conv:
            assert (
                kernel_size - 1
            ) % 2 == 0, "Not support even number kernel size."
            padding = (kernel_size - 1) // 2 * dilation
            self.conv_in = nn.Conv1d(
                in_channels,
                in_channels * 2,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=True,
            )
            self.norm_layer = nn.GroupNorm(
                num_groups=2,
                num_channels=in_channels * 2,
                eps=1e-5,
                affine=True,
            )

        else:
            padding = (kernel_size - 1) * dilation
            self.conv_in = nn.Conv1d(
                in_channels,
                in_channels * 2,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=True,
            )
            self.norm_layer = None

        if cond_channels is not None and cond_channels > 0:
            self.conv_cond = nn.Conv1d(
                cond_channels,
                in_channels * 4 if use_affine else in_channels * 2,
                1,
                bias=True,
            )
        else:
            self.conv_cond = None
        self.res_skip_layers = nn.Conv1d(
            in_channels, in_channels + skip_channels, 1, bias=True
        )

        self.in_channels = in_channels

    def forward(self, x, c=None):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, cond_channels, 1).
        Returns:
            Tensor: Output tensor for skip connection (B, in_channels, T).
        """

        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_res = self.conv_in(x)
        if self.conv_cond and c is not None:
            x_c1 = 1.0
            x_c2 = self.conv_cond(c)
            if self.use_affine:
                x_c1 = x_c2[:, : self.in_channels * 2].exp()
                x_c2 = x_c2[:, self.in_channels * 2 :]
        else:
            x_c1 = 1.0
            x_c2 = 0.0

        if not self.use_causal_conv:
            x_res = self.norm_layer(x_c1 * x_res + x_c2)
        else:
            x_res = x_c1 * x_res[..., : x.size(-1)] + x_c2

        x_res_tanh = torch.tanh(x_res[:, : self.in_channels])
        x_res_sigmoid = torch.sigmoid(x_res[:, self.in_channels :])
        x_res = x_res_tanh * x_res_sigmoid

        x_res_skip = self.res_skip_layers(x_res)

        x = (x_res_skip[:, : self.in_channels, :] + x) * math.sqrt(0.5)
        x_skip = x_res_skip[:, self.in_channels :, :]

        return x, x_skip
