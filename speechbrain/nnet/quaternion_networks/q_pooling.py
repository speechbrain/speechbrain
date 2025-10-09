"""Library implementing quaternion-valued max and average pooling layers.

Authors
 * Drew Wagner 2024
"""

import torch

import speechbrain as sb


class QPooling2d(sb.nnet.pooling.Pooling2d):
    """This class implements the quaternion average pooling and max pooling
    by magnitude as described in: "Geometric methods of perceptual organisation for
    computer vision", Altamirano G.

    Arguments
    ---------
    pool_type : str
        It is the type of pooling function to use ('avg','max').
    kernel_size : int
        It is the kernel size that defines the pooling dimension.
        For instance, kernel size=3,3 performs a 2D Pooling with a 3x3 kernel.
    pool_axis : tuple
        It is a list containing the axis that will be considered
        during pooling.
    ceil_mode : bool
        When True, will use ceil instead of floor to compute the output shape.
    padding : int
        It is the number of padding elements to apply.
    dilation : int
        Controls the dilation factor of pooling.
    stride : int
        It is the stride size.

    Example
    -------
    >>> pool = QPooling2d("max", (5, 3))
    >>> inputs = torch.rand(10, 15, 12)
    >>> output = pool(inputs)
    >>> output.shape
    torch.Size([10, 3, 4])
    """

    def __init__(
        self,
        pool_type,
        kernel_size,
        pool_axis=(1, 2),
        ceil_mode=False,
        padding=0,
        dilation=1,
        stride=None,
    ):
        super().__init__(
            pool_type,
            kernel_size,
            pool_axis=pool_axis,
            ceil_mode=ceil_mode,
            padding=padding,
            dilation=dilation,
            stride=stride,
        )

        if self.pool_type == "max":
            self.pool_layer.return_indices = True

    def forward(self, x):
        """Performs 2d pooling to the input tensor.

        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.

        Returns
        -------
        The pooled tensor.
        """
        x_r, x_i, x_j, x_k = torch.chunk(x, 4, dim=-1)

        if self.pool_type == "avg":
            # Perform average pooling over each of the components of the quaternion
            x_r = super().forward(x_r)
            x_i = super().forward(x_i)
            x_j = super().forward(x_j)
            x_k = super().forward(x_k)

        elif self.pool_type == "max":
            # Compute the magnitude of the quaternion
            m = x_r**2 + x_i**2 + x_j**2 + x_k**2

            # Add extra two dimension at the last two, and then swap the pool_axis to them
            # Example: pool_axis=[1,2]
            # [a,b,c,d] => [a,b,c,d,1,1]
            # [a,b,c,d,1,1] => [a,1,c,d,b,1]
            # [a,1,c,d,b,1] => [a,1,1,d,b,c]
            # [a,1,1,d,b,c] => [a,d,b,c]
            m = (
                m.unsqueeze(-1)
                .unsqueeze(-1)
                .transpose(-2, self.pool_axis[0])
                .transpose(-1, self.pool_axis[1])
                .squeeze(self.pool_axis[1])
                .squeeze(self.pool_axis[0])
            )

            # Perform max pooling of the magnitude, returning only the indices
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
            # Select the r, i, j & k components of the quaternion with the max magnitude
            x_r = x_r.flatten()[idx_flat].reshape(idx.shape)
            x_i = x_i.flatten()[idx_flat].reshape(idx.shape)
            x_j = x_j.flatten()[idx_flat].reshape(idx.shape)
            x_k = x_k.flatten()[idx_flat].reshape(idx.shape)

        return torch.concat((x_r, x_i, x_j, x_k), dim=-1)
