"""Library implementing normalization.

Author
    Mirco Ravanelli 2020
"""
import torch
import torch.nn as nn


class normalize(nn.Module):
    """Normalizes the input tensors accoding to the specified normalization
    technique.

    Arguments
    ---------
    norm_type : str
         It is the type of normalization used.

         "batchnorm": it applies the standard batch normalization by
         normalizing mean and std of the input tensor over the  batch axis.

         "layernorm": it applies the standard layer normalization by
         normalizing mean and std of the input tensor over the neuron axis.

         "groupnorm": it applies group normalization over a mini-batch of
         inputs. See torch.nn documentation for more info.

         "instancenorm": it applies instance norm over a mini-batch of inputs.
         It is similar to layernorm, but different statistic for each channel
         are computed.

        "localresponsenorm": it applies local response normalization over an
        input signal composed of several input planes.See torch.nn
        documentation for more info.

    eps : float
        This value is added to std deviation estimationto improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    alpha : float
        Alpha factor for localresponsenorm.
    beta : float
        Beta factor for localresponsenorm.
    k : float
        It is the k factor for localresponsenorm.
    neigh_ch : int
        It is amount of neighbouring channels used for localresponse
        normalization.
    affine : bool
        When set to True, the affine parameters are learned.
    elementwise_affine : bool
        it is used for the layer normalization. If True, this module has
        learnable per-element affine parameters initialized to ones
        (for weights) and zeros (for biases).
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
        Set it to True for batch normalization and to False for instancenorm.
    num_groups : bool
        It is number of groups to separate the channels into for the group
        normalization.


    Example
    -------
    >>> normalize = normalize('batchnorm')
    >>> inputs = torch.rand(10, 50, 40)
    >>> normalize.init_params(inputs)
    >>> output=normalize(inputs)
    >>> output.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
        self,
        norm_type,
        eps=1e-05,
        momentum=0.1,
        alpha=0.0001,
        beta=0.75,
        k=1.0,
        affine=True,
        elementwise_affine=True,
        track_running_stats=True,
        num_groups=1,
        neigh_ch=2,
        output_folder=None,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.eps = eps
        self.momentum = momentum
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.affine = affine
        self.elementwise_affine = elementwise_affine
        self.track_running_stats = track_running_stats
        self.num_groups = num_groups
        self.neigh_ch = neigh_ch
        self.output_folder = output_folder
        self.reshape = False

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        if self.norm_type == "batchnorm":
            self.norm = self._batchnorm(first_input)

        if self.norm_type == "groupnorm":
            n_ch = first_input.shape[1]
            self.norm = torch.nn.GroupNorm(
                self.num_groups, n_ch, eps=self.eps, affine=self.affine
            )

        if self.norm_type == "instancenorm":
            self.norm = self._instancenorm(first_input)

        if self.norm_type == "layernorm":
            self.norm = torch.nn.LayerNorm(
                first_input.size()[1:-1],
                eps=self.eps,
                elementwise_affine=self.elementwise_affine,
            )

            self.reshape = True

        if self.norm_type == "localresponsenorm":
            self.norm = torch.nn.LocalResponseNorm(
                self.neigh_ch, alpha=self.alpha, beta=self.beta, k=self.k
            )

    def forward(self, x, init_params=False):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor
            input to transform linearly.
        """

        if init_params:
            self.init_params(x)

        x = x.transpose(-1, 1)
        x_n = self.norm(x)
        x_n = x_n.transpose(1, -1)

        return x_n

    def _batchnorm(self, first_input):
        """Initializes batch normalization. BatchNorm1d is used for 2d or 3d
        input vectors, while nn.BatchNorm2d is used for 4d inputs.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """

        fea_dim = first_input.shape[-1]

        # Based on the shape of the input tensor I can use 1D or 2D batchn
        if len(first_input.shape) <= 3:
            norm = nn.BatchNorm1d(
                fea_dim,
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )

        if len(first_input.shape) == 4:
            norm = nn.BatchNorm2d(
                fea_dim,
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )

        return norm.to(first_input.device)

    def _instancenorm(self, first_input):
        """Initializes instance normalization. InstanceNorm1d is used for 2d
        or or 3d input vectors, while InstanceNorm2d is used for 4d inputs.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """

        fea_dim = first_input.shape[1]

        # Use 1D or 2D based in input dimensionality
        if len(first_input.shape) == 3:
            norm = nn.InstanceNorm1d(
                fea_dim,
                eps=self.eps,
                momentum=self.momentum,
                track_running_stats=self.track_running_stats,
            )

        if len(first_input.shape) == 4:
            norm = nn.InstanceNorm2d(
                fea_dim,
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )

        return norm
