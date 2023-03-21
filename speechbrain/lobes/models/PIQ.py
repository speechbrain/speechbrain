"""This file implements the necessary classes and functions to implement Posthoc Interpretations via Quantization.

 Authors
 * Cem Subakan 2023
 * Francesco Paissan 2023
"""
import torch
import torch.nn as nn
from torch.autograd import Function


def get_irrelevant_regions(labels, K, num_classes, N_shared=5, stage="TRAIN"):
    """This class returns binary matrix that indicates the irrelevant regions in the VQ-dictionary given the labels array

    Arguments
    ---------
    labels : torch.tensor
        1 dimensional torch.tensor of size [B]
    K : int
        Number of keys in the dictionary
    num_classes : int
        Number of possible classes
    N_shared : int
        Number of shared keys
    stage : str
        "TRAIN" or else

    Example:
    --------
    >>> labels = torch.Tensor([1, 0, 2])
    >>> irrelevant_regions = get_irrelevant_regions(labels, 20, 3, 5)
    >>> print(irrelevant_regions.shape)
    torch.Size([3, 20])
    """

    uniform_mat = torch.round(
        torch.linspace(-0.5, num_classes - 0.51, K - N_shared)
    ).to(labels.device)

    uniform_mat = uniform_mat.unsqueeze(0).repeat(labels.shape[0], 1)

    labels_expanded = labels.unsqueeze(1).repeat(1, K - N_shared)

    irrelevant_regions = uniform_mat != labels_expanded

    if stage == "TRAIN":
        irrelevant_regions = (
            torch.cat(
                [
                    irrelevant_regions,
                    torch.ones(irrelevant_regions.shape[0], N_shared).to(
                        labels.device
                    ),
                ],
                dim=1,
            )
            == 1
        )
    else:
        irrelevant_regions = (
            torch.cat(
                [
                    irrelevant_regions,
                    torch.zeros(irrelevant_regions.shape[0], N_shared).to(
                        labels.device
                    ),
                ],
                dim=1,
            )
            == 1
        )
    return irrelevant_regions


def weights_init(m):
    """
    Applies Xavier initialization to network weights.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VectorQuantization(Function):
    """This class defines the forward method for vector quantization. As VQ is not differentiable, it returns a RuntimeError in case `.grad()` is called. Refer to `VectorQuantizationStraightThrough` for a straight_through estimation of the gradient for the VQ operation."""

    @staticmethod
    def forward(
        ctx,
        inputs,
        codebook,
        labels=None,
        num_classes=10,
        activate_class_partitioning=True,
        shared_keys=10,
        training=True,
    ):
        """
        Applies VQ to vectors `input` with `codebook` as VQ dictionary.

        Arguments
        ---------
        inputs : torch.Tensor
            Hidden representations to quantize. Expected shape is `torch.Size([B, W, H, C])`.
        codebook : torch.Tensor
            VQ-dictionary for quantization. Expected shape of `torch.Size([K, C])` with K dictionary elements.
        labels : torch.Tensor
            Classification labels. Used to define irrelevant regions and divide the latent space based on predicted class. Shape should be `torch.Size([B])`.
        num_classes : int
            Number of possible classes
        activate_class_partitioning : bool
            `True` if latent space should be quantized for different classes.
        shared_keys : int
            Number of shared keys among classes.
        training : bool
            `True` if stage is TRAIN.

        Returns
        --------
        Codebook's indices for quantized representation : torch.Tensor

        Example:
        --------
        >>> inputs = torch.ones(3, 14, 25, 256)
        >>> codebook = torch.randn(1024, 256)
        >>> labels = torch.Tensor([1, 0, 2])
        >>> print(VectorQuantization.apply(inputs, codebook, labels).shape)
        torch.Size([3, 14, 25])
        """
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            labels_expanded = labels.reshape(-1, 1, 1).repeat(
                1, inputs_size[1], inputs_size[2]
            )
            labels_flatten = labels_expanded.reshape(-1)
            irrelevant_regions = get_irrelevant_regions(
                labels_flatten,
                codebook.shape[0],
                num_classes,
                N_shared=shared_keys,
                stage="TRAIN" if training else "VALID",
            )

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(
                codebook_sqr + inputs_sqr,
                inputs_flatten,
                codebook.t(),
                alpha=-2.0,
                beta=1.0,
            )

            # intervene and boost the distances for irrelevant codes
            if activate_class_partitioning:
                distances[irrelevant_regions] = torch.inf

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        """Handles error in case grad() is called on the VQ operation. """
        raise RuntimeError(
            "Trying to call `.grad()` on graph containing "
            "`VectorQuantization`. The function `VectorQuantization` "
            "is not differentiable. Use `VectorQuantizationStraightThrough` "
            "if you want a straight-through estimator of the gradient."
        )


class VectorQuantizationStraightThrough(Function):
    """This class defines the forward method for vector quantization. As VQ is not differentiable, it approximates the gradient of the VQ as in https://arxiv.org/abs/1711.00937."""

    @staticmethod
    def forward(
        ctx,
        inputs,
        codebook,
        labels=None,
        num_classes=10,
        activate_class_partitioning=True,
        shared_keys=10,
        training=True,
    ):
        """
        Applies VQ to vectors `input` with `codebook` as VQ dictionary and estimates gradients with a
        Straight-Through (id) approximation of the quantization steps.

        Arguments
        ---------
        inputs : torch.Tensor
            Hidden representations to quantize. Expected shape is `torch.Size([B, W, H, C])`.
        codebook : torch.Tensor
            VQ-dictionary for quantization. Expected shape of `torch.Size([K, C])` with K dictionary elements.
        labels : torch.Tensor
            Classification labels. Used to define irrelevant regions and divide the latent space based on predicted class. Shape should be `torch.Size([B])`.
        num_classes : int
            Number of possible classes
        activate_class_partitioning : bool
            `True` if latent space should be quantized for different classes.
        shared_keys : int
            Number of shared keys among classes.
        training : bool
            `True` if stage is TRAIN.

        Returns
        --------
        Quantized representation and codebook's indices for quantized representation : tuple

        Example:
        --------
        >>> inputs = torch.ones(3, 14, 25, 256)
        >>> codebook = torch.randn(1024, 256)
        >>> labels = torch.Tensor([1, 0, 2])
        >>> quant, quant_ind = VectorQuantizationStraightThrough.apply(inputs, codebook, labels)
        >>> print(quant.shape, quant_ind.shape)
        torch.Size([3, 14, 25, 256]) torch.Size([1050])
        """
        indices = VectorQuantization.apply(
            inputs,
            codebook,
            labels,
            num_classes,
            activate_class_partitioning,
            shared_keys,
            training,
        )
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(
            codebook, dim=0, index=indices_flatten
        )
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(
        ctx,
        grad_output,
        grad_indices,
        labels=None,
        num_classes=None,
        activate_class_partitioning=True,
        shared_keys=10,
        training=True,
    ):
        """
        Estimates gradient assuming vector quantization as identity function. (https://arxiv.org/abs/1711.00937)
        """
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = grad_output.contiguous().view(
                -1, embedding_size
            )
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook, None, None, None, None, None)


class Conv2dEncoder_v2(nn.Module):
    """
    This class implements a convolutional encoder to extract classification embeddings from logspectra.

    Arguments
    ---------
    dim : int
        Number of channels of the extracted embeddings.

    Returns
    --------
    Latent representations to feed inside classifier and/or intepreter.

    Example:
    --------
    >>> inputs = torch.ones(3, 431, 513)
    >>> model = Conv2dEncoder_v2()
    >>> print(model(inputs).shape)
    torch.Size([3, 256, 26, 32])
    """

    def __init__(self, dim=256):
        """
        Extracts embeddings from logspectrograms.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, dim, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, dim, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.conv4 = nn.Conv2d(dim, dim, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(dim)

        self.resblock = ResBlockAudio(dim)
        self.nonl = nn.ReLU()

    def forward(self, x):
        """
        Computes forward pass.
        Arguments
        --------
        x : torch.Tensor
            Log-power spectrogram. Expected shape `torch.Size([B, T, F])`.

        Returns
        --------
        Embeddings : torch.Tensor

        """
        x = x.unsqueeze(1)
        h1 = self.conv1(x)
        h1 = self.bn1(h1)
        h1 = self.nonl(h1)

        h2 = self.conv2(h1)
        h2 = self.bn2(h2)
        h2 = self.nonl(h2)

        h3 = self.conv3(h2)
        h3 = self.bn3(h3)
        h3 = self.nonl(h3)

        h4 = self.conv4(h3)
        h4 = self.bn4(h4)
        h4 = self.nonl(h4)

        h4 = self.resblock(h4)

        return h4


class ResBlockAudio(nn.Module):
    """This class implements a residual block.

    Arguments
    --------
    dim : int
    Input channels of the tensor to process. Matches output channels of the residual block.

    Returns
    --------
    Residual block output : torch.Tensor

    Example
    --------
    >>> res = ResBlockAudio(128)
    >>> x = torch.randn(2, 128, 16, 16)
    >>> print(x.shape)
    torch.Size([2, 128, 16, 16])
    """

    def __init__(self, dim):
        """Implements a residual block."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        """Forward step.

        Arguments
        --------
        x : torch.Tensor
            Tensor to process. Expected shape is `torch.Size([B, C, H, W])`.

        Returns
        --------
        Residual block output : torch.Tensor
        """
        return x + self.block(x)


class VectorQuantizedPSI_Audio(nn.Module):
    """
    This class reconstructs log-power spectrograms from classifier's representations.

    Arguments
    ---------
    dim : int
        Dimensionality of VQ vectors.
    K : int
        Number of elements of VQ dictionary.
    numclasses : int
        Number of possible classes
    activate_class_partitioning : bool
        `True` if latent space should be quantized for different classes.
    shared_keys : int
        Number of shared keys among classes.
    use_adapter : bool
        `True` to learn an adapter for classifier's representations.
    adapter_reduce_dim : bool
        `True` if adapter should compress representations.

    Returns
    --------
    Reconstructed log-power spectrograms, adapted classifier's representations, quantized classifier's representations. : tuple

    Example:
    --------
    >>> psi = VectorQuantizedPSI_Audio(dim=256, K=1024)
    >>> x = torch.randn(2, 256, 16, 16)
    >>> labels = torch.Tensor([0, 2])
    >>> logspectra, hcat, z_q_x = psi(x, labels)
    >>> print(logspectra.shape, hcat.shape, z_q_x.shape)
    torch.Size([2, 1, 257, 257]) torch.Size([2, 256, 8, 8]) torch.Size([2, 256, 8, 8])
    """

    def __init__(
        self,
        dim=128,
        K=512,
        numclasses=50,
        activate_class_partitioning=True,
        shared_keys=0,
        use_adapter=True,
        adapter_reduce_dim=True,
    ):
        super().__init__()
        self.codebook = VQEmbedding(
            K,
            dim,
            numclasses=numclasses,
            activate_class_partitioning=activate_class_partitioning,
            shared_keys=shared_keys,
        )
        self.use_adapter = use_adapter
        self.adapter_reduce_dim = adapter_reduce_dim
        if use_adapter:
            self.adapter = ResBlockAudio(dim)

            if adapter_reduce_dim:
                self.down = nn.Conv2d(dim, dim, 4, (2, 2), 1)
                self.up = nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 3, (2, 2), 1),
            nn.ReLU(True),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, 1, 12, 1, 1),
        )
        self.apply(weights_init)

    def forward(self, hs, labels):
        """
        Forward step. Reconstructs log-power based on provided label's keys in VQ dictionary.

        Arguments
        --------
        hs : torch.Tensor
            Classifier's representations.
        labels : torch.Tensor
            Predicted labels for classifier's representations.

        Returns
        --------
        Reconstructed log-power spectrogram, reduced classifier's representations and quantized classifier's representations. : tuple
        """
        if self.use_adapter:
            hcat = self.adapter(hs)
        else:
            hcat = hs

        if self.adapter_reduce_dim:
            hcat = self.down(hcat)
            z_q_x_st, z_q_x = self.codebook.straight_through(hcat, labels)
            z_q_x_st = self.up(z_q_x_st)
        else:
            z_q_x_st, z_q_x = self.codebook.straight_through(hcat, labels)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, hcat, z_q_x


class VQEmbedding(nn.Module):
    """
    Implements VQ Dictionary. Wraps `VectorQuantization` and `VectorQuantizationStraightThrough`. For more details refer to the specific class.

    Arguments
    ---------
    K : int
        Number of elements of VQ dictionary.
    D : int
        Dimensionality of VQ vectors.
    num_classes : int
        Number of possible classes
    activate_class_partitioning : bool
        `True` if latent space should be quantized for different classes.
    shared_keys : int
        Number of shared keys among classes.

    """

    def __init__(
        self,
        K,
        D,
        numclasses=50,
        activate_class_partitioning=True,
        shared_keys=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(K, D)

        self.embedding.weight.data.uniform_(-1.0 / K, 1.0 / K)

        self.numclasses = numclasses
        self.activate_class_partitioning = activate_class_partitioning
        self.shared_keys = shared_keys

    def forward(self, z_e_x, labels=None):
        """
        Wraps VectorQuantization. Computes VQ-dictionary indices for input quantization. Note that this forward step is not differentiable.

        Arguments
        ---------
        z_e_x : torch.Tensor
            Input tensor to be quantized.

        Returns
        --------
        Codebook's indices for quantized representation : torch.Tensor

        Example:
        --------
        >>> inputs = torch.ones(3, 256, 14, 25)
        >>> codebook = VQEmbedding(1024, 256)
        >>> labels = torch.Tensor([1, 0, 2])
        >>> print(codebook(inputs, labels).shape)
        torch.Size([3, 14, 25])
        """
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = VectorQuantization.apply(
            z_e_x_, self.embedding.weight, labels
        )
        return latents

    def straight_through(self, z_e_x, labels=None):
        """
        Implements the vector quantization with straight through approximation of the gradient.

        Arguments
        ---------
        z_e_x : torch.Tensor
            Input tensor to be quantized.
        labels : torch.Tensor
            Predicted class for input representations (used for latent space quantization).

        Returns
        --------
        Straigth through quantized representation and quantized representation : tuple

        Example:
        --------
        >>> inputs = torch.ones(3, 256, 14, 25)
        >>> codebook = VQEmbedding(1024, 256)
        >>> labels = torch.Tensor([1, 0, 2])
        >>> quant, quant_ind = codebook.straight_through(inputs, labels)
        >>> print(quant.shape, quant_ind.shape)
        torch.Size([3, 256, 14, 25]) torch.Size([3, 256, 14, 25])

        """
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = VectorQuantizationStraightThrough.apply(
            z_e_x_,
            self.embedding.weight.detach(),
            labels,
            self.numclasses,
            self.activate_class_partitioning,
            self.shared_keys,
            self.training,
        )
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(
            self.embedding.weight, dim=0, index=indices
        )
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar
