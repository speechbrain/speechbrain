"""Neural network layers
"""

from .activations import Softmax, GumbelSoftmax, Swish
from .attention import (
    ContentBasedAttention,
    LocationAwareAttention,
    MultiheadAttention,
    PositionalwiseFeedForward,
)
from .CNN import (
    SincConv,
    Conv1d,
    Conv2d,
    DepthwiseSeparableConv1d,
    DepthwiseSeparableConv2d,
)
from .containers import Sequential, ConnectBlocks
from .dropout import Dropout2d
from .embedding import Embedding
from .linear import Linear
from .losses import (
    transducer_loss,
    PitWrapper,
    ctc_loss,
    l1_loss,
    mse_loss,
    classification_error,
    nll_loss,
    kldiv_loss,
    compute_masked_loss,
)
from .loss.stoi_loss import stoi_loss
from .lr_schedulers import (
    NewBobLRScheduler,
    LinearLRScheduler,
    StepLRScheduler,
    NoamScheduler,
    CustomLRScheduler,
)
from .normalization import (
    BatchNorm1d,
    BatchNorm2d,
    LayerNorm,
    InstanceNorm1d,
    InstanceNorm2d,
)
from .optimizers import Optimizer
from .pooling import Pooling1d, Pooling2d, StatisticsPooling, AdaptivePool
from .RNN import RNN, LSTM, GRU, AttentionalRNNDecoder, LiGRU, QuasiRNN

# Complex networks
from .complex_networks.CNN import ComplexConv1d, ComplexConv2d
from .complex_networks.linear import ComplexLinear
from .complex_networks.normalization import ComplexBatchNorm, ComplexLayerNorm
from .complex_networks.RNN import ComplexRNN, ComplexLiGRU

__all__ = [
    "Softmax",
    "GumbelSoftmax",
    "Swish",
    "ContentBasedAttention",
    "LocationAwareAttention",
    "MultiheadAttention",
    "PositionalwiseFeedForward",
    "SincConv",
    "Conv1d",
    "Conv2d",
    "DepthwiseSeparableConv1d",
    "DepthwiseSeparableConv2d",
    "Sequential",
    "ConnectBlocks",
    "Dropout2d",
    "Embedding",
    "Linear",
    "transducer_loss",
    "PitWrapper",
    "ctc_loss",
    "l1_loss",
    "mse_loss",
    "classification_error",
    "nll_loss",
    "kldiv_loss",
    "compute_masked_loss",
    "stoi_loss",
    "NewBobLRScheduler",
    "LinearLRScheduler",
    "StepLRScheduler",
    "NoamScheduler",
    "CustomLRScheduler",
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "Optimizer",
    "Pooling1d",
    "Pooling2d",
    "StatisticsPooling",
    "AdaptivePool",
    "RNN",
    "LSTM",
    "GRU",
    "AttentionalRNNDecoder",
    "LiGRU",
    "QuasiRNN",
    "ComplexConv1d",
    "ComplexConv2d",
    "ComplexLinear",
    "ComplexBatchNorm",
    "ComplexLayerNorm",
    "ComplexRNN",
    "ComplexLiGRU",
]
