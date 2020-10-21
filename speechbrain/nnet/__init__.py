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
    BCE_loss,
    nll_loss,
    kldiv_loss,
    compute_masked_loss,
)
from .loss.stoi_loss import stoi_loss
from .normalization import (
    BatchNorm1d,
    BatchNorm2d,
    LayerNorm,
    InstanceNorm1d,
    InstanceNorm2d,
)
from .pooling import Pooling1d, Pooling2d, StatisticsPooling, AdaptivePool
from .schedulers import (
    update_learning_rate,
    NewBobScheduler,
    LinearScheduler,
    StepScheduler,
    NoamScheduler,
)
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
    "BCE_loss",
    "nll_loss",
    "kldiv_loss",
    "compute_masked_loss",
    "stoi_loss",
    "update_learning_rate",
    "NewBobScheduler",
    "LinearScheduler",
    "StepScheduler",
    "NoamScheduler",
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
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
